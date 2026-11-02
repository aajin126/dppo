"""Original imitation AIRL reward learning with the existing DPPO optimizer.

The BC architecture, parameters loaded from its checkpoint, DDIM sampler and
denoising PPO loss remain the existing implementation. A separately fitted
conditional density is qualified before either AIRL or PPO updates may run.
"""

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
from agent.finetune.airl.dataset import AIRLExpertSequences
from agent.finetune.airl.reward import DPPOAIRLReward
from agent.finetune.airl.validation import (
    DensityValidationError,
    ValidatedActionDensity,
    policy_fingerprint,
)


class TrainAIRLDiffusionAgent(TrainPPODiffusionAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.n_cond_step != 1 or self.act_steps != self.horizon_steps:
            raise ValueError("AIRL currently requires one observation and a full action chunk")
        self.settings = cfg.airl
        self.airl_dir = Path(self.logdir) / "airl"
        self.airl_dir.mkdir(parents=True, exist_ok=True)
        self.expert = AIRLExpertSequences(
            self.settings.expert_dataset_path, self.horizon_steps, self.obs_dim
        )
        self.discriminator_batch_size = int(self.settings.get("batch_size", 16))
        # Preserve the original initial BC evaluation random stream.
        devices = [torch.device(self.device).index or 0] if "cuda" in self.device else []
        with torch.random.fork_rng(devices=devices):
            self.airl = DPPOAIRLReward(
                self.obs_dim,
                (self.horizon_steps, self.action_dim),
                self.gamma,
                self.device,
                self.airl_dir / "discriminator",
                self.discriminator_batch_size,
            )
        reward_checkpoint = self.settings.get("reward_checkpoint")
        if reward_checkpoint:
            saved = torch.load(reward_checkpoint, map_location=self.device, weights_only=True)
            if saved["policy_observation_dim"] != self.obs_dim:
                raise ValueError("Reward checkpoint observation dimension differs")
            self.airl.reward_train.load_state_dict(saved["reward"], strict=True)
            self.airl._disc_opt.load_state_dict(saved["reward_optimizer"])
            self.airl._disc_step = int(saved["discriminator_updates"])
        self.density = None
        self.initial_policy_sha256 = policy_fingerprint(self.model)

    def _progress(self):
        if not self.venv.env.headless:
            self.venv.env.sim.render()

    def _fit_density(self, states, actions, query_groups=None):
        output = self.airl_dir / f"iteration_{self.itr:06d}"
        self.density = ValidatedActionDensity(self.settings.density, output)
        return self.density.fit_and_validate(
            self.model, states, actions, progress=self._progress,
            query_groups=query_groups,
        )

    def run(self):
        try:
            if self.settings.get("validate_only", False):
                count = int(self.settings.get("density_states_per_source", 16))
                queries = self.expert.sample(count, self.device)
                self._fit_density(
                    queries["obs"], queries["acts"], ["expert"] * count
                )
                self.save_model()
                print("[AIRL] Density qualification passed. No AIRL/PPO updates performed.", flush=True)
                return
            super().run()
        except DensityValidationError:
            # Preserve a reviewable state; a failed density is never applied to
            # the discriminator and cannot silently trigger a PPO update.
            self.save_model()
            raise
        finally:
            self.airl.close()

    def get_training_rewards(
        self, observations, next_observations, chains, environment_rewards, terminated
    ):
        shape = environment_rewards.shape
        state = torch.as_tensor(
            observations.reshape(-1, self.obs_dim), device=self.device, dtype=torch.float32
        )
        next_state = torch.as_tensor(
            next_observations.reshape(-1, self.obs_dim),
            device=self.device,
            dtype=torch.float32,
        )
        # Only the final action is read from the saved chain. Its path log
        # probability is never used as AIRL's final-action log probability.
        actions = torch.as_tensor(
            chains[:, :, -1].reshape(-1, self.horizon_steps, self.action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        eligible = torch.as_tensor(
            np.flatnonzero(~terminated.reshape(-1).astype(bool)), device=self.device
        )
        if not len(eligible):
            raise ValueError("No nonterminal policy transitions available for AIRL")
        count = min(int(self.settings.get("density_states_per_source", 32)), len(eligible))
        devices = [torch.device(self.device).index or 0] if "cuda" in self.device else []
        with torch.random.fork_rng(devices=devices):
            ids = eligible[torch.randperm(len(eligible), device=self.device)[:count]]
            expert = self.expert.sample(count, self.device)
            generated = dict(
                obs=state[ids],
                acts=actions[ids],
                next_obs=next_state[ids],
                dones=torch.zeros(count, device=self.device),
            )
            all_states = torch.cat([expert["obs"], generated["obs"]])
            all_actions = torch.cat([expert["acts"], generated["acts"]])
            report = self._fit_density(
                all_states, all_actions, ["expert"] * count + ["policy"] * count
            )
            # This line is unreachable when qualification fails.
            expert["logp"] = self.density.log_prob(expert["acts"], expert["obs"])
            generated["logp"] = self.density.log_prob(
                generated["acts"], generated["obs"]
            )
            self.airl._global_step = self.itr
            statistics = []
            for _ in range(int(self.settings.get("discriminator_updates", 4))):
                selections = torch.randint(
                    count, (self.discriminator_batch_size,), device=self.device
                )
                expert_batch = {name: value[selections] for name, value in expert.items()}
                gen_batch = {name: value[selections] for name, value in generated.items()}
                # Upstream AIRL train_disc, logits and BCE are used unchanged.
                statistics.append(self.airl.train_disc(
                    expert_samples=expert_batch, gen_samples=gen_batch
                ))
            with torch.no_grad():
                learned = self.airl.reward_train(
                    state,
                    actions,
                    next_state,
                    torch.as_tensor(
                        terminated.reshape(-1), device=self.device, dtype=torch.float32
                    ),
                )
            self.density.invalidate()
        if not torch.isfinite(learned).all():
            raise ValueError("AIRL produced a nonfinite training reward")
        status = dict(
            iteration=self.itr,
            discriminator_updates=self.airl._disc_step,
            environment_reward_used_for_policy_update=False,
            mean_learned_reward=float(learned.mean()),
            density_validation=report,
            discriminator_statistics=statistics,
        )
        with (self.airl_dir / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(status) + "\n")
        return learned.cpu().numpy().reshape(shape)

    def save_model(self):
        super().save_model()
        if not hasattr(self, "airl"):
            return
        torch.save(
            dict(
                algorithm="AIRL with a validated learned final-action density",
                iteration=self.itr,
                discriminator_updates=self.airl._disc_step,
                reward=self.airl.reward_train.state_dict(),
                transferable_base_reward=self.airl.reward_test.state_dict(),
                reward_optimizer=self.airl._disc_opt.state_dict(),
                density_report=self.density.report if self.density is not None else None,
                initial_policy_sha256=self.initial_policy_sha256,
                source_bc_checkpoint=self.cfg.base_policy_path,
                policy_observation_dim=self.obs_dim,
                skill_conditioned=False,
                task_context_conditioned=False,
                sampling_configuration=dict(
                    denoising_steps=self.model.denoising_steps,
                    ddim_steps=self.model.ddim_steps,
                    randn_clip_value=self.model.randn_clip_value,
                    min_sampling_std=self.model.get_min_sampling_denoising_std(),
                ),
                config=OmegaConf.to_container(self.settings, resolve=True),
            ),
            Path(self.checkpoint_dir) / f"airl_{self.itr}.pt",
        )
