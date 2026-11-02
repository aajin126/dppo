"""Shared success replay, BC anchoring and evaluated-best policy retention."""

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from agent.finetune.firl.success import SuccessfulRolloutBuffer
from model.diffusion.diffusion import DiffusionModel


class PolicyRetention:
    def initialize_retention(self, cfg, saved=None):
        self.success_settings = self.settings.get("success_replay", {})
        self.success_replay = None
        self.best_evaluation = None
        self._success_bc_losses = []
        self._new_successes = 0
        if self.success_settings.get("enabled", False):
            if self.act_steps != self.horizon_steps:
                raise ValueError("Success BC currently requires act_steps == horizon_steps")
            if int(self.success_settings.batch_size) < 1 or int(self.success_settings.updates) < 0:
                raise ValueError("Success BC batch size must be positive and update count nonnegative")
            if int(self.success_settings.updates) and float(self.success_settings.coefficient) <= 0:
                raise ValueError("Success BC coefficient must be positive")
            self.success_replay = SuccessfulRolloutBuffer(
                self.retention_dir / "success", int(self.success_settings.capacity),
                (self.n_cond_step, self.obs_dim), (self.horizon_steps, self.action_dim),
                cfg.env_name, self.normalizer_hash,
            )
            replay_checkpoint = self.success_settings.get("checkpoint")
            replay_saved = (
                torch.load(replay_checkpoint, map_location="cpu", weights_only=True)
                if replay_checkpoint else saved
            )
            command_mode = getattr(self, "reward_metadata", {}).get("policy_gripper_command_memory")
            if (replay_saved and replay_saved.get("success_replay") is not None
                    and command_mode is not None
                    and replay_saved.get("policy_gripper_command_memory", "accepted_command") != command_mode):
                raise ValueError("Success replay uses different gripper command semantics")
            if replay_saved and replay_saved.get("success_replay") is not None:
                self.success_replay.load_state_dict(replay_saved["success_replay"])
            elif replay_checkpoint:
                raise ValueError("The requested checkpoint contains no success replay")

    def on_rollout_collected(self, observations, chains, rewards, episodes, eval_mode):
        self._success_bc_losses = []
        self._new_successes = 0
        outcomes = []
        for env_id, start, end in episodes:
            cumulative = np.cumsum(rewards[start:end + 1, env_id])
            outcomes.append(float(cumulative[-1]))
            successful = np.flatnonzero(cumulative >= self.best_reward_threshold_for_success)
            if len(successful) and self.success_replay is not None:
                stop = start + int(successful[0]) + 1
                self.success_replay.add(
                    observations[start:stop, env_id], chains[start:stop, env_id, -1],
                    dict(iteration=int(self.itr), env_id=int(env_id),
                         source="evaluation" if eval_mode else "training",
                         environment_return=float(cumulative[-1]),
                         success_chunk=int(successful[0])),
                )
                self._new_successes += 1
        if eval_mode and outcomes:
            self._maybe_save_best(outcomes)

    def _maybe_save_best(self, environment_returns):
        if not self.settings.get("save_best", True):
            return
        returns = np.asarray(environment_returns)
        if not np.isfinite(returns).all():
            raise ValueError("Nonfinite evaluation environment returns")
        score = (
            float(np.mean(returns >= self.best_reward_threshold_for_success)),
            float(returns.mean()),
        )
        previous = self.best_evaluation
        if previous is not None and score <= (previous["success_rate"], previous["mean_environment_return"]):
            return
        self.best_evaluation = dict(
            iteration=int(self.itr), success_rate=score[0],
            mean_environment_return=score[1], episodes=len(returns),
            environment_returns=returns.tolist(), criterion="evaluation assembly success rate, then environment return",
        )
        self.save_model()
        directory = Path(self.checkpoint_dir)
        prefix = getattr(self, "reward_checkpoint_prefix", "firl")
        for source, target in ((f"state_{self.itr}.pt", "best.pt"), (f"{prefix}_{self.itr}.pt", f"{prefix}_best.pt")):
            temporary = directory / f"{target}.tmp"
            shutil.copyfile(directory / source, temporary)
            temporary.replace(directory / target)
        (directory / "best.json").write_text(json.dumps(self.best_evaluation, indent=2) + "\n")
        print(f"[Best policy] {json.dumps(self.best_evaluation)}", flush=True)

    def auxiliary_policy_loss(self):
        if (
            self.success_replay is None or not len(self.success_replay)
            or self.itr < self.n_critic_warmup_itr
            or len(self._success_bc_losses) >= int(self.success_settings.updates)
        ):
            return 0.0
        observations, actions = self.sample_success_batch(int(self.success_settings.batch_size))
        # Reuse the original diffusion BC objective, targeting the trainable
        # denoiser only at noise levels used by its actual sampler.
        timesteps = (
            self.model.ddim_t[-self.model.ft_denoising_steps:]
            if self.model.use_ddim else
            torch.arange(self.model.ft_denoising_steps, device=self.device)
        )
        t = timesteps[torch.randint(len(timesteps), (len(actions),), device=self.device)]
        loss = DiffusionModel.p_losses(
            self.model, actions, {"state": observations}, t,
            network=self.model.actor_ft,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite success BC loss")
        self._success_bc_losses.append(float(loss.detach()))
        return float(self.success_settings.coefficient) * loss

    def sample_success_batch(self, batch_size):
        return self.success_replay.sample(batch_size, self.device)

    def additional_retention_state(self):
        return {}

    def on_iteration_end(self, eval_mode):
        statistics = dict(
            iteration=int(self.itr), evaluation=bool(eval_mode),
            new_successes=self._new_successes,
            replay_episodes=len(self.success_replay) if self.success_replay is not None else 0,
            success_bc_updates=len(self._success_bc_losses),
            success_bc_loss=float(np.mean(self._success_bc_losses)) if self._success_bc_losses else None,
            bc_anchor_enabled=bool(self.use_bc_loss), bc_anchor_coefficient=float(self.bc_loss_coeff),
        )
        with (self.retention_dir / "retention.jsonl").open("a") as stream:
            stream.write(json.dumps(statistics) + "\n")
        if self.use_wandb:
            import wandb
            wandb.log(
                {
                    f"retention/{key}": value
                    for key, value in statistics.items() if value is not None
                },
                step=self.itr,
                commit=False,
            )
        print(f"[Retention] {json.dumps(statistics)}", flush=True)

