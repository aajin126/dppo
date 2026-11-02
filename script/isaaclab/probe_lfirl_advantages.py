"""Native LFIRL value calibration with an explicit ban on actor updates."""

import json
from pathlib import Path

import numpy as np
import torch

from agent.finetune.train_lfirl_diffusion_agent import TrainLFIRLCurriculumAgent, TrainLFIRLDiffusionAgent


class NativeRewardCurriculumControl(TrainLFIRLCurriculumAgent):
    """Diagnostic control only: original environment reward, unchanged DPPO.

    This is not an LFIRL candidate and must never be reported as IRL improvement.
    Its purpose is to isolate the learned reward from policy/control failures.
    """

    def __init__(self, cfg):
        if not cfg.lfirl.get("diagnostic_native_reward_control", False):
            raise ValueError("Native-reward control requires its explicit diagnostic flag")
        self.training_reward_source = "native_environment_reward_control"
        super().__init__(cfg)
        self.reward_metadata["training_reward_source"] = self.training_reward_source
        metadata = dict(
            method="Original native assembly reward plus unchanged DPPO; diagnostic control",
            irl_policy_candidate=False, learned_reward_used_for_policy_training=False,
            policy_architecture_changed=False, native_reward_threshold_changed=False,
        )
        self.reward_metadata["diagnostic_control"] = metadata
        (self.retention_dir / "diagnostic_control.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("[Native reward control] Training uses the original assembly reward. This is not an IRL result.", flush=True)

    def get_training_rewards(self, observations, next_observations, chains, environment_rewards, terminated):
        from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent

        return TrainPPODiffusionAgent.get_training_rewards(
            self, observations, next_observations, chains, environment_rewards, terminated,
        )


class ProbeLFIRLAdvantages(TrainLFIRLDiffusionAgent):
    allow_actor_updates = False
    audit_policy_credit = False

    def __init__(self, cfg):
        if not self.allow_actor_updates and cfg.train.n_critic_warmup_itr < cfg.train.n_train_itr:
            raise ValueError("This probe must never permit actor updates")
        super().__init__(cfg)
        self.probe_directory = Path(self.logdir) / "mechanism"
        self.probe_directory.mkdir()
        self.original_actors = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
            if name.startswith(("actor.", "actor_ft."))
        }

    def reset_env_all(self, *args, **kwargs):
        observation = super().reset_env_all(*args, **kwargs)
        from script.isaaclab.check_lfirl_branches import measure

        env = self.venv.unwrapped
        self.probe_initial_leg = env._parts["square_table_leg4"].data.root_pos_w.clone()
        self.probe_initial = measure(env, self.probe_initial_leg)
        from script.isaaclab.evaluate_lfirl import physical_state_snapshot
        self.probe_initial_physics = physical_state_snapshot(env)
        self.probe_initial["grip_command"] = env.last_grasp.cpu().numpy().copy()
        self.probe_geometry = {key: [] for key in self.probe_initial}
        return observation

    def on_transition_collected(self, step, observations, actions, next_observations, info, eval_mode):
        super().on_transition_collected(step, observations, actions, next_observations, info, eval_mode)
        from script.isaaclab.check_lfirl_branches import measure

        env = self.venv.unwrapped
        values = measure(env, self.probe_initial_leg)
        values["grip_command"] = env.last_grasp.cpu().numpy().copy()
        for key, value in values.items():
            self.probe_geometry[key].append(value.copy())

    def on_rollout_collected(self, observations, chains, rewards, episodes, eval_mode):
        super().on_rollout_collected(observations, chains, rewards, episodes, eval_mode)
        self.probe_native_rewards = rewards.copy()

    def on_advantages_computed(self, observations, chains, advantages, returns, values, rewards):
        geometry = {key: np.stack(value) for key, value in self.probe_geometry.items()}
        previous = {key: np.concatenate((self.probe_initial[key][None], value[:-1]))
                    for key, value in geometry.items()}
        before_success = self.probe_native_rewards.cumsum(0) - self.probe_native_rewards
        held = ((previous["height"] > .06) & (previous["distance"] < .04)
                & (previous["width"] > .015) & (previous["width"] < .045)
                & (previous["grip_command"] > .019)
                & (before_success < 1) & (self.probe_native_rewards < 1))
        groups = {
            "held_kept_closed": held & (geometry["grip_command"] > .019),
            "held_opened_before_assembly": held & (geometry["grip_command"] < -.019),
        }
        valid = getattr(self, "training_validity", np.ones_like(advantages, dtype=bool))
        success = self.probe_native_rewards.sum(0) >= 1
        groups["successful_episode_prefix"] = valid & success[None]
        groups["failed_episode"] = valid & ~success[None]
        centered = (advantages - advantages[valid].mean()) / max(float(advantages[valid].std()), 1e-8)
        statistics = {}
        for key, mask in groups.items():
            statistics[key] = dict(transitions=int(mask.sum()), episodes=int(mask.any(0).sum()))
            if mask.any():
                statistics[key].update(
                    mean_native_GAE=float(advantages[mask].mean()),
                    mean_rollout_centered_GAE=float(centered[mask].mean()),
                    positive_centered_fraction=float((centered[mask] > 0).mean()),
                )
        result = dict(
            iteration=int(self.itr), groups=statistics,
            raw_GAE_mean=float(advantages[valid].mean()), raw_GAE_std=float(advantages[valid].std()),
            native_assembly_successes=int((self.probe_native_rewards.sum(0) >= 1).sum()),
            note="Actual native GAE; rollout centering is descriptive. DPPO normalizes within its actual minibatches, so these centered signs are not exact per-minibatch policy gradients.",
        )
        path = self.probe_directory / f"advantages_{self.itr}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(
            self.probe_directory / f"rollout_{self.itr}.npz",
            observations=observations.detach().cpu().numpy(),
            actions=chains[:, :, -1].astype(np.float32),
            advantages=advantages, returns=returns, values=values, scaled_rewards=rewards,
            assembly_rewards=self.probe_native_rewards,
            initial_physics=self.probe_initial_physics,
            **geometry,
        )
        print(f"[Value mechanism] {json.dumps(result)}", flush=True)
        if self.audit_policy_credit:
            self.audit_credit_gradients(observations, chains, advantages, valid, rewards, values)
        if self.allow_actor_updates and self.itr >= self.n_critic_warmup_itr:
            completed = statistics["successful_episode_prefix"]
            failed = statistics["failed_episode"]
            if (not completed["transitions"] or not failed["transitions"]
                    or completed["mean_native_GAE"] <= failed["mean_native_GAE"]):
                raise RuntimeError("Actor probe stopped: actual assembly-prefix credit did not exceed failure credit")

    def audit_credit_gradients(self, observations, chains, advantages, valid, rewards, values):
        """Compare actual on-policy gradient directions without optimizer steps."""
        from agent.finetune.lfirl.episodes import finite_episode_returns

        if not self.use_bc_loss:
            raise ValueError("This audit compares the enabled BC anchor with recovered-reward credit")
        if not (self.probe_native_rewards.sum(0) >= 1).any():
            raise RuntimeError("Gradient comparison requires a native successful episode")
        goal_returns = finite_episode_returns(self.probe_native_rewards, valid, self.gamma)
        finite_returns = finite_episode_returns(rewards * self.reward_scale_const, valid, self.gamma)
        finite_advantages = finite_returns - values
        indices = np.flatnonzero(valid.reshape(-1))
        states = observations.reshape(-1, *observations.shape[2:])
        paths = chains.reshape(-1, *chains.shape[2:])
        parameters = tuple(self.model.actor_ft.parameters())

        def gradient(loss):
            derivatives = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            return torch.cat([
                (value if value is not None else torch.zeros_like(parameter)).detach().flatten()
                for value, parameter in zip(derivatives, parameters)
            ])

        def cosine(left, right):
            return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-20))

        device = torch.device(self.device)
        cuda_devices = [device.index or 0] if device.type == "cuda" else []
        rows = []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(2026110181)
            for batch in range(3):
                total = len(indices) * self.model.ft_denoising_steps
                chosen = torch.randperm(total, device=self.device)[:self.batch_size]
                transition, denoising = torch.unravel_index(chosen, (len(indices), self.model.ft_denoising_steps))
                selected = indices[transition.cpu().numpy()]
                condition = {"state": states[selected]}
                path = torch.as_tensor(paths[selected], device=self.device).float()
                row = torch.arange(len(selected), device=self.device)
                previous, following_path = path[row, denoising], path[row, denoising + 1]
                with torch.no_grad():
                    old_logprobs = self.model.get_logprobs_subsample(
                        condition, previous, following_path, denoising,
                    )
                common = (condition, previous, following_path, denoising)
                zeros = torch.zeros(len(selected), device=self.device)
                learned = torch.as_tensor(advantages.reshape(-1)[selected], device=self.device).float()
                goal = torch.as_tensor(goal_returns.reshape(-1)[selected], device=self.device).float()
                finite = torch.as_tensor(finite_advantages.reshape(-1)[selected], device=self.device).float()
                np.savez_compressed(
                    self.probe_directory / f"credit_batch_{self.itr}_{batch}.npz",
                    states=condition["state"].detach().cpu().numpy(),
                    previous=previous.detach().cpu().numpy(),
                    following=following_path.detach().cpu().numpy(),
                    denoising=denoising.cpu().numpy(), old_logprobs=old_logprobs.cpu().numpy(),
                    learned_advantages=learned.cpu().numpy(), goal_returns=goal.cpu().numpy(),
                    finite_advantages=finite.cpu().numpy(),
                )
                losses = self.model.loss(
                    *common, zeros, zeros, learned, old_logprobs,
                    use_bc_loss=self.use_bc_loss, reward_horizon=self.reward_horizon,
                )
                recovered_gradient = gradient(losses[0])
                anchor_gradient = gradient(self.bc_loss_coeff * losses[6])
                target_losses = self.model.loss(
                    *common, zeros, zeros, goal, old_logprobs,
                    use_bc_loss=False, reward_horizon=self.reward_horizon,
                )
                goal_gradient = gradient(target_losses[0])
                finite_losses = self.model.loss(
                    *common, zeros, zeros, finite, old_logprobs,
                    use_bc_loss=False, reward_horizon=self.reward_horizon,
                )
                finite_gradient = gradient(finite_losses[0])
                rows.append(dict(
                    batch=batch, sampled_denoising_steps=len(selected),
                    nonzero_native_return_samples=int((goal != 0).sum()),
                    recovered_vs_native_cosine=cosine(recovered_gradient, goal_gradient),
                    anchor_vs_native_cosine=cosine(anchor_gradient, goal_gradient),
                    combined_vs_native_cosine=cosine(recovered_gradient + anchor_gradient, goal_gradient),
                    finite_return_vs_native_cosine=cosine(finite_gradient, goal_gradient),
                    finite_return_plus_anchor_vs_native_cosine=cosine(finite_gradient + anchor_gradient, goal_gradient),
                    recovered_gradient_norm=float(recovered_gradient.norm()),
                    weighted_anchor_gradient_norm=float(anchor_gradient.norm()),
                    native_goal_gradient_norm=float(goal_gradient.norm()),
                ))
                if not all(np.isfinite(value) for value in rows[-1].values()):
                    raise FloatingPointError("Nonfinite policy-credit gradient diagnostic")
        report = dict(
            iteration=int(self.itr), batches=rows, optimizer_steps_by_audit=0,
            native_successes=int((self.probe_native_rewards.sum(0) >= 1).sum()),
            finite_return_semantics="Counterfactual full observed episode returns; zero bootstrap beyond completion or the rollout budget; same learned rewards and current value baseline",
            caveat="Read-only counterfactual on actual sampled denoising paths. Sparse native discounted returns use a zero baseline; cosine alignment is diagnostic, not a policy improvement guarantee.",
        )
        (self.probe_directory / f"gradient_credit_{self.itr}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[Credit gradients] {json.dumps(report)}", flush=True)

    def on_iteration_end(self, eval_mode):
        super().on_iteration_end(eval_mode)
        current = self.model.state_dict()
        changed = [name for name, value in self.original_actors.items()
                   if not torch.equal(value, current[name].detach().cpu())]
        if (not self.allow_actor_updates or self.itr < self.n_critic_warmup_itr) and (changed or self.actor_optimizer.state):
            raise RuntimeError("Value-only mechanism probe changed the actor")
        actor_steps = max((float(state["step"]) for state in self.actor_optimizer.state.values()), default=0.)
        snapshot = dict(
            critic={name: value.detach().cpu() for name, value in self.model.critic.state_dict().items()},
            reward_scaler=self.running_reward_scaler.state_dict(),
            actor_weights_identical=not changed, actor_optimizer_steps=actor_steps,
            gamma=self.gamma, environments=self.n_envs,
        )
        torch.save(snapshot, self.probe_directory / f"value_{self.itr}.pt")
        (self.probe_directory / f"actor_guard_{self.itr}.json").write_text(
            json.dumps(dict(actor_weights_identical=not changed, actor_optimizer_steps=actor_steps)) + "\n"
        )


class ProbeOnePolicyIteration(ProbeLFIRLAdvantages):
    """One actor-update iteration, only after successful-prefix credit passes."""

    allow_actor_updates = True

    def __init__(self, cfg):
        if (cfg.train.n_train_itr - cfg.train.n_critic_warmup_itr != 1
                or cfg.train.n_train_itr > 4 or not cfg.train.get("force_train", False)
                or not cfg.lfirl.get("assembly_terminal", False)):
            raise ValueError("This probe permits exactly one forced actor-update iteration")
        super().__init__(cfg)


class AuditPolicyCredit(ProbeLFIRLAdvantages):
    """Keep the actor fixed while auditing the learned and native objectives."""

    audit_policy_credit = True
