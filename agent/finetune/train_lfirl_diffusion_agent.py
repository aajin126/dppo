"""Fine-tune native DPPO with a fixed reward recovered by official LFIRL."""

import json
from pathlib import Path

import numpy as np
import torch

from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
from agent.finetune.policy_retention import PolicyRetention
from agent.finetune.lfirl.reward import (
    LFIRLReward, discounted_action_rewards, sha256, validate_reward_provenance,
)


class TrainLFIRLDiffusionAgent(PolicyRetention, TrainPPODiffusionAgent):
    def __init__(self, cfg):
        checkpoint = Path(cfg.lfirl.reward_checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"LFIRL reward checkpoint missing: {checkpoint}. "
                "Run script/isaaclab/train_lfirl.py first. No f-IRL fallback is used."
            )
        self.settings = cfg.lfirl
        super().__init__(cfg)
        self.retention_dir = Path(self.logdir) / "lfirl"
        self.retention_dir.mkdir(parents=True, exist_ok=True)
        self.reward_checkpoint_prefix = "lfirl"
        self.normalizer_hash = sha256(cfg.normalization_path)
        self.source_bc_hash = sha256(cfg.base_policy_path)
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if saved.get("requires_calibration_refit"):
            raise ValueError(
                "Q-only experimental checkpoint: refit V/b/R before policy training."
            )
        if not saved.get("algorithm", "").startswith("LFIRL"):
            raise ValueError("Expected an LFIRL reward checkpoint")
        validate_reward_provenance(
            saved, self.normalizer_hash, self.source_bc_hash,
        )
        self.irl_action_horizon = saved["horizon_steps"]
        if ((saved["observation_dim"], saved["action_dim"]) != (self.obs_dim, self.action_dim)
                or self.horizon_steps != self.act_steps
                or self.irl_action_horizon not in (1, self.act_steps)):
            raise ValueError("LFIRL and policy observation/action dimensions differ")
        self.single_action_reward = self.irl_action_horizon == 1 and self.act_steps > 1
        self.reward_gamma = saved["gamma"]
        expected_gamma = self.reward_gamma ** self.act_steps if self.single_action_reward else self.reward_gamma
        if abs(expected_gamma - self.gamma) > 1e-9:
            raise ValueError(f"DPPO chunk discount must be {expected_gamma} for this reward time step")
        if self.single_action_reward:
            if self.n_cond_step != 1:
                raise ValueError("Single-action reward capture requires one observation step")
            self.venv.record_action_observations = True
        rng_devices = (
            [torch.device(self.device).index or 0] if "cuda" in self.device else []
        )
        with torch.random.fork_rng(devices=rng_devices):
            self.reward = LFIRLReward(
                saved["learner"]["settings"],
                self.obs_dim,
                self.irl_action_horizon * self.action_dim,
                self.device,
            )
            self.reward.load_state_dict(saved["learner"])
        self.reward_metadata = {
            key: value
            for key, value in saved.items()
            if key not in ("learner", "success_replay", "best_evaluation", "iteration", "training_state")
        }
        self.reward_metadata["policy_macro_gamma"] = self.gamma
        self.reward_metadata["policy_gripper_command_memory"] = (
            "upstream_raw_action" if getattr(self.venv.unwrapped, "upstream_gripper_memory", True)
            else "accepted_command"
        )
        self.preference_calibrated = bool(saved.get("preference_calibration"))
        self.assembly_terminal = bool(self.settings.get("assembly_terminal", False))
        self.reward_metadata["policy_assembly_terminal"] = self.assembly_terminal
        if self.assembly_terminal and (not self.reset_at_iteration or not self.furniture_sparse_reward
                or self.n_steps * self.act_steps > self.max_episode_steps + self.act_steps):
            raise ValueError("Assembly terminals require native sparse events and one freshly reset episode per rollout")
        self.initialize_retention(cfg, saved)
        if self.settings.get("resume_checkpoint"):
            self.resume_training(self.settings.resume_checkpoint)
        print(
            f"[{'LFIRL + preferences' if self.preference_calibrated else 'LFIRL'}] Fixed offline reward, horizon={self.irl_action_horizon}; "
            "native BC/DDIM/DPPO.",
            flush=True,
        )
        if saved.get("limited_training"):
            print(
                "[LFIRL] This reward checkpoint is from a limited integration run; "
                "task performance is unverified.",
                flush=True,
            )

    def resume_training(self, checkpoint):
        """Restore optimizer/scaler state when present; report legacy warm restarts."""
        checkpoint = Path(checkpoint).expanduser().resolve()
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        completed = int(saved["itr"])
        if completed < 0 or completed + 1 >= self.n_train_itr:
            raise ValueError("Resume checkpoint must precede the requested final iteration")
        companion = torch.load(
            checkpoint.with_name(f"lfirl_{completed}.pt"),
            map_location="cpu", weights_only=True,
        )
        if companion.get("training_reward_source", "learned_lfirl") != getattr(
                self, "training_reward_source", "learned_lfirl"):
            raise ValueError("Resume checkpoint uses a different policy-training reward source")
        if companion.get("preference_calibration") != getattr(self, "reward_metadata", {}).get("preference_calibration"):
            raise ValueError("Resume checkpoint uses a different preference-calibrated reward")
        if companion.get("policy_gripper_command_memory", "accepted_command") != getattr(
                self, "reward_metadata", {}).get("policy_gripper_command_memory", "accepted_command"):
            raise ValueError("Resume checkpoint uses different gripper command semantics")
        if bool(companion.get("policy_assembly_terminal", False)) != bool(getattr(self, "assembly_terminal", False)):
            raise ValueError("Resume checkpoint uses different assembly stopping semantics")
        if (companion["iteration"] != completed
                or companion["normalization_sha256"] != self.normalizer_hash
                or companion["gamma"] != self.reward_gamma
                or companion.get("source_bc_sha256") != self.source_bc_hash):
            raise ValueError("Policy and LFIRL resume checkpoints do not match")
        if (self.model.ft_denoising_steps_d or self.model.ft_denoising_steps_t
                or not isinstance(self.model.min_sampling_denoising_std, float)):
            raise ValueError("Legacy resume requires the original constant sampling schedule")
        self.model.load_state_dict(saved["model"], strict=True)
        self.reward.load_state_dict(companion["learner"])
        if self.success_replay is not None and companion.get("success_replay") is not None:
            self.success_replay.load_state_dict(companion["success_replay"])
        self.best_evaluation = companion.get("best_evaluation")
        self.itr = completed + 1
        training_state = companion.get("training_state")
        if training_state is not None:
            self.restore_training_state(training_state, completed)
        else:
            for iteration in range(self.itr):
                if iteration >= self.n_critic_warmup_itr:
                    self.actor_lr_scheduler.step()
                    if self.learn_eta:
                        self.eta_lr_scheduler.step()
                self.critic_lr_scheduler.step()
            self.model.ft_denoising_steps_cnt = self.itr
        report = dict(
            checkpoint=str(checkpoint), completed_iteration=completed,
            next_iteration=self.itr, final_iteration=self.n_train_itr - 1,
            optimizer_restored=training_state is not None,
            reward_scaler_restored=training_state is not None and self.reward_scale_running,
            environment_reset=True, exact_continuation=False,
        )
        (Path(self.logdir) / "resume.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[LFIRL resume] {json.dumps(report)}", flush=True)

    def capture_training_state(self):
        names = ("actor", "critic", "eta") if self.learn_eta else ("actor", "critic")
        return dict(
            completed_iteration=int(self.itr), environments=self.n_envs,
            optimizers={name: getattr(self, f"{name}_optimizer").state_dict() for name in names},
            schedulers={name: getattr(self, f"{name}_lr_scheduler").state_dict() for name in names},
            reward_scaler=(self.running_reward_scaler.state_dict() if self.reward_scale_running else None),
            model_steps=int(self.model.ft_denoising_steps_cnt),
        )

    def restore_training_state(self, saved, completed):
        names = ("actor", "critic", "eta") if self.learn_eta else ("actor", "critic")
        if (saved["completed_iteration"] != completed or saved["environments"] != self.n_envs
                or set(saved["optimizers"]) != set(names)
                or (saved["reward_scaler"] is not None) != self.reward_scale_running):
            raise ValueError("Saved optimizer/scaler training configuration differs")
        for name in names:
            getattr(self, f"{name}_optimizer").load_state_dict(saved["optimizers"][name])
            getattr(self, f"{name}_lr_scheduler").load_state_dict(saved["schedulers"][name])
        if self.reward_scale_running:
            self.running_reward_scaler.load_state_dict(saved["reward_scaler"])
        self.model.ft_denoising_steps_cnt = int(saved["model_steps"])

    def on_iteration_end(self, eval_mode):
        super().on_iteration_end(eval_mode)
        self._completed_iteration = self.itr

    def on_transition_collected(self, step, observations, actions, next_observations, info, eval_mode):
        if not self.single_action_reward or eval_mode:
            return
        if step == 0:
            self._action_reward_trajs = np.empty((self.n_steps, self.n_envs), dtype=np.float32)
        post_states = np.asarray(info["action_observations"])
        expected = (self.n_envs, self.act_steps, self.obs_dim)
        if post_states.shape != expected:
            raise ValueError(f"Expected recorded per-action states {expected}, got {post_states.shape}")
        pre_states = np.concatenate((observations["state"][:, -1:, :], post_states[:, :-1]), axis=1)
        values = self.reward.predict(
            pre_states.reshape(-1, 1, self.obs_dim),
            np.asarray(actions).reshape(-1, self.action_dim),
        ).reshape(self.n_envs, self.act_steps)
        self._action_reward_trajs[step] = discounted_action_rewards(values, self.reward_gamma)

    def get_training_rewards(
        self, observations, next_observations, chains, environment_rewards, terminated
    ):
        states = np.asarray(observations, dtype=np.float32)
        chains = np.asarray(chains)
        reward_shape = np.shape(environment_rewards)
        expected_states = (*reward_shape, self.n_cond_step, self.obs_dim)
        if len(reward_shape) != 2 or states.shape != expected_states:
            raise ValueError(f"Expected rollout states {expected_states}, got {states.shape}")
        if np.shape(next_observations) != expected_states or np.shape(terminated) != reward_shape:
            raise ValueError("Next states and terminal flags must align with the rollout")
        if (chains.ndim != 5 or chains.shape[:2] != reward_shape
                or chains.shape[-2:] != (self.horizon_steps, self.action_dim)):
            raise ValueError("Expected chains with aligned time/env axes and native horizon/action sizes")
        actions = np.asarray(chains[:, :, -1], dtype=np.float32)
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("LFIRL reward inputs contain nonfinite states or actions")
        if getattr(self, "single_action_reward", False):
            rewards = self._action_reward_trajs.copy()
        else:
            rewards = self.reward.predict(
                states.reshape(-1, self.n_cond_step, self.obs_dim),
                actions.reshape(-1, self.horizon_steps * self.action_dim),
            ).reshape(reward_shape)
        if rewards.shape != reward_shape:
            raise ValueError(f"Expected scalar rewards per transition {reward_shape}, got {rewards.shape}")
        if not np.isfinite(rewards).all():
            raise FloatingPointError("Nonfinite fixed LFIRL reward")
        metrics = dict(
            iteration=int(self.itr), method=("LFIRL + native-outcome preferences + native DPPO"
                                            if getattr(self, "preference_calibrated", False)
                                            else "LFIRL offline reward + native DPPO"),
            reward_fixed=True, mean_learned_reward=float(rewards.mean()),
            environment_reward_used_for_ppo_returns=False,
            reward_action_horizon=getattr(self, "irl_action_horizon", self.horizon_steps),
            success_replay_episodes=len(self.success_replay) if self.success_replay is not None else 0,
        )
        with (self.retention_dir / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(f"[LFIRL] {json.dumps(metrics)}", flush=True)
        return rewards

    def get_training_boundaries(self, environment_rewards):
        if not getattr(self, "assembly_terminal", False):
            self.training_validity = np.ones_like(environment_rewards, dtype=bool)
            return None
        from agent.finetune.lfirl.episodes import assembly_boundaries

        valid, terminal = assembly_boundaries(environment_rewards, self.best_reward_threshold_for_success)
        self.training_validity = valid
        statistics = dict(iteration=int(self.itr), completed=int(terminal.sum()),
                          valid_transitions=int(valid.sum()), discarded_post_assembly=int((~valid).sum()),
                          success_bonus_added=False)
        with (self.retention_dir / "assembly_boundaries.jsonl").open("a") as stream:
            stream.write(json.dumps(statistics) + "\n")
        return valid, terminal

    def save_model(self):
        super().save_model()
        if not hasattr(self, "reward"):
            return
        checkpoint = dict(
            **self.reward_metadata,
            iteration=int(self.itr),
            learner=self.reward.state_dict(),
            success_replay=(
                self.success_replay.state_dict()
                if self.success_replay is not None else None
            ),
            best_evaluation=self.best_evaluation,
            training_state=(self.capture_training_state()
                            if getattr(self, "_completed_iteration", None) == self.itr else None),
            **self.additional_retention_state(),
        )
        path = Path(self.checkpoint_dir) / f"lfirl_{self.itr}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)


class TrainLFIRLCurriculumAgent(TrainLFIRLDiffusionAgent):
    """Reuse native LFIRL/DPPO with a mixture of real pre-completion starts.

    This is an explicit reset-distribution extension, not original LFIRL.
    Policy, sampler, learned rewards, observations and native goals are unchanged.
    All reported policy evaluations must use ordinary full-task starts.
    """

    def __init__(self, cfg):
        self.curriculum_bank = None
        self.curriculum_success_replay = None
        super().__init__(cfg)
        settings = self.settings.curriculum
        if (not self.force_train or not self.reset_at_iteration
                or self.n_cond_step != 1 or self.obs_dim != 58):
            raise ValueError("Curriculum requires forced training, freshly reset rollouts and original 58D inputs")
        initial, final = float(settings.initial_fraction), float(settings.final_fraction)
        if not 0. < final <= initial < 1. or int(settings.ramp_actor_iterations) < 1:
            raise ValueError("Mix ordinary starts with a positive, decreasing curriculum fraction")
        from agent.finetune.lfirl.reset_bank import load_reset_bank

        env = self.venv.unwrapped
        roots = ["robot", *env._parts.keys(), *env._obstacles.keys()]
        self.curriculum_bank, report = load_reset_bank(
            settings.bank, self.normalizer_hash, self.source_bc_hash, roots, env.robot.num_joints,
        )
        accepted = [row for row in report["rows"] if row["accepted"]]
        if (report.get("mode") != "near_completion" or len(accepted) != len(self.curriculum_bank)
                or any(row.get("native_goal_already_satisfied", True)
                       or not row.get("recorded_continuation_native_success_verified") for row in accepted)):
            raise ValueError("Require real unassembled starts whose recorded continuation succeeds")
        diagnostic_path = Path(settings.training_sampler_diagnostic)
        diagnostic = json.loads(diagnostic_path.read_text())
        if (not diagnostic.get("completed") or diagnostic.get("native_sampler_mode") != "training"
                or diagnostic.get("bank_sha256") != report["bank_sha256"]
                or diagnostic["native_assembly_successes"] < int(settings.minimum_diagnostic_successes)):
            raise ValueError("The unchanged training sampler must show actual completion opportunities before curriculum training")
        self.curriculum_settings = settings
        self.curriculum_mask = np.zeros(self.n_envs, dtype=bool)
        self.curriculum_donor_ids = np.full(self.n_envs, -1, dtype=np.int64)
        self.curriculum_initial_actors = {
            key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()
            if key.startswith(("actor.", "actor_ft."))
        }
        self.probe_directory = self.retention_dir / "curriculum_credit"
        self.probe_directory.mkdir()
        metadata = dict(
            method="Real pre-completion reset curriculum; explicit LFIRL extension",
            bank=str(Path(settings.bank).resolve()), bank_sha256=report["bank_sha256"],
            initial_fraction=initial, final_fraction=final,
            ramp_actor_iterations=int(settings.ramp_actor_iterations),
            training_sampler_diagnostic=str(diagnostic_path.resolve()),
            training_sampler_diagnostic_sha256=sha256(diagnostic_path),
            policy_evaluation_requires_original_full_task_starts=True,
            success_replay="Ordinary full-task successes only; curriculum transitions train PPO on-policy",
            native_success_bonus_added=False, policy_architecture_changed=False,
        )
        self.curriculum_replay_fraction = float(settings.get("successful_replay_fraction", 0.))
        if self.curriculum_replay_fraction:
            if not 0. < self.curriculum_replay_fraction < 1. or self.success_replay is None:
                raise ValueError("Curriculum replay must share an enabled full-task buffer")
            from agent.finetune.firl.success import SuccessfulRolloutBuffer

            self.curriculum_success_replay = SuccessfulRolloutBuffer(
                self.retention_dir / "completion_success", int(self.success_settings.capacity),
                (self.n_cond_step, self.obs_dim), (self.horizon_steps, self.action_dim),
                cfg.env_name, self.normalizer_hash,
            )
            metadata["curriculum_success_replay_fraction"] = self.curriculum_replay_fraction
            metadata["success_replay"] = (
                "Separate real pre-completion and full-task success buffers; "
                "fixed batch mixture with the existing auxiliary diffusion BC objective"
            )
        self.reward_metadata["training_reset_curriculum"] = metadata
        if self.settings.get("resume_checkpoint"):
            policy_path = Path(self.settings.resume_checkpoint)
            companion = torch.load(
                policy_path.with_name(f"lfirl_{self.itr - 1}.pt"), map_location="cpu", weights_only=True,
            )
            if companion.get("training_reset_curriculum") != metadata:
                raise ValueError("Resume checkpoint uses a different training reset curriculum")
            if self.curriculum_success_replay is not None:
                if "curriculum_success_replay" not in companion:
                    raise ValueError("Resume checkpoint is missing curriculum success replay")
                self.curriculum_success_replay.load_state_dict(companion["curriculum_success_replay"])
        (self.retention_dir / "curriculum.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("[Curriculum] Training outcomes include restored starts; they are not full-task success rates.", flush=True)

    def reset_env_all(self, *args, **kwargs):
        observation = super().reset_env_all(*args, **kwargs)
        if self.curriculum_bank is None:
            return observation
        from agent.finetune.lfirl.reset_bank import mix_reset_snapshots
        from script.isaaclab.evaluate_lfirl import physical_state_snapshot, restore_reference

        settings = self.curriculum_settings
        actor_iterations = max(0, self.itr - self.n_critic_warmup_itr)
        fraction = float(settings.initial_fraction) + (
            float(settings.final_fraction) - float(settings.initial_fraction)
        ) * min(actor_iterations / int(settings.ramp_actor_iterations), 1.)
        indices = np.random.permutation(self.n_envs)[:round(self.n_envs * fraction)]
        donors = np.random.randint(len(self.curriculum_bank), size=len(indices))
        env = self.venv.unwrapped
        initial = physical_state_snapshot(env)
        mixed = mix_reset_snapshots(initial, self.curriculum_bank, indices, donors,
                                    env.robot.num_joints, 1 + len(env._parts) + len(env._obstacles))
        original_observation = observation
        observation = restore_reference(self, mixed)
        if float(np.abs(physical_state_snapshot(env) - mixed).max()) > 1e-5:
            raise RuntimeError("Curriculum restoration changed the recorded physical state")
        self.curriculum_mask[:] = False
        self.curriculum_mask[indices] = True
        # Preserve the original measurement for untouched ordinary starts too.
        observation["state"][~self.curriculum_mask] = original_observation["state"][~self.curriculum_mask]
        self.curriculum_donor_ids[:] = -1
        self.curriculum_donor_ids[indices] = donors
        self.curriculum_initial_physics = physical_state_snapshot(env)
        parts, founds = env._get_parts_poses(sim_coord=True)
        for index in indices:
            furniture = env.furnitures[int(index)]
            if any(furniture.is_assembled_idx(first, second, parts[index].cpu().numpy(),
                                             founds[index].cpu().numpy())
                   for first, second in furniture.should_be_assembled):
                raise RuntimeError("Curriculum start already satisfies the native assembly goal")
        return observation

    def on_rollout_collected(self, observations, chains, rewards, episodes, eval_mode):
        if eval_mode:
            raise RuntimeError("Evaluate curriculum checkpoints with original full-task starts in evaluate_lfirl.py")
        ordinary_episodes = [row for row in episodes if not self.curriculum_mask[row[0]]]
        # Keep the original successful pick/transport demonstrations in replay;
        # short restored episodes must not evict all full-task experience.
        super().on_rollout_collected(observations, chains, rewards, ordinary_episodes, eval_mode)
        if self.curriculum_success_replay is not None:
            for index, start, end in episodes:
                if not self.curriculum_mask[index]:
                    continue
                events = np.flatnonzero(np.cumsum(rewards[start:end + 1, index])
                                        >= self.best_reward_threshold_for_success)
                if len(events):
                    stop = start + int(events[0]) + 1
                    self.curriculum_success_replay.add(
                        observations[start:stop, index], chains[start:stop, index, -1],
                        dict(iteration=int(self.itr), env_id=int(index),
                             donor_id=int(self.curriculum_donor_ids[index]),
                             source="real_curriculum_training_success", success_chunk=int(events[0])),
                    )
        self.probe_native_rewards = np.asarray(rewards).copy()
        success = np.asarray(rewards).sum(0) >= self.best_reward_threshold_for_success
        statistics = dict(
            iteration=int(self.itr), restored_starts=int(self.curriculum_mask.sum()),
            restored_successes=int(success[self.curriculum_mask].sum()),
            ordinary_starts=int((~self.curriculum_mask).sum()),
            ordinary_successes=int(success[~self.curriculum_mask].sum()),
            evaluation=False,
        )
        with (self.retention_dir / "curriculum_outcomes.jsonl").open("a") as stream:
            stream.write(json.dumps(statistics) + "\n")
        print(f"[Curriculum training outcomes] {json.dumps(statistics)}", flush=True)

    def sample_success_batch(self, batch_size):
        if self.curriculum_success_replay is None or not len(self.curriculum_success_replay):
            return super().sample_success_batch(batch_size)
        local_size = round(batch_size * self.curriculum_replay_fraction)
        if not local_size:
            return super().sample_success_batch(batch_size)
        local = self.curriculum_success_replay.sample(local_size, self.device)
        if local_size == batch_size:
            return local
        full = super().sample_success_batch(batch_size - local_size)
        return tuple(torch.cat((ordinary, completion), dim=0)
                     for ordinary, completion in zip(full, local))

    def additional_retention_state(self):
        if self.curriculum_success_replay is None:
            return super().additional_retention_state()
        return dict(curriculum_success_replay=self.curriculum_success_replay.state_dict())

    def on_advantages_computed(self, observations, chains, advantages, returns, values, rewards):
        super().on_advantages_computed(observations, chains, advantages, returns, values, rewards)
        if (self.itr >= self.n_critic_warmup_itr
                and (self.itr - self.n_critic_warmup_itr + 1) % 5):
            return
        from script.isaaclab.probe_lfirl_advantages import AuditPolicyCredit

        valid = self.training_validity
        success = self.probe_native_rewards.sum(0) >= self.best_reward_threshold_for_success
        statistics = {}
        for label, starts in (("restored", self.curriculum_mask), ("ordinary", ~self.curriculum_mask)):
            for outcome, selected in (("successful", success), ("failed", ~success)):
                mask = valid & (starts & selected)[None]
                statistics[f"{label}_{outcome}"] = dict(
                    transitions=int(mask.sum()), episodes=int((starts & selected).sum()),
                    mean_GAE=float(advantages[mask].mean()) if mask.any() else None,
                )
        (self.probe_directory / f"advantages_{self.itr}.json").write_text(
            json.dumps(statistics, indent=2) + "\n"
        )
        np.savez_compressed(
            self.probe_directory / f"rollout_{self.itr}.npz",
            observations=observations.detach().cpu().numpy(), actions=chains[:, :, -1],
            assembly_rewards=self.probe_native_rewards, advantages=advantages,
            returns=returns, values=values, initial_physics=self.curriculum_initial_physics,
            donor_ids=self.curriculum_donor_ids,
        )
        # Reuse the existing read-only diagnostic, not a new optimization loss.
        AuditPolicyCredit.audit_credit_gradients(self, observations, chains, advantages,
                                                valid, rewards, values)

    def on_iteration_end(self, eval_mode):
        super().on_iteration_end(eval_mode)
        if self.itr < self.n_critic_warmup_itr:
            current = self.model.state_dict()
            for key, value in self.curriculum_initial_actors.items():
                if not torch.equal(value, current[key].detach().cpu()):
                    raise RuntimeError("Critic-only curriculum diagnostic changed the actor")
            if self.actor_optimizer.state:
                raise RuntimeError("Critic-only curriculum diagnostic updated the actor optimizer")
            (self.probe_directory / f"actor_guard_{self.itr}.json").write_text(
                json.dumps(dict(actors_unchanged=True, actor_optimizer_steps=0)) + "\n"
            )
