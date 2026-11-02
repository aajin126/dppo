"""Official f-IRL JS reward updates with native DPPO as the inner policy solver.

This adaptation keeps the BC architecture, DDIM sampler and PPO implementation.
The original f-IRL reference uses maximum-entropy SAC; using DPPO here is an
approximate inner solver, not a reproduction of its SAC convergence guarantees.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
from agent.finetune.firl.dataset import FIRLExpertStates
from agent.finetune.firl.learner import FIRLRewardLearner
from agent.finetune.policy_retention import PolicyRetention


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TrainFIRLDiffusionAgent(PolicyRetention, TrainPPODiffusionAgent):
    def __init__(self, cfg):
        self.settings = cfg.firl
        if self.settings.get("reward_scope", "task") != "task":
            raise ValueError(
                "Skill labels were deleted. Restore reviewed annotations before "
                "requesting skill-conditioned reward training."
            )
        if cfg.env.n_envs < 2 and not self.settings.get("freeze_reward", False):
            raise ValueError("Use env.n_envs>=2 for f-IRL trajectory covariance")
        super().__init__(cfg)
        self.firl_dir = Path(self.logdir) / "firl"
        self.firl_dir.mkdir(parents=True, exist_ok=True)
        self.normalizer_hash = file_sha256(cfg.normalization_path)
        paired_normalizer = Path(self.settings.expert_dataset_path).with_name("normalization.pth")
        if not paired_normalizer.is_file() or file_sha256(paired_normalizer) != self.normalizer_hash:
            raise ValueError("Expert NPZ and environment must use the same paired normalization")
        self.expert = FIRLExpertStates(
            self.settings.expert_dataset_path, self.obs_dim,
            int(self.settings.get("max_expert_trajectories", 0)), self.seed,
        )
        devices = [torch.device(self.device).index or 0] if "cuda" in self.device else []
        # Initial BC evaluation should use the native random stream.
        with torch.random.fork_rng(devices=devices):
            self.reward_learner = FIRLRewardLearner(
                self.obs_dim, self.device, self.settings
            )
        checkpoint = self.settings.get("reward_checkpoint")
        saved = None
        if checkpoint:
            saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
            if saved["observation_dim"] != self.obs_dim:
                raise ValueError("Reward observation dimensions differ; a common feature adapter is required")
            if saved["normalization_sha256"] != self.normalizer_hash:
                raise ValueError("Reward normalization differs; transfer needs a shared feature normalization")
            self.reward_learner.load_state_dict(saved["learner"])
        elif self.settings.get("freeze_reward", False):
            raise ValueError("A frozen reward requires firl.reward_checkpoint")
        self.retention_dir = self.firl_dir
        self.reward_checkpoint_prefix = "firl"
        self.initialize_retention(cfg, saved)
        print(
            "[f-IRL] JS state-marginal reward learning; native BC/DDIM/DPPO. "
            "No action likelihood estimator. Reward scope: task.", flush=True,
        )


    def get_training_rewards(
        self, observations, next_observations, chains, environment_rewards, terminated
    ):
        # Original f-IRL samples state sequences after transitions. Preserve
        # trajectory membership instead of treating individual states as episodes.
        states = np.asarray(next_observations[:, :, -1], dtype=np.float32)
        trajectories = states.transpose(1, 0, 2).copy()
        terminal = np.asarray(terminated, dtype=bool)
        if terminal[:-1].any():
            raise ValueError("f-IRL requires fixed-horizon rollouts without internal episode resets")
        if self.settings.get("freeze_reward", False):
            statistics = dict(reward_frozen=True)
        else:
            statistics = self.reward_learner.update(self.expert.states, trajectories)
        rewards = self.reward_learner.predict(states.reshape(-1, self.obs_dim))
        if not np.isfinite(rewards).all():
            raise FloatingPointError("Nonfinite f-IRL training reward")
        status = dict(
            iteration=self.itr,
            method="f-IRL-JS with DPPO inner solver",
            reward_scope="task",
            uses_action_likelihood=False,
            environment_reward_used_for_policy_update=getattr(self, "success_replay", None) is not None,
            environment_reward_used_for_ppo_returns=False,
            environment_reward_used_for_success_selection=getattr(self, "success_replay", None) is not None,
            mean_learned_reward=float(rewards.mean()),
            **statistics,
        )
        with (self.firl_dir / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(status) + "\n")
        if self.use_wandb:
            import wandb
            wandb.log({f"firl/{key}": value for key, value in statistics.items()}, step=self.itr)
        print(f"[f-IRL] {json.dumps(status)}", flush=True)
        return rewards.reshape(environment_rewards.shape)

    def save_model(self):
        super().save_model()
        if not hasattr(self, "reward_learner"):
            return
        torch.save(dict(
            algorithm="f-IRL-JS with native DPPO inner solver",
            iteration=self.itr,
            observation_dim=self.obs_dim,
            feature_schema="normalized Furniture policy state, original ordering",
            normalization_sha256=self.normalizer_hash,
            learner=self.reward_learner.state_dict(),
            expert_episode_ids=self.expert.episode_ids,
            source_bc_checkpoint=self.cfg.base_policy_path,
            skill_conditioned=False,
            config=OmegaConf.to_container(self.settings, resolve=True),
            success_replay=self.success_replay.state_dict() if self.success_replay is not None else None,
            best_evaluation=self.best_evaluation,
        ), Path(self.checkpoint_dir) / f"firl_{self.itr}.pt")
