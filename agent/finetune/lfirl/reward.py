"""Official LFIRL stages, with native Furniture data and policy adapters."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from agent.finetune.lfirl import official as source


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_reward_provenance(saved, normalizer_hash, bc_hash):
    """Reject rewards fitted to a different teacher or inconsistent time scale."""
    if saved["normalization_sha256"] != normalizer_hash:
        raise ValueError("LFIRL reward normalization differs from the simulator")
    if saved.get("source_bc_sha256") != bc_hash:
        raise ValueError("LFIRL reward was fitted to a different BC checkpoint")
    fitted_gamma = float(saved["learner"]["settings"]["irl"]["gamma"])
    if abs(fitted_gamma - float(saved["gamma"])) > 1e-9:
        raise ValueError("LFIRL fitted discount and exported reward discount differ")


def discounted_action_rewards(rewards, gamma):
    """Combine per-action rewards into one correctly discounted policy chunk."""
    rewards = np.asarray(rewards)
    if rewards.ndim != 2:
        raise ValueError("Expected rewards with shape (environments, action steps)")
    return (rewards * np.power(gamma, np.arange(rewards.shape[1]))).sum(axis=1)


class LFIRLReward:
    def __init__(self, settings, observation_dim, action_dim, device):
        self.settings = settings
        self.device = torch.device(device)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        example = torch.zeros(1, 1, observation_dim, device=self.device)
        self.q = source.QNetwork.build_q_from_cfg(
            settings["q_net"], example, action_dim, self.device
        )
        self.b = source.VNetwork.build_v_from_cfg(settings["b_net"], example, self.device)
        self.v = source.VNetwork.build_v_from_cfg(settings["v_net"], example, self.device)
        self.r = source.RewardNetwork.build_reward_from_cfg(
            settings["r_net"], example, action_shape=action_dim, device=self.device,
        )
        relative = settings.get("relative_residual")
        if relative:
            from agent.finetune.lfirl.relative_reward import RelativeRewardResidual

            if relative.get("feature_contract") != "one_leg_tabletop_to_leg4_translation_squared_translation_row6d":
                raise ValueError("Unknown relative reward observation/feature contract")
            normalizer = Path(relative["normalization_path"])
            if sha256(normalizer) != relative["normalization_sha256"]:
                raise ValueError("Relative reward policy-normalization provenance differs")
            self.r = RelativeRewardResidual(
                self.r, torch.load(normalizer, map_location=self.device, weights_only=True),
            ).to(self.device)
        self.rms = source.RunningMeanStd()
        self.mean = torch.zeros(observation_dim, device=self.device)
        self.std = torch.ones(observation_dim, device=self.device)
        self.set_normalization(self.mean, self.std)
        self.target_states = None
        self.target_mode = "final_frozen"

    def set_normalization(self, mean, std):
        self.mean, self.std = mean.to(self.device), std.to(self.device)
        for network in (self.q, self.b, self.v, self.r):
            network.set_obs_normalizer(mean=self.mean, std=self.std)

    def reset_reward_head(self):
        """Fit newly calibrated Bellman targets without stale reward statistics."""
        if self.settings.get("relative_residual"):
            raise ValueError("Preserve the calibrated residual; reset the original LFIRL parent instead")
        example = torch.zeros(1, 1, self.observation_dim, device=self.device)
        self.r = source.RewardNetwork.build_reward_from_cfg(
            self.settings["r_net"], example, action_shape=self.action_dim,
            device=self.device,
        )
        self.r.set_obs_normalizer(mean=self.mean, std=self.std)
        self.rms = source.RunningMeanStd()

    def reset_calibration_heads(self):
        """Start original V/b/R calibration afresh while preserving fitted Q."""
        if self.settings.get("relative_residual"):
            raise ValueError("Preserve calibrated residuals; refit their LFIRL parent separately")
        example = torch.zeros(1, 1, self.observation_dim, device=self.device)
        self.b = source.VNetwork.build_v_from_cfg(
            self.settings["b_net"], example, self.device,
        )
        self.v = source.VNetwork.build_v_from_cfg(
            self.settings["v_net"], example, self.device,
        )
        for network in (self.b, self.v):
            network.set_obs_normalizer(mean=self.mean, std=self.std)
        self.reset_reward_head()
        self.target_states = None
        self.target_mode = "requires_calibration_refit"

    def fit(self, transitions, policy, output, max_batches_per_stage=0,
            start_stage="QHAT", checkpoint_interval=0, update_normalization=True,
            shuffle_reward=False, shuffle_calibration=False):
        if self.settings.get("relative_residual"):
            raise ValueError("Preference-calibrated residuals are frozen for policy training; refit the LFIRL parent separately")
        if (transitions.summary.get("reference_transitions", 0)
                and (start_stage not in ("V_INIT", "B", "V_CALIBRATED", "R")
                     or update_normalization)):
            raise ValueError("Unlabeled reference transitions require frozen Q and fixed input normalization")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if start_stage == "QHAT" and update_normalization:
            self.set_normalization(
                transitions.observations[:, 0].mean(0),
                transitions.observations[:, 0].std(0, unbiased=False).clamp_min(1e-6),
            )
        optimizers = {}
        for name, network in (("q", self.q), ("b", self.b), ("v", self.v), ("r", self.r)):
            options = self.settings[f"{name}_net"]
            optimizers[name] = torch.optim.AdamW(
                network.parameters(), lr=float(options["lr"]),
                weight_decay=float(options.get("weight_decay", 0.)),
            )
        q_target, b_target, v_target = deepcopy(self.q), deepcopy(self.b), deepcopy(self.v)
        self.history = []
        cfg = self.settings["irl"]
        gamma, tau = float(cfg["gamma"]), float(cfg["ema_tau"])
        bounds = dict(
            action_low=-np.ones(self.action_dim, dtype=np.float32),
            action_high=np.ones(self.action_dim, dtype=np.float32),
        )
        stages = (
            ("QHAT", cfg["qhat_passes"]), ("V_INIT", cfg["initial_v_passes"]),
            ("B", cfg["b_passes"]), ("V_CALIBRATED", cfg["calibrated_v_passes"]),
            ("R", cfg["r_passes"]),
        )
        names = [name for name, _ in stages]
        reward_generator = torch.Generator().manual_seed(42)
        if start_stage not in names:
            raise ValueError(f"Unknown LFIRL starting stage: {start_stage}")
        stages = stages[names.index(start_stage):]
        for stage, passes in stages:
            if passes < 1:
                raise ValueError("All official LFIRL stages must run at least one pass")
            if stage == "R":
                # Upstream explicitly freezes the finalized stage outputs just
                # before R fitting (train_lfirl.py, after V_CALIBRATED).
                q_target.load_state_dict(self.q.state_dict(), strict=True)
                b_target.load_state_dict(self.b.state_dict(), strict=True)
                v_target.load_state_dict(self.v.state_dict(), strict=True)
                self.target_mode = "final_frozen"
                self.target_states = {
                    name: deepcopy(target.state_dict())
                    for name, target in (("q", q_target), ("b", b_target), ("v", v_target))
                }
            active = {
                "QHAT": self.q,
                "V_INIT": self.v,
                "B": self.b,
                "V_CALIBRATED": self.v,
                "R": self.r,
            }[stage]
            for network in (self.q, self.b, self.v, self.r):
                network.requires_grad_(network is active)
                network.train(network is active)
            count = 0
            losses = []
            for epoch in range(int(passes)):
                batches = transitions.batches(
                    int(cfg["batch_size"]), self.device,
                    shuffle=((shuffle_reward and stage == "R")
                             or (shuffle_calibration and stage != "QHAT")),
                    generator=reward_generator,
                )
                for obs, action, next_obs, done in batches:
                    if stage == "QHAT":
                        loss = source.train_qhat_LFIRL_step(
                            q_hat=self.q, q_optim=optimizers["q"], dp_model=policy,
                            obs=obs, action=action, device=self.device, policy_obs_keys=None,
                            q_loss_cfg=self.settings["q_net"]["loss"], **bounds,
                        )
                        source.soft_update(q_target, self.q, tau)
                    elif stage in ("V_INIT", "V_CALIBRATED"):
                        view = (
                            source.QHatOnlyWrapper(self.q) if stage == "V_INIT"
                            else source.QFullWrapper(self.q, self.b)
                        )
                        loss = source.train_v_function_step(
                            v_net=self.v, v_optim=optimizers["v"], q_full=view,
                            obs=obs, action=action, cfg=self.settings["v_net"],
                        )
                        source.soft_update(v_target, self.v, tau)
                    elif stage == "B":
                        loss = source.train_b_offset_step(
                            q_hat=self.q, v_net=self.v, b_net=self.b, b_optim=optimizers["b"],
                            obs=obs, action=action, next_obs=next_obs, done_t=done,
                            gamma=gamma, device=self.device, action_dim=self.action_dim,
                            cfg=self.settings["b_net"], **bounds,
                        )
                        source.soft_update(b_target, self.b, tau)
                    else:
                        loss, _ = source.train_reward_function_step(
                            r_net=self.r, r_optim=optimizers["r"],
                            q_targ=source.QFullWrapper(q_target, b_target), v_targ=v_target,
                            obs=obs, action=action, next_obs=next_obs, done_t=done,
                            gamma=gamma, r_loss_cfg=self.settings["r_net"]["loss"], r_rms=self.rms,
                        )
                    if not np.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite original LFIRL loss in {stage}")
                    count += 1
                    losses.append(loss)
                    if checkpoint_interval and stage == "R" and count % checkpoint_interval == 0:
                        torch.save(self.state_dict(), output / f"reward_step_{count}.pt")
                    if count % 50 == 0:
                        print(f"[LFIRL] {stage} updates={count} loss={np.mean(losses[-50:]):.6f}", flush=True)
                    if max_batches_per_stage and count >= max_batches_per_stage:
                        break
                if max_batches_per_stage and count >= max_batches_per_stage:
                    break
            result = dict(stage=stage, updates=count, mean_loss=float(np.mean(losses)), final_loss=float(losses[-1]))
            self.history.append(result)
            print(f"[LFIRL stage] {json.dumps(result)}", flush=True)
        self.r.requires_grad_(False).eval()
        return self.history

    @torch.no_grad()
    def predict(self, observations, actions):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if obs.ndim == 2:
            obs = obs[:, None]
        return self.r(obs, act.reshape(len(act), -1)).cpu().numpy()

    def state_dict(self):
        return dict(
            settings=self.settings, observation_dim=self.observation_dim, action_dim=self.action_dim,
            models={name: net.state_dict() for name, net in (("q", self.q), ("b", self.b), ("v", self.v), ("r", self.r))},
            observation_mean=self.mean, observation_std=self.std,
            calibration_targets=self.target_states,
            calibration_target_mode=self.target_mode,
            reward_target_statistics=dict(mean=self.rms.mean, variance=self.rms.var, count=self.rms.count),
        )

    def load_state_dict(self, saved, load_reward_head=True):
        for name in ("q", "b", "v", "r"):
            if name == "r" and not load_reward_head:
                continue
            getattr(self, name).load_state_dict(saved["models"][name], strict=True)
        self.set_normalization(saved["observation_mean"], saved["observation_std"])
        statistics = saved["reward_target_statistics"]
        self.rms.mean = float(statistics["mean"])
        self.rms.var = float(statistics["variance"])
        self.rms.count = float(statistics["count"])
        self.target_states = saved.get("calibration_targets")
        self.target_mode = saved.get("calibration_target_mode", "unspecified_legacy")
        self.r.requires_grad_(False).eval()


def default_settings():
    with (source.SOURCE / "environments/FrankaKitchen-v1.yml").open() as stream:
        settings = yaml.safe_load(stream)
    irl = settings["irl"]
    irl["initial_v_passes"] = irl["v_passes"]
    irl["calibrated_v_passes"] = irl["v_passes"]
    return {key: settings[key] for key in ("q_net", "b_net", "v_net", "r_net", "irl")}
