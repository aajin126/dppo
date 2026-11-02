"""Summarize frozen reward quality on saved expert and simulator trajectories."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from scipy.stats import rankdata, spearmanr
    from agent.finetune.firl.learner import FIRLRewardLearner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    args = parser.parse_args()
    saved = torch.load(args.run_dir / "checkpoint/firl_99.pt", map_location="cpu", weights_only=True)
    learner = FIRLRewardLearner(58, "cpu", OmegaConf.create(saved["config"]))
    learner.load_state_dict(saved["learner"])
    normalizer = torch.load(ROOT / "data/furniture/one_leg_low/normalization.pth", weights_only=True)
    low_min = normalizer["stats.observations.min"].numpy()
    low_max = normalizer["stats.observations.max"].numpy()
    report = {"reward_checkpoint": str(args.run_dir / "checkpoint/firl_99.pt"), "expert": {}}
    low_states = None
    low_raw = None
    low_initial_raw = None
    low_fingerprints = set()
    expert_med_episode_rewards = []
    for task in ("one_leg_low", "one_leg_med"):
        data_path = ROOT / "data/furniture" / task
        norm = torch.load(data_path / "normalization.pth", weights_only=True)
        minimum = norm["stats.observations.min"].numpy()
        maximum = norm["stats.observations.max"].numpy()
        with np.load(data_path / "train.npz") as data:
            states = data["states"]
            valid = ~((states == 0).all(-1) & (data["actions"] == 0).all(-1))
            raw = (states + 1) / 2 * (maximum - minimum) + minimum
            lengths = data["traj_lengths"]
        boundaries = np.r_[0, lengths.cumsum()]
        fingerprints = [
            hashlib.sha256(np.round(raw[start:end][valid[start:end]], 5).astype("<f4").tobytes()).hexdigest()
            for start, end in zip(boundaries[:-1], boundaries[1:])
            if valid[start:end].any()
        ]
        transformed = 2 * (raw[valid] - low_min) / (low_max - low_min) - 1
        transformed = np.clip(transformed, -5, 5)
        rewards = learner.predict(transformed)
        logits = learner.disc.log_density_ratio(transformed).cpu().numpy()
        valid_counts = [int(valid[start:end].sum()) for start, end in zip(boundaries[:-1], boundaries[1:])]
        valid_boundaries = np.r_[0, np.cumsum(valid_counts)]
        episode_rewards = [
            float(rewards[start:end].mean())
            for start, end in zip(valid_boundaries[:-1], valid_boundaries[1:])
            if end > start
        ]
        report["expert"][task] = {
            "scope": "training experts" if task == "one_leg_low" else "separate medium-randomness experts, renormalized to low; distribution-shift check",
            "states": len(rewards),
            "valid_demonstrations": len(fingerprints),
            "declared_trajectories_including_all_zero_padding": len(lengths),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "mean_reward_per_demonstration": episode_rewards,
            "discriminator_logit_mean": float(logits.mean()),
            "discriminator_expert_accuracy": float((logits > 0).mean()),
            "whole_trajectory_rounded_fingerprint_overlap_with_low": None if task == "one_leg_low" else len(set(fingerprints) & low_fingerprints),
        }
        if task == "one_leg_low":
            low_states = transformed
            low_raw = raw[valid]
            low_initial_raw = np.stack([
                raw[start + np.flatnonzero(valid[start:end])[0]]
                for start, end in zip(boundaries[:-1], boundaries[1:])
                if valid[start:end].any()
            ])
            low_fingerprints = set(fingerprints)
        else:
            expert_med_episode_rewards = episode_rewards
    parts = ["square_table_top", "square_table_leg1", "square_table_leg2", "square_table_leg3", "square_table_leg4", "front_obstacle"]
    names = ["tcp_x", "tcp_y", "tcp_z"] + [f"tcp_rotation6d_{i}" for i in range(6)]
    names += [f"tcp_linear_velocity_{axis}" for axis in "xyz"]
    names += [f"tcp_angular_velocity_{axis}" for axis in "xyz"] + ["gripper_width"]
    names += [f"{part}_{component}" for part in parts for component in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    groups = {"robot_pose": list(range(9)), "robot_velocity": list(range(9, 15)), "gripper": [15], "obstacle": list(range(51, 58))}
    groups.update({part: list(range(16 + i * 7, 23 + i * 7)) for i, part in enumerate(parts[:-1])})
    expert_reference = np.median(low_states, axis=0)
    rollouts = json.loads((args.diagnostic_dir / "report.json").read_text())["rollouts"]
    report["rollouts"] = []
    report["policy_comparison"] = []
    baseline_by_seed = {r["seed"]: r for r in rollouts if r["label"] == "state_0"}
    if baseline_by_seed:
        reset_states = np.concatenate([
            np.load(row["trajectory_path"])["initial_states"]
            for row in baseline_by_seed.values()
        ])
        reset_raw = (reset_states + 1) / 2 * (low_max - low_min) + low_min
        report["initial_layout_comparison"] = [{
            "name": name,
            "expert_median_xyz_m": np.median(low_initial_raw[:, start:start + 3], axis=0).tolist(),
            "sim_median_xyz_m": np.median(reset_raw[:, start:start + 3], axis=0).tolist(),
        } for name, start in [("tcp", 0)] + [(part, 16 + i * 7) for i, part in enumerate(parts)]]
    for label in ("state_0", "state_60", "state_99", "frozen_trained"):
        selected = [r for r in rollouts if r["label"] == label]
        if not selected:
            continue
        differences = np.concatenate([
            np.asarray(r["frozen_reward_returns"]) - baseline_by_seed[r["seed"]]["frozen_reward_returns"]
            for r in selected
        ])
        rng = np.random.default_rng(19)
        bootstrap = differences[rng.integers(len(differences), size=(10000, len(differences)))].mean(1)
        reset_errors = []
        position_indices = list(range(3)) + [j for start in range(16, 58, 7) for j in range(start, start + 3)]
        for row in selected:
            initial = np.load(row["trajectory_path"])["initial_states"]
            baseline = np.load(baseline_by_seed[row["seed"]]["trajectory_path"])["initial_states"]
            error = np.abs(initial - baseline) * (low_max - low_min) / 2
            reset_errors.append(float(error[:, position_indices].max() * 1000))
        report["policy_comparison"].append({
            "policy": label,
            "episodes": sum(r["episodes"] for r in selected),
            "assembly_successes": sum(sum(r["assembly_successes"]) for r in selected),
            "sustained_leg_lift_proxy_count": sum(sum(r["leg_lift_3cm_sustained_3_chunks"]) for r in selected),
            "mean_frozen_reward_return": float(np.mean([v for r in selected for v in r["frozen_reward_returns"]])),
            "paired_reward_difference_from_bc": float(differences.mean()),
            "descriptive_paired_bootstrap_95_percent_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
            "maximum_initial_position_difference_from_bc_mm": max(reset_errors),
        })
    report["comparison_caveats"] = [
        "Small seeded samples, with initial latent noise shared between policies.",
        "PhysX resets are not bitwise identical; initial position discrepancies are measured above.",
        "The lift proxy is root height gain >3cm for >=3 action chunks; it does not certify a grasp.",
        "The short frozen-reward experiment tests local behavior only, not convergence.",
    ]
    all_returns, all_success, all_lift = [], [], []
    all_lift_proxy = []
    for result in rollouts:
        data = np.load(result["trajectory_path"])
        states = data["states"].reshape(-1, 58)
        raw = (states + 1) / 2 * (low_max - low_min) + low_min
        outside = ((states < low_states.min(0) - .01) | (states > low_states.max(0) + .01)).mean(0)
        top = np.argsort(-outside)[:8]
        logits = learner.disc.log_density_ratio(states).cpu().numpy()
        # Counterfactual feature replacement locates sensitivity only. These
        # synthetic states are not physical interventions or proposed fixes.
        counterfactual = {}
        reward_counterfactual = {}
        base_reward = learner.predict(states)
        for name, indices in groups.items():
            altered = states.copy()
            altered[:, indices] = expert_reference[indices]
            shifted = learner.disc.log_density_ratio(altered).cpu().numpy()
            counterfactual[name] = float(shifted.mean() - logits.mean())
            reward_counterfactual[name] = float(learner.predict(altered).mean() - base_reward.mean())
        entry = {
            "label": result["label"], "seed": result["seed"],
            "successes": int(sum(result["assembly_successes"])),
            "lift_proxy_count": int(sum(result["leg_lift_3cm_sustained_3_chunks"])),
            "mean_frozen_reward_return": float(np.mean(result["frozen_reward_returns"])),
            "policy_discriminator_negative_clamp_fraction": float((logits <= -9.999).mean()),
            "policy_reward_clipped_fraction": float((np.abs(base_reward) >= 9.999).mean()),
            "js_covariance_signal_per_episode": (-np.logaddexp(0, data["discriminator_logits"]).mean(1)).tolist(),
            "out_of_training_range_features": [{
                "name": names[i], "fraction": float(outside[i]),
                "expert_raw_range": [float(low_raw[:, i].min()), float(low_raw[:, i].max())],
                "policy_raw_range": [float(raw[:, i].min()), float(raw[:, i].max())],
            } for i in top],
            "classifier_mean_logit_shift_when_group_replaced_with_expert_median": counterfactual,
            "reward_mean_shift_when_group_replaced_with_expert_median": reward_counterfactual,
        }
        report["rollouts"].append(entry)
        all_returns.extend(result["frozen_reward_returns"])
        all_success.extend(result["assembly_successes"])
        all_lift.extend(result["max_leg_height_gain_m"])
        all_lift_proxy.extend(result["leg_lift_3cm_sustained_3_chunks"])
    if all_returns:
        scores = np.asarray(all_returns)
        success = np.asarray(all_success)
        count = int(success.sum())
        lift_proxy = np.asarray(all_lift_proxy)
        lift_count = int(lift_proxy.sum())
        report["reward_quality"] = {
            "episodes": len(scores), "successes": count,
            "mean_return_success": float(scores[success].mean()) if count else None,
            "mean_return_failure": float(scores[~success].mean()) if (~success).any() else None,
            "assembly_success_auc": float((rankdata(scores)[success].sum() - count * (count + 1) / 2) / (count * (~success).sum())) if 0 < count < len(scores) else None,
            "return_vs_max_leg_height_spearman": float(spearmanr(scores, all_lift).statistic) if np.ptp(all_lift) else None,
            "sustained_lift_proxy_auc": float((rankdata(scores)[lift_proxy].sum() - lift_count * (lift_count + 1) / 2) / (lift_count * (~lift_proxy).sum())) if 0 < lift_count < len(scores) else None,
            "interpretation": "Descriptive diagnostics on a small mixed-policy sample; no reward validity or causality guarantee.",
        }
        failed_scores = scores[~success] / data["states"].shape[1]
        if len(failed_scores):
            combined = np.r_[expert_med_episode_rewards, failed_scores]
            n_expert = len(expert_med_episode_rewards)
            report["reward_quality"]["separate_med_expert_vs_failed_policy_auc"] = float(
                (rankdata(combined)[:n_expert].sum() - n_expert * (n_expert + 1) / 2)
                / (n_expert * len(failed_scores))
            )
            report["reward_quality"]["expert_comparison_caveat"] = "Expert and policy data come from different simulators and initial-state distributions; this AUC can reflect those differences."
    frozen_checkpoints = args.diagnostic_dir / "frozen_training/checkpoint"
    if frozen_checkpoints.is_dir():
        policies = sorted(frozen_checkpoints.glob("state_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        if policies:
            final_policy = torch.load(policies[-1], map_location="cpu", weights_only=True)
            initial_policy = torch.load(args.run_dir / "checkpoint/state_0.pt", map_location="cpu", weights_only=True)["model"]
            reward_checkpoint = frozen_checkpoints / f"firl_{final_policy['itr']}.pt"
            frozen = torch.load(reward_checkpoint, map_location="cpu", weights_only=True)
            report["frozen_training_verification"] = {
                "last_saved_iteration": int(final_policy["itr"]),
                "reward_weights_bitwise_unchanged": all(
                    torch.equal(value, frozen["learner"]["reward"][name])
                    for name, value in saved["learner"]["reward"].items()
                ),
                "discriminator_weights_bitwise_unchanged": all(
                    torch.equal(value, frozen["learner"]["discriminator"][name])
                    for name, value in saved["learner"]["discriminator"].items()
                ),
                "reward_optimizer_update_count_unchanged": frozen["learner"]["reward_updates"] == saved["learner"]["reward_updates"],
                "actor_maximum_parameter_change_from_bc": max(
                    float((final_policy["model"][name] - value).abs().max())
                    for name, value in initial_policy.items() if name.startswith("actor_ft.")
                ),
            }
    output = args.diagnostic_dir / "reward_analysis.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
