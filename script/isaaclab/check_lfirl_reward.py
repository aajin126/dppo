"""Describe recovered LFIRL reward on expert commands and saved real rollouts."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def rollout_transitions(raw, horizon):
    """Extract aligned pre-action states without inventing intermediate states."""
    import numpy as np

    actions = raw["actions"]
    if horizon == 1 and actions.ndim == 4:
        if "micro_pre_states" not in raw:
            raise ValueError("Single-action reward requires recorded micro_pre_states")
        states = raw["micro_pre_states"]
        if states.shape[:-1] != actions.shape[:-1]:
            raise ValueError("Recorded action/state time axes differ")
        return (states.reshape(states.shape[0], -1, states.shape[-1]),
                actions.reshape(actions.shape[0], -1, actions.shape[-1]))
    if "pre_states" in raw:
        states = raw["pre_states"]
    elif "states" in raw and "initial_states" in raw:
        states = np.concatenate((raw["initial_states"][:, None], raw["states"][:, :-1]), axis=1)
    else:
        raise ValueError("Rollout has no aligned pre-action states")
    if states.shape[:2] != actions.shape[:2]:
        raise ValueError("Recorded action/state time axes differ")
    return states, actions.reshape(*actions.shape[:2], -1)


def predict_batches(reward, states, actions, batch_size=2048):
    import numpy as np

    return np.concatenate([
        reward.predict(states[start:start + batch_size], actions[start:start + batch_size])
        for start in range(0, len(actions), batch_size)
    ])


def bellman_regression_report(reward, saved):
    """Recompute regression against the exact frozen LFIRL targets."""
    import numpy as np
    import torch
    from agent.finetune.lfirl.dataset import MacroTransitions

    targets = reward.target_states
    if not targets or reward.target_mode != "final_frozen":
        raise ValueError("Frozen calibration targets are required for regression verification")
    for name in ("q", "b", "v"):
        for key, value in getattr(reward, name).state_dict().items():
            torch.testing.assert_close(value.cpu(), targets[name][key].cpu(), rtol=0, atol=0)
    data = MacroTransitions(
        saved["expert_dataset_path"], saved["horizon_steps"],
        saved["data"]["terminal_mode"],
        window_stride=saved["data"].get("window_stride", saved["horizon_steps"]),
    )
    predictions, references = [], []
    with torch.no_grad():
        for obs, action, next_obs, done in data.batches(1024, "cpu"):
            reference = (reward.q(obs, action).reshape(-1) + reward.b(obs).reshape(-1)
                         - saved["gamma"] * (1 - done) * reward.v(next_obs).reshape(-1))
            limit = float(reward.settings["r_net"]["loss"].get("target_clip", 1e6))
            references.append(reference.clamp(-limit, limit).numpy())
            predictions.append(reward.predict(obs, action).reshape(-1))
    predictions, references = np.concatenate(predictions), np.concatenate(references)
    if not np.isfinite(predictions).all() or not np.isfinite(references).all():
        raise FloatingPointError("Nonfinite regression predictions or targets")
    fitted_ids = np.unique(data.episode_ids)[:saved["data"]["expert_episodes"]]
    fitted = np.isin(data.episode_ids, fitted_ids)
    result = dict(
        target="Original frozen Q+b-gamma*(1-done)*V(next_state), clipped as in source",
        comparison_with_ground_truth_reward=False,
        excluded_episode_caveat="Excluded from fitting; may have been used for model selection.",
    )
    for label, mask in (("fitted", fitted), ("excluded", ~fitted)):
        if not mask.any():
            result[label] = None
            continue
        error = predictions[mask] - references[mask]
        variance = float(references[mask].var())
        result[label] = dict(
            transitions=int(mask.sum()), episodes=int(len(np.unique(data.episode_ids[mask]))),
            rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
            r_squared=float(1 - np.mean(error**2) / variance) if variance > 0 else None,
            target_standard_deviation=float(np.sqrt(variance)),
        )
    return result


def main():
    import numpy as np
    import torch
    from scipy.stats import rankdata
    from agent.finetune.lfirl.dataset import MacroTransitions
    from agent.finetune.lfirl.reward import LFIRLReward

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-rollouts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-bellman-fit", action="store_true")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    reward = LFIRLReward(saved["learner"]["settings"], saved["observation_dim"], saved["horizon_steps"] * saved["action_dim"], "cpu")
    reward.load_state_dict(saved["learner"])
    # Reward fitting may use overlapping windows. Trajectory comparison must
    # advance by the same executed action horizon as the simulator instead.
    # The R network does not consume terminal flags; keep complete interiors.
    data = MacroTransitions(
        saved["expert_dataset_path"], saved["horizon_steps"], "truncate",
        window_stride=saved["horizon_steps"],
    )
    scores = predict_batches(reward, data.observations, data.actions)
    random_actions = np.random.default_rng(42).uniform(-1, 1, data.actions.shape).astype(np.float32)
    random_scores = predict_batches(reward, data.observations, random_actions)
    expert_returns = [float(scores[data.episode_ids == episode].mean()) for episode in np.unique(data.episode_ids)]
    report = dict(
        checkpoint=str(args.checkpoint),
        expert_data_scope="All available demonstrations; includes reward fitting and model-selection episodes",
        expert_probe_window_stride=saved["horizon_steps"],
        reward_training_window_stride=saved["data"].get("window_stride", saved["horizon_steps"]),
        expert_mean_reward_per_chunk=float(scores.mean()), expert_std=float(scores.std()),
        random_action_mean_reward_per_chunk=float(random_scores.mean()),
        fraction_expert_action_scored_above_random=float(
            ((scores > random_scores) + .5 * (scores == random_scores)).mean()),
        random_action_probe_applicable=bool(reward.r.include_action),
        expert_mean_reward_by_time_quartile=np.mean([
            [float(part.mean()) for part in np.array_split(scores[data.episode_ids == episode], 4)]
            for episode in np.unique(data.episode_ids)
        ], axis=0).tolist(),
        expert_episode_mean_rewards=expert_returns, rollouts=[],
        note="Descriptive probes only, no automatic quality threshold or success guarantee.",
    )
    if args.check_bellman_fit:
        report["bellman_regression"] = bellman_regression_report(reward, saved)
    failed_scores = []
    successful_scores = []
    matched_length_comparisons = []
    if args.policy_rollouts:
        for path in sorted(args.policy_rollouts.glob("*.npz")):
            with np.load(path) as raw:
                if not all(key in raw for key in ("actions", "assembly_rewards")):
                    continue
                states, actions = rollout_transitions(raw, saved["horizon_steps"])
                values = predict_batches(
                    reward, states.reshape(-1, 1, saved["observation_dim"]),
                    actions.reshape(-1, data.actions.shape[-1]),
                ).reshape(states.shape[:2])
                assembled = raw["assembly_rewards"].sum(1) >= 1
                failed_scores.extend(values.mean(1)[~assembled].tolist())
                successful_scores.extend(values.mean(1)[assembled].tolist())
                cumulative = values.cumsum(1)
                for episode in np.unique(data.episode_ids):
                    expert = scores[data.episode_ids == episode]
                    if len(expert) > values.shape[1]:
                        continue
                    # Upstream evaluate.py sums rewards over length-matched
                    # expert/generated trajectories. Reuse that comparison.
                    policy_sums = cumulative[~assembled, len(expert) - 1]
                    matched_length_comparisons.extend(
                        ((expert.sum() > policy_sums).astype(float)
                         + .5 * (expert.sum() == policy_sums)).tolist()
                    )
                report["rollouts"].append(dict(
                    source=str(path), episodes=len(states), assembly_successes=int(assembled.sum()),
                    mean_reward_per_reward_step=float(values.mean()),
                    episode_mean_rewards=values.mean(1).tolist(),
                ))
    if failed_scores:
        combined = np.r_[expert_returns, failed_scores]
        n = len(expert_returns)
        report["expert_vs_failed_policy_mean_reward_auc"] = float(
            (rankdata(combined)[:n].sum() - n * (n + 1) / 2) / (n * len(failed_scores))
        )
        report["policy_evaluation_caveat"] = "Previously collected rollouts; expert and simulator distributions differ. The AUC uses per-reward-step averages; length-matched comparisons separately use reward sums, as in upstream evaluate.py."
    if matched_length_comparisons:
        report["length_matched_expert_above_failed_policy_fraction"] = float(np.mean(matched_length_comparisons))
        report["length_matched_pair_count"] = len(matched_length_comparisons)
    if successful_scores and failed_scores:
        comparisons = np.asarray(successful_scores)[:, None] - np.asarray(failed_scores)
        report["actual_success_vs_failure_reward_auc"] = float(
            ((comparisons > 0) + .5 * (comparisons == 0)).mean()
        )
    if args.policy_rollouts and not report["rollouts"]:
        raise ValueError("No compatible policy rollout archives found; no policy ranking was computed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("rollouts", "expert_episode_mean_rewards")}, indent=2))


if __name__ == "__main__":
    main()
