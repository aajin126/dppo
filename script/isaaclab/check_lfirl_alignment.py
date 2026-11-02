"""Audit recovered rewards against complete, recorded assembly outcomes.

Assembly labels are evaluation targets only. No policy actions, rewards or
expert demonstrations are changed by this script.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def ranking_auc(positive, negative):
    import numpy as np
    from scipy.stats import rankdata

    if not len(positive) or not len(negative):
        return None
    values = np.r_[positive, negative]
    count = len(positive)
    return float((rankdata(values)[:count].sum() - count * (count + 1) / 2)
                 / (count * len(negative)))


def outcome_metrics(values, assembly, gamma):
    import numpy as np

    weights = gamma ** np.arange(values.shape[1])
    cumulative = (values * weights).cumsum(1)
    success = assembly.sum(1) >= 1
    positive, negative = cumulative[success, -1], cumulative[~success, -1]
    order = np.argsort(cumulative[:, -1])[::-1]
    top_counts = {str(percent): max(1, int(np.ceil(len(order) * percent / 100)))
                  for percent in (1, 5, 10)}
    preferences = []
    for index in np.flatnonzero(success):
        first = int(np.flatnonzero(assembly[index] >= 1)[0])
        if first >= values.shape[1]:
            continue
        difference = cumulative[index, first] - cumulative[~success, first]
        preferences.extend(((difference > 0) + .5 * (difference == 0)).tolist())
    return dict(
        successes=int(success.sum()), failures=int((~success).sum()),
        discounted_return_success_auc=ranking_auc(positive, negative),
        completion_prefix_preference=float(np.mean(preferences)) if preferences else None,
        completion_prefix_comparisons=len(preferences),
        successful_mean_return=float(positive.mean()) if len(positive) else None,
        failed_mean_return=float(negative.mean()) if len(negative) else None,
        failed_return_p99=float(np.quantile(negative, .99)) if len(negative) else None,
        failed_return_maximum=float(negative.max()) if len(negative) else None,
        top_return_success_fraction={key: float(success[order[:count]].mean())
                                     for key, count in top_counts.items()},
    )


def paired_outcome_metrics(policy_scores, policy_assembly, gamma):
    """Compare success/failure only within the same initial physical reset."""
    import itertools
    import numpy as np

    result = {}
    for first, second in itertools.combinations(policy_scores, 2):
        left, right = policy_scores[first], policy_scores[second]
        if left.shape != right.shape:
            raise ValueError("Paired reward comparison requires matching episode/time axes")
        left_success = policy_assembly[first].sum(1) >= 1
        right_success = policy_assembly[second].sum(1) >= 1
        different = left_success != right_success
        weights = gamma ** np.arange(left.shape[1])
        left_prefix = (left * weights).cumsum(1)
        right_prefix = (right * weights).cumsum(1)
        difference = (left_prefix[:, -1] - right_prefix[:, -1]) * np.where(left_success, 1, -1)
        margins = difference[different]
        prefix = []
        for index in np.flatnonzero(different):
            events = policy_assembly[first if left_success[index] else second][index]
            end = int(np.flatnonzero(events >= 1)[0])
            if end < left.shape[1]:
                delta = (left_prefix[index, end] - right_prefix[index, end]) * (1 if left_success[index] else -1)
                prefix.append(float((delta > 0) + .5 * (delta == 0)))
        result[f"{first}_vs_{second}"] = dict(
            informative_resets=int(different.sum()),
            success_preference=float(np.mean((margins > 0) + .5 * (margins == 0))) if len(margins) else None,
            completion_prefix_preference=float(np.mean(prefix)) if prefix else None,
            completion_prefix_resets=len(prefix),
            mean_success_minus_failure_return=float(margins.mean()) if len(margins) else None,
        )
    return result


def main():
    import numpy as np
    import torch
    from agent.finetune.lfirl.reward import LFIRLReward

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", nargs="+", default=["bc", "best", "final"])
    parser.add_argument("--audit-bellman-targets", action="store_true")
    parser.add_argument("--save-scores", action="store_true")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved["horizon_steps"] != 8:
        raise ValueError("This audit requires recorded eight-action macro transitions")
    model = LFIRLReward(saved["learner"]["settings"], saved["observation_dim"],
                        saved["action_dim"] * saved["horizon_steps"], "cpu")
    model.load_state_dict(saved["learner"])
    groups = {}
    policy_scores, policy_targets, policy_assembly, policy_paths = {}, {}, {}, {}
    all_scores, all_targets, all_labels, sources = [], [], [], []
    for label in args.policy:
        report = json.loads((args.rollouts / label / "report.json").read_text())
        if not report.get("completed"):
            raise ValueError(f"Policy evaluation is incomplete: {label}")
        values, targets, outcomes = [], [], []
        paths = sorted((args.rollouts / label).glob("*.npz"))
        policy_paths[label] = paths
        for path in paths:
            with np.load(path) as raw:
                states = raw["pre_states"][:, :-1].copy()
                following = raw["pre_states"][:, 1:].copy()
                commands = raw["actions"][:, :-1].reshape(*states.shape[:2], -1).copy()
                assembly = raw["assembly_rewards"].copy()
            shape = states.shape[:2]
            flat = states.reshape(-1, 1, saved["observation_dim"])
            next_flat = following.reshape(flat.shape)
            action_flat = commands.reshape(len(flat), -1)
            predictions, implied = [], []
            with torch.no_grad():
                for start in range(0, len(flat), 2048):
                    obs = torch.from_numpy(flat[start:start + 2048])
                    action = torch.from_numpy(action_flat[start:start + 2048])
                    next_obs = torch.from_numpy(next_flat[start:start + 2048])
                    predictions.append(model.predict(obs, action).reshape(-1))
                    if args.audit_bellman_targets:
                        target = (model.q(obs, action).reshape(-1) + model.b(obs).reshape(-1)
                                  - saved["gamma"] * model.v(next_obs).reshape(-1))
                        implied.append(target.numpy())
            scores = np.concatenate(predictions).reshape(shape)
            values.append(scores)
            outcomes.append(assembly)
            all_scores.append(scores)
            all_labels.append(assembly)
            sources.append(dict(policy=label, archive=str(path), episodes=len(states)))
            if implied:
                targets.append(np.concatenate(implied).reshape(shape))
                all_targets.append(targets[-1])
        if not values:
            raise ValueError(f"No complete rollout archives for {label}")
        values, outcomes = np.concatenate(values), np.concatenate(outcomes)
        policy_scores[label], policy_assembly[label] = values, outcomes
        group = dict(learned_reward=outcome_metrics(values, outcomes, saved["gamma"]))
        if targets:
            targets = np.concatenate(targets)
            policy_targets[label] = targets
            group["actual_transition_bellman_target"] = outcome_metrics(targets, outcomes, saved["gamma"])
            group["reward_vs_actual_target_rmse"] = float(np.sqrt(np.mean((targets - values)**2)))
        groups[label] = group
    combined = outcome_metrics(np.concatenate(all_scores), np.concatenate(all_labels), saved["gamma"])
    if len(policy_paths) > 1:
        anchor = args.policy[0]
        for label in args.policy[1:]:
            if len(policy_paths[label]) != len(policy_paths[anchor]):
                raise ValueError("Paired policies have different reset archives")
            for first, second in zip(policy_paths[anchor], policy_paths[label]):
                if first.stem.split("seed_")[-1] != second.stem.split("seed_")[-1]:
                    raise ValueError("Paired reward comparison requires matching seeds")
                with np.load(first) as left, np.load(second) as right:
                    for key in ("initial_physics", "initial_states"):
                        if not np.array_equal(left[key], right[key]):
                            raise ValueError(f"Different {key}: cross-episode reward ranks are confounded")
    result = dict(
        checkpoint=str(args.checkpoint), sources=sources, by_policy=groups,
        combined_learned_reward=combined,
        matched_reset_reward_comparisons=paired_outcome_metrics(policy_scores, policy_assembly, saved["gamma"]),
        initial_state_offsets_controlled=len(policy_paths) > 1,
        scored_chunks=int(all_scores[0].shape[1]), assembly_label_chunks=int(all_labels[0].shape[1]),
        training_on_rollout_labels=False,
        note="The final transition is omitted because its next state was not recorded. Completion-prefix comparisons use equal-length prefixes and never relabel failures as experts.",
    )
    if all_targets:
        result["combined_actual_transition_bellman_target"] = outcome_metrics(
            np.concatenate(all_targets), np.concatenate(all_labels), saved["gamma"],
        )
        result["matched_reset_bellman_target_comparisons"] = paired_outcome_metrics(
            policy_targets, policy_assembly, saved["gamma"],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_scores:
        extra = {"bellman_targets": np.concatenate(all_targets)} if all_targets else {}
        np.savez_compressed(
            args.output.with_suffix(".npz"), rewards=np.concatenate(all_scores),
            assembly=np.concatenate(all_labels), gamma=np.array(saved["gamma"]),
            sources_json=np.array(json.dumps(sources)),
            **extra,
        )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "sources"}, indent=2))


if __name__ == "__main__":
    main()
