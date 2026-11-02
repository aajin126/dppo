"""Compare reward preferences with measured, paired physical interventions."""

import argparse
import json
from pathlib import Path

import numpy as np


def rescore_branches(arrays, checkpoint):
    """Use the same physical trajectories to separate R error from Q/V targets."""
    import torch
    from agent.finetune.lfirl.reward import LFIRLReward, sha256

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = LFIRLReward(saved["learner"]["settings"], 58, 80, "cpu")
    model.load_state_dict(saved["learner"])
    limit = float(saved["learner"]["settings"]["r_net"]["loss"].get("target_clip", 1e6))
    for values in arrays.values():
        shape = values["rewards"].shape
        states = torch.from_numpy(values["states"][:, :-1].reshape(-1, 1, 58))
        next_states = torch.from_numpy(values["states"][:, 1:].reshape(-1, 1, 58))
        actions = torch.from_numpy(values["actions"].reshape(-1, 80))
        predictions, targets = [], []
        with torch.no_grad():
            for start in range(0, len(states), 1024):
                end = start + 1024
                state, action = states[start:end], actions[start:end]
                q = model.q(state, action).reshape(-1) + model.b(state).reshape(-1)
                target = q - saved["gamma"] * model.v(next_states[start:end]).reshape(-1)
                predictions.append(model.r(state, action).reshape(-1).numpy())
                targets.append(target.clamp(-limit, limit).numpy())
            initial = torch.from_numpy(values["states"][:, :1])
            first = torch.from_numpy(values["actions"][:, 0].reshape(-1, 80))
            values["start_q"] = (model.q(initial, first) + model.b(initial)).reshape(-1).numpy()
            values["end_v"] = model.v(torch.from_numpy(values["states"][:, -1:])).reshape(-1).numpy()
        values["rewards"] = np.concatenate(predictions).reshape(shape)
        values["bellman_targets"] = np.concatenate(targets).reshape(shape)
    return dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint), gamma=saved["gamma"])


def matched_physics(left, right, joints=9):
    """Require close pre-intervention state; never match only policy inputs."""
    if left.shape != right.shape or (left.shape[1] - 4 * joints - 1) % 13:
        raise ValueError("Unexpected branch physical snapshot schema")
    cursor = 2 * joints
    roots = (left.shape[1] - 4 * joints - 1) // 13
    # Schema: joint position/velocity, root states, grip command, efforts,
    # joint targets. Root count is derived without assuming part names.
    if cursor + roots * 13 + 1 + 2 * joints != left.shape[1]:
        raise ValueError("Unknown physical state layout")
    differences = {
        "joint_position_rad": np.abs(left[:, :joints] - right[:, :joints]).max(1),
        "joint_velocity_rad_s": np.abs(left[:, joints:cursor] - right[:, joints:cursor]).max(1),
    }
    lroot = left[:, cursor:cursor + roots * 13].reshape(-1, roots, 13)
    rroot = right[:, cursor:cursor + roots * 13].reshape(lroot.shape)
    differences["root_position_m"] = np.abs(lroot[:, :, :3] - rroot[:, :, :3]).max((1, 2))
    differences["root_quaternion"] = np.minimum(
        np.abs(lroot[:, :, 3:7] - rroot[:, :, 3:7]).max(2),
        np.abs(lroot[:, :, 3:7] + rroot[:, :, 3:7]).max(2),
    ).max(1)
    differences["root_velocity"] = np.abs(lroot[:, :, 7:] - rroot[:, :, 7:]).max((1, 2))
    cursor += roots * 13
    differences["grip_command"] = np.abs(left[:, cursor] - right[:, cursor])
    differences["effort_Nm"] = np.abs(left[:, cursor + 1:cursor + 1 + joints]
                                       - right[:, cursor + 1:cursor + 1 + joints]).max(1)
    differences["joint_target_rad"] = np.abs(left[:, cursor + 1 + joints:]
                                              - right[:, cursor + 1 + joints:]).max(1)
    limits = dict(joint_position_rad=1e-4, joint_velocity_rad_s=1e-3, root_position_m=5e-5,
                  root_quaternion=1e-4, root_velocity=1e-3, grip_command=0., effort_Nm=1e-3,
                  joint_target_rad=1e-4)
    valid = np.ones(len(left), dtype=bool)
    for key, values in differences.items():
        valid &= values <= limits[key]
    return valid, {key: float(value.max()) for key, value in differences.items()}, limits


def preference(gap, physical_effect, noise, threshold):
    """Score only effects larger than the declared threshold and repeat noise."""
    informative = np.abs(physical_effect) > np.maximum(threshold, 2 * np.abs(noise))
    if not informative.any():
        return dict(informative=0, direction_agreement=None, ci95=None)
    preference = ((gap[informative] * physical_effect[informative] > 0)
                  + .5 * (gap[informative] == 0))
    rng = np.random.default_rng(2026110133)
    indices = rng.integers(len(preference), size=(10000, len(preference)))
    return dict(informative=int(informative.sum()), direction_agreement=float(preference.mean()),
                ci95=np.quantile(preference[indices].mean(1), [.025, .975]).tolist())


def retained(values):
    relative = np.linalg.norm(values["relative"][:, -1] - values["relative"][:, 0], axis=-1)
    return ((values["assembly"].sum(1) >= 1)
            | ((values["distance"][:, -1] < .04) & (relative < .02)
               & (values["height"][:, -1] >= values["height"][:, 0] - .03)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--rescore-reward", type=Path)
    parser.add_argument("--use-reward-discount", action="store_true",
                        help="Offline rescore only: use that reward's fitted discount, retaining acquisition metadata")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rescore_reward and args.output is None:
        parser.error("A rescore requires an explicit output path to preserve the recorded-reward analysis")
    plan = json.loads(args.plan.read_text())
    reports, arrays = {}, {}
    for branch in ("actual", "repeat", "hold"):
        reports[branch] = json.loads((args.directory / branch / "report.json").read_text())
        if not reports[branch]["completed"]:
            raise ValueError(f"Incomplete branch: {branch}")
        with np.load(args.directory / branch / "branches.npz", allow_pickle=False) as raw:
            arrays[branch] = {key: raw[key].copy() for key in raw.files}
    for key in ("reward_sha256", "policy_sha256", "gamma", "plan"):
        if len({report[key] for report in reports.values()}) != 1:
            raise ValueError(f"Branch configurations differ: {key}")
    actual, repeat, hold = (arrays[name] for name in ("actual", "repeat", "hold"))
    rescored = rescore_branches(arrays, args.rescore_reward) if args.rescore_reward else None
    repeated, repeat_errors, limits = matched_physics(actual["start_physics"], repeat["start_physics"])
    intervened, hold_errors, _ = matched_physics(actual["start_physics"], hold["start_physics"])
    np.testing.assert_array_equal(actual["states"][:, 0], repeat["states"][:, 0])
    np.testing.assert_array_equal(actual["states"][:, 0], hold["states"][:, 0])
    valid = repeated & intervened & actual["hold_feasible"] & hold["hold_feasible"]
    valid &= (actual["prefix_assembly"] < 1) & (repeat["prefix_assembly"] < 1) & (hold["prefix_assembly"] < 1)
    gamma = reports["actual"]["gamma"]
    acquisition_gamma = gamma
    if rescored and rescored["gamma"] != gamma:
        if not args.use_reward_discount:
            raise ValueError("Rescoring must retain the original macro discount unless explicitly comparing another fitted reward")
        gamma = rescored["gamma"]
    weights = gamma ** np.arange(plan["horizon"])
    returns = {name: (values["rewards"] * weights).sum(1) for name, values in arrays.items()}
    gaps = dict(
        immediate_reward=actual["rewards"][:, 0] - hold["rewards"][:, 0],
        short_return=returns["actual"] - returns["hold"],
        short_return_with_frozen_v_tail=returns["actual"] - returns["hold"]
            + gamma ** plan["horizon"] * (actual["end_v"] - hold["end_v"]),
        recovered_q=actual["start_q"] - hold["start_q"],
    )
    if rescored:
        target_returns = {name: (values["bellman_targets"] * weights).sum(1)
                          for name, values in arrays.items()}
        gaps["actual_transition_bellman_target_return"] = target_returns["actual"] - target_returns["hold"]
    group_names = np.asarray([row["group"] for row in plan["rows"]])
    eligible = np.ones(len(valid), dtype=bool)
    eligible[group_names == "approach"] = ((actual["distance"][:, 0] > .06)
        & (actual["distance"][:, 0] < .20))[group_names == "approach"]
    eligible[group_names == "held_leg"] = ((actual["height"][:, 0] > .06)
        & (actual["distance"][:, 0] < .04) & (actual["width"][:, 0] > .015)
        & (actual["width"][:, 0] < .045))[group_names == "held_leg"]
    eligible[group_names == "recorded_height_drop"] = (actual["height"][:, 0] > .06)[group_names == "recorded_height_drop"]
    valid &= eligible
    groups = {}
    for name in np.unique(group_names):
        selected = valid & (group_names == name)
        if name == "approach":
            outcome = {key: values["distance"][:, 0] - values["distance"][:, 1:].min(1)
                       for key, values in arrays.items()}
            immediate = {key: values["distance"][:, 0] - values["distance"][:, 1]
                         for key, values in arrays.items()}
            threshold = .005
            units = "m of best TCP-to-leg approach within the branch window"
        else:
            outcome = {key: retained(values).astype(float) for key, values in arrays.items()}
            immediate = {key: (values["height"][:, 1] - values["height"][:, 0]) for key, values in arrays.items()}
            threshold = .5
            units = "retained-leg motion proxy or assembly event; not measured grasp contact"
        effect = outcome["actual"] - outcome["hold"]
        noise = outcome["actual"] - outcome["repeat"]
        row = dict(
            probes=int((group_names == name).sum()), valid_pre_states=int(selected.sum()),
            outcome_definition=units,
        )
        if selected.any():
            row.update(
                actual_mean_outcome=float(outcome["actual"][selected].mean()),
                hold_mean_outcome=float(outcome["hold"][selected].mean()),
                mean_physical_effect=float(effect[selected].mean()),
                mean_immediate_physical_effect=float((immediate["actual"] - immediate["hold"])[selected].mean()),
                repeat_absolute_effect_p95=float(np.quantile(np.abs(noise[selected]), .95)),
                native_assembly_events={key: int((values["assembly"][selected].sum(1) >= 1).sum())
                                        for key, values in arrays.items()},
                reward_direction={key: preference(value[selected], effect[selected], noise[selected], threshold)
                                  for key, value in gaps.items()},
            )
        groups[str(name)] = row
    result = dict(
        groups=groups, valid_pre_states=int(valid.sum()), probes=len(valid),
        rescored_reward=rescored,
        acquisition_gamma=acquisition_gamma, scored_gamma=gamma,
        maximum_pre_state_errors=dict(repeat=repeat_errors, hold=hold_errors),
        pre_state_limits=limits, policy_updated=False, reward_updated=False,
        caveats=[
            "Only one chunk is intervened on, followed by seven frozen-BC replanning chunks.",
            "Retrospective cohorts and diagnostic motion proxies do not establish full task policy performance.",
            "Short returns include only eight chunks. Frozen V tail and recovered Q are inferred quantities, not true values or the PPO critic.",
            "An informative effect must exceed both its declared magnitude threshold and twice the same-action repeat discrepancy.",
        ],
    )
    output = args.output or args.directory / "comparison.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
