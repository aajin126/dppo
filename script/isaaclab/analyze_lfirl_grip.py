"""Check whether closed-grip interventions improve retention and earn reward."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.isaaclab.analyze_lfirl_branches import matched_physics, preference, retained, rescore_branches


def grip_pairs(original, closed, plan):
    valid, errors, limits = matched_physics(original["start_physics"], closed["start_physics"])
    valid &= (original["prefix_assembly"] < 1) & (closed["prefix_assembly"] < 1)
    grip_column = original["start_physics"].shape[1] - 19
    valid &= original["start_physics"][:, grip_column] > .019
    groups = np.asarray([row["group"] for row in plan["rows"]])
    np.testing.assert_array_equal(original["states"][:, 0], closed["states"][:, 0])
    return valid, groups, errors, limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--closed", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    arrays, reports = {}, {}
    for label, directory in (("original", args.original), ("closed", args.closed)):
        reports[label] = json.loads((directory / "report.json").read_text())
        if not reports[label]["completed"]:
            raise ValueError("Both physical branches must complete")
        with np.load(directory / "branches.npz", allow_pickle=False) as raw:
            arrays[label] = {key: raw[key].copy() for key in raw.files}
    for key in ("policy_sha256", "gamma", "plan", "reward_sha256"):
        if reports["original"][key] != reports["closed"][key]:
            raise ValueError(f"Intervention changed {key}")
    original, closed = arrays["original"], arrays["closed"]
    valid, groups, errors, limits = grip_pairs(original, closed, plan)
    score_metadata = rescore_branches(arrays, args.reward)
    gamma = score_metadata["gamma"]
    weights = gamma ** np.arange(plan["horizon"])
    sums = {key: (value["rewards"] * weights).sum(1) for key, value in arrays.items()}
    targets = {key: (value["bellman_targets"] * weights).sum(1) for key, value in arrays.items()}
    outcomes = {key: retained(value).astype(float) for key, value in arrays.items()}
    results = {}
    for group in ("held_leg", "recorded_height_drop"):
        selected = valid & (groups == group)
        selected &= (original["height"][:, 0] > .06) & (original["distance"][:, 0] < .05)
        row = dict(valid_initially_closed_pre_states=int(selected.sum()))
        if selected.any():
            effect = (outcomes["closed"] - outcomes["original"])[selected]
            row.update(
                original_retained=int(outcomes["original"][selected].sum()),
                closed_retained=int(outcomes["closed"][selected].sum()),
                improved=int((effect > 0).sum()), worsened=int((effect < 0).sum()),
                original_median_end_width_m=float(np.median(original["width"][selected, -1])),
                closed_median_end_width_m=float(np.median(closed["width"][selected, -1])),
                original_open_end_count=int((original["width"][selected, -1] > .05).sum()),
                closed_open_end_count=int((closed["width"][selected, -1] > .05).sum()),
                reward_direction=preference((sums["closed"] - sums["original"])[selected], effect,
                                            np.zeros_like(effect), .5),
                target_direction=preference((targets["closed"] - targets["original"])[selected], effect,
                                            np.zeros_like(effect), .5),
                native_assembly_events={key: int((value["assembly"][selected].sum(1) >= 1).sum())
                                        for key, value in arrays.items()},
            )
        results[group] = row
    report = dict(
        results=results, reward=score_metadata, maximum_pre_state_errors=errors, pre_state_limits=limits,
        policy_updated=False, reward_updated=False,
        interpretation="Gripper commands alone are forced closed for eight chunks; pose commands replan with frozen BC. Retention is a geometry proxy, not measured contact or assembly success.",
        caveat="Only physically matched, initially closed, nonassembled pre-states are included. The same-action repeat branch was exact in the original paired probe.",
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
