"""Export a validated relative correction without changing the LFIRL parent."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "external/furniture-bench")]


def main():
    import numpy as np
    import torch
    from agent.finetune.lfirl.preferences import load_outcome_pairs
    from agent.finetune.lfirl.reward import LFIRLReward, sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--direction-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing exports; choose a new output directory")
    candidate = torch.load(args.residual, map_location="cpu", weights_only=True)
    metadata = candidate["metadata"]
    validation = metadata["selected"]["validation"]
    if not (metadata["validation_gate_passed"] and validation["accuracy"] >= .75
            and validation["cross_entropy"] < np.log(2.)):
        raise ValueError("Relative reward did not pass model-selection validation")
    parent = Path(metadata["source_checkpoint"])
    if sha256(parent) != metadata["source_sha256"]:
        raise ValueError("LFIRL parent changed after fitting the correction")
    saved = torch.load(parent, map_location="cpu", weights_only=True)
    if sha256(args.normalizer) != saved["normalization_sha256"]:
        raise ValueError("Correction and policy normalizers differ")

    reports = []
    seen_seeds = set()
    count, preferred, ce_sum = 0, 0, 0.
    for path in args.direction_report:
        report = json.loads(path.read_text())
        result = report["results"][1]
        if (result["checkpoint_sha256"] != sha256(args.residual)
                or report["reward_updated"] or report["policy_updated"]):
            raise ValueError("Outcome checks must evaluate this exact frozen correction")
        if set(report["source_seeds"]) & {metadata["training_seed"], metadata["validation_seed"]}:
            raise ValueError("Independent outcome scenes overlap reward fitting or selection")
        if seen_seeds & set(report["source_seeds"]):
            raise ValueError("Outcome reports repeat initial-scene seeds")
        seen_seeds.update(report["source_seeds"])
        count += result["pairs"]
        preferred += result["native_success_preferred"]
        ce_sum += result["pairs"] * result["cross_entropy"]
        reports.append((path, report))
    if count < 12 or preferred / count < .75 or ce_sum / count >= np.log(2.):
        raise ValueError("Independent native-outcome direction check failed")

    settings = deepcopy(saved["learner"]["settings"])
    settings["relative_residual"] = dict(
        normalization_path=str(args.normalizer.resolve()),
        normalization_sha256=sha256(args.normalizer),
        feature_contract="one_leg_tabletop_to_leg4_translation_squared_translation_row6d",
    )
    reward = LFIRLReward(settings, saved["observation_dim"],
                         saved["horizon_steps"] * saved["action_dim"], "cpu")
    state = deepcopy(saved["learner"])
    state["settings"] = settings
    relative_state = reward.r.state_dict()
    for key, value in state["models"]["r"].items():
        relative_state[f"base.{key}"] = value.clone()
    relative_state["linear.weight"] = candidate["weights"]["weight"].clone()
    relative_state["feature_center"] = candidate["center"].clone()
    relative_state["residual_scale"] = torch.tensor(metadata["residual_output_scale"])
    state["models"]["r"] = relative_state
    reward.load_state_dict(state)

    maximum_gap_error = 0.
    for path, report in reports:
        manifest = path.with_name(path.name.replace("_reward_direction.json", "_native_goal_pairs.json"))
        pairs = load_outcome_pairs(manifest)
        reproduced = []
        for _, (success, failure) in pairs:
            values = [reward.predict(fragment.obs[:-1], fragment.acts).reshape(-1)
                      for fragment in (success, failure)]
            reproduced.append(float(np.sum((values[0] - values[1])
                                           * saved["gamma"] ** np.arange(len(success)))))
        expected = np.asarray(report["results"][1]["discounted_equal_window_reward_gaps"])
        np.testing.assert_allclose(reproduced, expected, rtol=1e-5, atol=1e-3)
        maximum_gap_error = max(maximum_gap_error, float(np.abs(expected - reproduced).max()))
    for name in ("q", "b", "v"):
        for key, value in saved["learner"]["models"][name].items():
            torch.testing.assert_close(value, state["models"][name][key], rtol=0, atol=0)
    for key, value in saved["learner"]["models"]["r"].items():
        torch.testing.assert_close(value, state["models"]["r"][f"base.{key}"], rtol=0, atol=0)

    provenance = dict(
        **metadata, method="LFIRL plus learned relative-pose preference correction",
        parent_reward=str(parent.resolve()), parent_sha256=sha256(parent),
        residual_checkpoint=str(args.residual.resolve()), residual_sha256=sha256(args.residual),
        independent_pairs=count, independent_success_preferred=preferred,
        independent_cross_entropy=ce_sum / count,
        independent_direction_reports=[dict(path=str(path.resolve()), sha256=sha256(path))
                                       for path, _ in reports],
        runtime_gap_parity_maximum_absolute_error=maximum_gap_error,
        policy_performance_improvement_verified=False,
    )
    payload = dict(saved, learner=state, preference_calibration=provenance,
                   algorithm="LFIRL + native-outcome preferences + relative-pose residual")
    args.output.mkdir(parents=True)
    torch.save(payload, args.output / "reward.pt")
    loaded = torch.load(args.output / "reward.pt", map_location="cpu", weights_only=True)
    restored = LFIRLReward(loaded["learner"]["settings"], 58, 80, "cpu")
    restored.load_state_dict(loaded["learner"])
    sample = pairs[0][1][0]
    np.testing.assert_array_equal(reward.predict(sample.obs[:-1], sample.acts),
                                  restored.predict(sample.obs[:-1], sample.acts))
    (args.output / "report.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(dict(exported=str(args.output / "reward.pt"), pairs=count,
                         success_preferred=preferred, runtime_gap_error=maximum_gap_error)), flush=True)


if __name__ == "__main__":
    main()
