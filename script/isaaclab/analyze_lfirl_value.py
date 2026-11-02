"""Inspect calibrated native critic credit on controlled gripper branches."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from util.reward_scaling import RunningRewardScaler
    from script.isaaclab.analyze_lfirl_grip import grip_pairs
    from script.isaaclab.analyze_lfirl_branches import preference, retained
    from agent.finetune.lfirl.reward import sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value-run", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--closed", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, action="append")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.value_run / "config.yaml")
    plan = json.loads(args.plan.read_text())
    arrays = {}
    for label, directory in (("original", args.original), ("closed", args.closed)):
        report = json.loads((directory / "report.json").read_text())
        if not report["completed"] or report["reward_sha256"] != sha256(cfg.lfirl.reward_checkpoint):
            raise ValueError("Branch reward and native critic reward differ")
        with np.load(directory / "branches.npz", allow_pickle=False) as raw:
            arrays[label] = {key: raw[key].copy() for key in raw.files}
    original, closed = arrays["original"], arrays["closed"]
    valid, groups, _, _ = grip_pairs(original, closed, plan)
    valid &= (original["height"][:, 0] > .06) & (original["distance"][:, 0] < .05)
    outcomes = retained(closed).astype(float) - retained(original).astype(float)
    critic = hydra.utils.instantiate(cfg.model.critic).eval()
    results = []
    paths = args.value_checkpoint or sorted((args.value_run / "mechanism").glob("value_*.pt"))
    for path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if not saved["actor_weights_identical"] or saved["actor_optimizer_steps"] != 0:
            raise ValueError("This analysis requires frozen actor weights")
        critic.load_state_dict(saved["critic"], strict=True)
        scaler = RunningRewardScaler(saved["environments"])
        scaler.load_state_dict(saved["reward_scaler"])
        advantages, vstart = {}, {}
        with torch.no_grad():
            for label, values in arrays.items():
                first = critic(torch.from_numpy(values["states"][:, :1])).reshape(-1).numpy()
                last = critic(torch.from_numpy(values["states"][:, -1:])).reshape(-1).numpy()
                rewards = scaler.transform(values["rewards"])
                prefix = (rewards * saved["gamma"] ** np.arange(rewards.shape[1])).sum(1)
                advantages[label] = prefix + saved["gamma"] ** rewards.shape[1] * last - first
                vstart[label] = first
        np.testing.assert_array_equal(vstart["original"], vstart["closed"])
        gap = advantages["closed"] - advantages["original"]
        record = dict(checkpoint=str(path), actor_updated=False, groups={})
        for group in ("held_leg", "recorded_height_drop"):
            selected = valid & (groups == group)
            record["groups"][group] = preference(gap[selected], outcomes[selected],
                                                np.zeros(int(selected.sum())), .5)
        results.append(record)
    if not args.value_checkpoint and len(results) != 2:
        raise ValueError("Both guarded native value-calibration iterations must finish")
    result = dict(
        value_calibration=results, actor_updated=False,
        interpretation="Eight-step bootstrapped credit with the actually calibrated native DPPO critic and its saved reward scaler. Not exact one-step GAE or a policy-improvement claim.",
        source_GAE_files=[str(path) for path in sorted((args.value_run / "mechanism").glob("advantages_*.json"))],
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
