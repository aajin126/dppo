"""Read-only, paired action sampling before and after one actor iteration."""

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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--device", help="Override the model device for read-only action sampling")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    if args.device:
        cfg.device = cfg.model.device = args.device
    plan = json.loads(args.plan.read_text())
    with np.load(args.probes) as raw:
        selected = np.array([row["group"] == "held_leg" for row in plan["rows"]])
        selected &= raw["start_physics"][:, -19] > .019
        states = raw["states"][selected, :1].copy()
    count = len(states)
    observations = np.repeat(states, args.draws, axis=0)
    model = hydra.utils.instantiate(cfg.model).eval()
    normalizer = torch.load(cfg.normalization_path, map_location="cpu", weights_only=True)
    low = float(normalizer["stats.actions.min"][9])
    high = float(normalizer["stats.actions.max"][9])
    results = {}
    for label, iteration in (("before", 1), ("after", 2)):
        saved = torch.load(args.run_dir / f"checkpoint/state_{iteration}.pt", map_location=cfg.device, weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        commands = []
        with torch.no_grad():
            for start in range(0, len(observations), 512):
                torch.manual_seed(2026110151 + start)
                obs = torch.as_tensor(observations[start:start + 512], device=cfg.device).float()
                action = model(cond={"state": obs}, deterministic=True).trajectories.cpu().numpy()
                commands.append((action[:, :, 9] + 1) * .5 * (high - low) + low)
        grip = np.concatenate(commands)
        memory, opened = np.ones(len(grip)), np.zeros(len(grip), dtype=bool)
        for time in range(grip.shape[1]):
            change = (np.sign(grip[:, time]) != np.sign(memory)) & (np.abs(grip[:, time]) > .019)
            memory[change] = grip[change, time]
            opened |= memory < 0
        results[label] = dict(
            any_open_fraction_by_state=opened.reshape(count, args.draws).mean(1),
            final_open_fraction_by_state=(memory < 0).reshape(count, args.draws).mean(1),
        )
    report = dict(states=count, draws_per_state=args.draws, results={}, policy_updated_by_probe=False,
                  caveat="Paired evaluation-mode policy samples on recorded held-leg states. Command changes are not simulated outcomes or proof of assembly improvement.")
    rng = np.random.default_rng(2026110152)
    samples = rng.integers(count, size=(10000, count))
    for metric in results["before"]:
        delta = results["after"][metric] - results["before"][metric]
        report["results"][metric] = dict(before=float(results["before"][metric].mean()),
                                        after=float(results["after"][metric].mean()),
                                        difference=float(delta.mean()),
                                        paired_state_bootstrap_ci95=np.quantile(delta[samples].mean(1), [.025, .975]).tolist())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
