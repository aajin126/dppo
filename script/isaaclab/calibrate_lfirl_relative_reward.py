"""Fit a small relative-pose reward correction with the original preference loss.

The LFIRL parent and policy remain frozen. Separate outcome validation and an
actual policy training/evaluation experiment are required before adoption.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "external/furniture-bench")]


def main():
    import numpy as np
    import torch
    from agent.finetune.lfirl.dataset import MacroTransitions
    from agent.finetune.lfirl.preferences import (
        AnchoredPreferenceLoss, PreferenceRewardAdapter, load_outcome_pairs,
    )
    from agent.finetune.lfirl.relative_reward import RelativeRewardResidual
    from agent.finetune.lfirl.reward import LFIRLReward, sha256
    from imitation.algorithms.preference_comparisons import CrossEntropyRewardLoss, PreferenceModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing calibrations; use a new output directory")
    args.output.mkdir(parents=True)
    torch.manual_seed(2026110331)
    rng = np.random.default_rng(2026110331)
    saved = torch.load(args.reward, map_location=args.device, weights_only=True)
    if sha256(args.normalizer) != saved["normalization_sha256"]:
        raise ValueError("Relative features and LFIRL use different policy normalization")
    base = LFIRLReward(saved["learner"]["settings"], 58, 80, args.device)
    base.load_state_dict(saved["learner"])
    original = {key: value.clone() for key, value in base.r.state_dict().items()}
    loaded = load_outcome_pairs(args.pairs)
    seeds = sorted({row["seed"] for row, _ in loaded})
    if len(seeds) != 2:
        raise ValueError("Use two separate initial-scene seeds for fitting and selection")
    training = [pair for row, pair in loaded if row["seed"] == seeds[0]]
    validation = [pair for row, pair in loaded if row["seed"] == seeds[1]]
    expert = MacroTransitions(
        saved["expert_dataset_path"], 8, saved["data"]["terminal_mode"],
        max_episodes=saved["data"]["expert_episodes"],
        window_stride=saved["data"]["window_stride"],
    )
    indices = rng.choice(len(expert.actions), 512, replace=False)
    anchor_states = expert.observations[indices].to(args.device)
    anchor_actions = expert.actions[indices].to(args.device)
    archive = next(row["failure_trajectory"] for row, _ in loaded if row["seed"] == seeds[0])
    with np.load(archive) as raw:
        states = raw["pre_states"].reshape(-1, 1, 58)
        actions = raw["actions"].reshape(len(states), 80)
        indices = rng.choice(len(states), 512, replace=False)
        anchor_states = torch.cat((anchor_states, torch.as_tensor(states[indices], device=args.device).float()))
        anchor_actions = torch.cat((anchor_actions, torch.as_tensor(actions[indices], device=args.device).float()))
    with torch.no_grad():
        targets = base.r(anchor_states, anchor_actions).detach()
        scale = max(float(targets.std()), 1.)
        gaps = []
        for first, second in training:
            values = [base.predict(fragment.obs[:-1], fragment.acts).reshape(-1)
                      for fragment in (first, second)]
            gaps.append(float(np.sum((values[0] - values[1])
                                     * saved["gamma"] ** np.arange(len(first)))))
    temperature = max(float(np.median(np.abs(gaps))), 1.)
    reward = RelativeRewardResidual(
        base.r, torch.load(args.normalizer, map_location=args.device, weights_only=True),
    ).to(args.device)
    with torch.no_grad():
        reward.residual_scale.fill_(scale)
        reward.feature_center.copy_(reward.features(anchor_states).mean(0))
    model = PreferenceModel(PreferenceRewardAdapter(reward, 58, 80, temperature),
                            discount_factor=saved["gamma"])
    loss = AnchoredPreferenceLoss(anchor_states, anchor_actions, targets, scale)
    criterion = CrossEntropyRewardLoss()
    optimizer = torch.optim.LBFGS(
        [reward.linear.weight], lr=1., max_iter=200, tolerance_grad=1e-7,
        tolerance_change=1e-9, line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def metrics(pairs):
        with torch.no_grad():
            result = criterion(pairs, np.ones(len(pairs), np.float32), model)
            return dict(cross_entropy=float(result.loss), accuracy=float(result.metrics["accuracy"]))

    baseline = dict(training=metrics(training), validation=metrics(validation))

    def closure():
        nonlocal closure_calls
        optimizer.zero_grad()
        result = loss(training, np.ones(len(training), np.float32), model)
        if not torch.isfinite(result.loss):
            raise FloatingPointError("Nonfinite relative reward objective")
        result.loss.backward()
        if not torch.isfinite(reward.linear.weight.grad).all():
            raise FloatingPointError("Nonfinite relative reward gradient")
        closure_calls += 1
        return result.loss

    optimizer.step(closure)
    final_loss = closure()
    for key, value in original.items():
        torch.testing.assert_close(value, base.r.state_dict()[key], rtol=0, atol=0)
    with torch.no_grad():
        residual = (reward(anchor_states, anchor_actions) - targets) / scale
    selected = dict(training=metrics(training), validation=metrics(validation),
                    normalized_output_drift=float(residual.square().mean().sqrt()),
                    mean_normalized_anchor_residual=float(residual.mean()),
                    final_training_objective=float(final_loss.detach()),
                    maximum_absolute_training_gradient=float(reward.linear.weight.grad.abs().max()))
    passed = bool(selected["validation"]["accuracy"] >= .75
                  and selected["validation"]["cross_entropy"] < np.log(2.))
    report = dict(
        extension="Frozen LFIRL plus learned centered 12-parameter relative-pose residual",
        source_checkpoint=str(args.reward.resolve()), source_sha256=sha256(args.reward),
        pair_manifest=str(args.pairs.resolve()), pair_manifest_sha256=sha256(args.pairs),
        normalization_sha256=sha256(args.normalizer),
        trainable_parameters=12, training_seed=seeds[0], validation_seed=seeds[1],
        features="Centered relative translation / 0.1m, its coordinate-wise squares, and relative rotation row-6D; no target pose or assembly threshold",
        training_pairs=len(training), validation_pairs=len(validation),
        optimizer="PyTorch LBFGS on the existing imitation preference CE plus anchor loss",
        closure_calls=closure_calls, baseline=baseline, selected=selected,
        anchor_coefficient=.1, maximum_output_drift_sigma=None,
        drift_handling="Reported rather than a hard gate; retain soft preservation and require separate native-outcome validation",
        residual_output_scale=scale, temperature=temperature,
        validation_gate_passed=passed, policy_updated=False,
        current_policy_reward_changed=False,
        separate_reward_check_and_fresh_policy_evaluation_required=True,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    payload = dict(weights=reward.linear.state_dict(), center=reward.feature_center,
                   metadata=report)
    torch.save(payload, args.output / "diagnostic_residual.pt")
    if passed:
        torch.save(payload, args.output / "candidate_residual.pt")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
