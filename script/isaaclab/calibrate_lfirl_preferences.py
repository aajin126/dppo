"""Seed-separated preference calibration of existing R layers, with frozen anchors."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/furniture-bench"))


def main():
    import numpy as np
    import torch
    from imitation.algorithms.preference_comparisons import (
        BasicRewardTrainer, CrossEntropyRewardLoss, PreferenceDataset, PreferenceModel,
    )
    from imitation.util import logger
    from agent.finetune.lfirl.dataset import MacroTransitions
    from agent.finetune.lfirl.preferences import (
        AnchoredPreferenceLoss, PreferenceRewardAdapter, load_outcome_pairs,
        load_curriculum_pairs, select_trainable_reward_layers,
    )
    from agent.finetune.lfirl.reward import LFIRLReward, sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trainable-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--curriculum-pairs", action="store_true",
                        help="Use real same-donor branches split by original source episode")
    parser.add_argument("--relative-head", action="store_true",
                        help="Fit only the existing 12-parameter relative correction")
    parser.add_argument("--anchor-coefficient", type=float, default=.1,
                        help="Preserve original reward predictions on expert and training anchors")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    if args.learning_rate <= 0 or args.steps < 25 or args.steps % 25 or args.anchor_coefficient < 0:
        parser.error("Use a positive learning rate and a positive multiple of 25 steps")
    args.output.mkdir(parents=True)
    torch.manual_seed(2026110201)
    rng = np.random.default_rng(2026110201)
    saved = torch.load(args.reward, map_location=args.device, weights_only=True)
    reward = LFIRLReward(saved["learner"]["settings"], saved["observation_dim"],
                         saved["horizon_steps"] * saved["action_dim"], args.device)
    reward.load_state_dict(saved["learner"])
    original = deepcopy(reward.state_dict())
    if args.curriculum_pairs:
        specification = json.loads(args.pairs.read_text())
        bank = json.loads((Path(specification["bank"]) / "report.json").read_text())
        if (bank["normalizer_sha256"] != saved["normalization_sha256"]
                or bank["source_bc_sha256"] != saved["source_bc_sha256"]):
            raise ValueError("Curriculum bank and reward use different BC or normalization")
    loaded = (load_curriculum_pairs if args.curriculum_pairs else load_outcome_pairs)(args.pairs)
    seeds = sorted({row["seed"] for row, _ in loaded}) if not args.curriculum_pairs else []
    if args.curriculum_pairs:
        training = [pair for row, pair in loaded if row["split"] == "training"]
        validation = [pair for row, pair in loaded if row["split"] == "validation"]
        if not training or not validation:
            raise ValueError("Require disjoint training and validation source episodes")
    else:
        if len(seeds) != 2:
            raise ValueError("This short protocol requires exactly two disjoint source seeds")
        training = [pair for row, pair in loaded if row["seed"] == seeds[0]]
        validation = [pair for row, pair in loaded if row["seed"] == seeds[1]]
    discount = saved["gamma"]
    gaps = []
    with torch.no_grad():
        for first, second in training:
            values = [reward.predict(fragment.obs[:-1], fragment.acts) for fragment in (first, second)]
            gaps.append(float(np.sum((values[0] - values[1]) * discount ** np.arange(len(first)))))
    temperature = max(float(np.median(np.abs(gaps))), 1.)
    expert = MacroTransitions(saved["expert_dataset_path"], saved["horizon_steps"],
                              saved["data"]["terminal_mode"], max_episodes=saved["data"]["expert_episodes"],
                              window_stride=saved["data"]["window_stride"])
    selected = rng.choice(len(expert.actions), 512, replace=False)
    anchor_states = expert.observations[selected].to(args.device)
    anchor_actions = expert.actions[selected].to(args.device)
    if args.curriculum_pairs:
        states = np.concatenate([fragment.obs[:-1] for pair in training for fragment in pair])[:, None]
        actions = np.concatenate([fragment.acts for pair in training for fragment in pair])
    else:
        training_archive = next(row["failure_trajectory"] for row, _ in loaded if row["seed"] == seeds[0])
        with np.load(training_archive) as raw:
            states = raw["pre_states"].reshape(-1, 1, saved["observation_dim"])
            actions = raw["actions"].reshape(len(states), -1)
    if len(states):
        selected = rng.choice(len(states), min(len(states), 512), replace=False)
        anchor_states = torch.cat((anchor_states, torch.as_tensor(states[selected], device=args.device)))
        anchor_actions = torch.cat((anchor_actions, torch.as_tensor(actions[selected], device=args.device)))
    with torch.no_grad():
        targets = reward.r(anchor_states, anchor_actions).detach()
    scale = max(float(targets.std()), 1.)
    trainable_prefixes = select_trainable_reward_layers(reward.r, args.trainable_layers,
                                                       relative_only=args.relative_head)
    trainable_parameters = sum(parameter.numel() for parameter in reward.r.parameters()
                               if parameter.requires_grad)
    model = PreferenceModel(
        PreferenceRewardAdapter(reward.r, saved["observation_dim"], saved["horizon_steps"] * saved["action_dim"], temperature),
        discount_factor=discount,
    )
    loss = AnchoredPreferenceLoss(anchor_states, anchor_actions, targets, scale,
                                 coefficient=args.anchor_coefficient)
    trainer = BasicRewardTrainer(
        model, loss, rng, batch_size=len(training), epochs=25, lr=args.learning_rate,
        custom_logger=logger.configure(str(args.output / "training"), format_strs=["json"]),
    )
    dataset = PreferenceDataset()
    dataset.push(training, np.ones(len(training), dtype=np.float32))
    criterion = CrossEntropyRewardLoss()
    history, best, best_state = [], None, None

    def metrics(pairs):
        with torch.no_grad():
            result = criterion(pairs, np.ones(len(pairs), dtype=np.float32), model)
            return dict(cross_entropy=float(result.loss), accuracy=float(result.metrics["accuracy"]))

    for step in range(0, args.steps + 1, 25):
        if step:
            trainer.train(dataset)
        with torch.no_grad():
            drift = float(((reward.r(anchor_states, anchor_actions) - targets) / scale).square().mean().sqrt())
        row = dict(step=step, training=metrics(training), validation=metrics(validation), normalized_output_drift=drift)
        history.append(row)
        if best is None or (drift <= .25 and row["validation"]["cross_entropy"] < best["validation"]["cross_entropy"]):
            best, best_state = row, deepcopy(reward.state_dict())
        print(json.dumps(row), flush=True)
    baseline = history[0]
    passed = (best["step"] > 0 and best["validation"]["cross_entropy"] <= baseline["validation"]["cross_entropy"] - .02
              and best["validation"]["accuracy"] >= baseline["validation"]["accuracy"])
    reward.load_state_dict(best_state)
    for name in ("q", "b", "v"):
        for key, value in original["models"][name].items():
            torch.testing.assert_close(value, best_state["models"][name][key], rtol=0, atol=0)
    for key, value in original["models"]["r"].items():
        if not key.startswith(trainable_prefixes):
            torch.testing.assert_close(value, best_state["models"]["r"][key], rtol=0, atol=0)
    metadata = dict(
        method="LFIRL + native-outcome preference calibration", implementation="imitation.PreferenceModel + CrossEntropyRewardLoss + BasicRewardTrainer",
        parent_reward=str(args.reward.resolve()), parent_sha256=sha256(args.reward),
        pair_manifest=str(args.pairs.resolve()), pair_manifest_sha256=sha256(args.pairs),
        training_seed=seeds[0] if seeds else None, validation_seed=seeds[1] if seeds else None,
        training_pairs=len(training), validation_pairs=len(validation),
        split_unit="original_source_episode" if args.curriculum_pairs else "seed",
        maximum_initial_observation_difference=max(row["maximum_initial_observation_difference"]
                                                  for row, _ in loaded) if args.curriculum_pairs else 0.,
        maximum_relative_physical_difference=max(row["maximum_relative_physical_difference"]
                                                for row, _ in loaded) if args.curriculum_pairs else 0.,
        source_groups={split: sorted({row["source_episode"] for row, _ in loaded if row["split"] == split})
                       for split in ("training", "validation")} if args.curriculum_pairs else None,
        temperature=temperature, anchor_scale=scale, anchor_coefficient=args.anchor_coefficient,
        trainable_parameters=trainable_parameters, trainable_layer_prefixes=list(trainable_prefixes),
        learning_rate=args.learning_rate, requested_steps=args.steps,
        selected=best, baseline=baseline, history=history, validation_gate_passed=passed,
        policy_trained=False, native_reward_bonus_added=False,
        caveat="Validation pairs select a checkpoint. A separate reward-direction check and fresh policy evaluation are still required.",
    )
    if passed:
        payload = dict(saved, learner=best_state, preference_calibration=metadata,
                       algorithm=saved["algorithm"] + " + native-outcome preference calibration")
        torch.save(payload, args.output / "selected_reward.pt")
    (args.output / "report.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(passed=passed, selected=best, baseline=baseline)), flush=True)


if __name__ == "__main__":
    main()
