"""Select a saved R head using excluded-episode frozen Bellman-target error."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from agent.finetune.lfirl.dataset import MacroTransitions, ReferenceRewardTransitions
    from agent.finetune.lfirl.reward import LFIRLReward

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    source = torch.load(args.run_dir / "reward.pt", map_location="cpu", weights_only=True)
    model = LFIRLReward(source["learner"]["settings"], source["observation_dim"],
                        source["action_dim"] * source["horizon_steps"], "cpu")
    model.load_state_dict(source["learner"])
    data = MacroTransitions(
        source["expert_dataset_path"], source["horizon_steps"], source["data"]["terminal_mode"],
        window_stride=source["data"]["window_stride"],
    )
    fitted_ids = np.unique(data.episode_ids)[:source["data"]["expert_episodes"]]
    fitted = np.isin(data.episode_ids, fitted_ids)
    if fitted.all():
        raise ValueError("Checkpoint selection requires excluded complete expert episodes")
    references = []
    with torch.no_grad():
        for obs, action, next_obs, done in data.batches(1024, "cpu"):
            target = (model.q(obs, action).reshape(-1) + model.b(obs).reshape(-1)
                      - source["gamma"] * (1 - done) * model.v(next_obs).reshape(-1))
            references.append(target.numpy())
    target = np.concatenate(references)
    limit = float(source["learner"]["settings"]["r_net"]["loss"].get("target_clip", 1e6))
    target = np.clip(target, -limit, limit)
    reference_data, reference_target = None, None
    if source.get("reward_reference_rollouts"):
        reference_data = ReferenceRewardTransitions(
            data, source["reward_reference_rollouts"], source["normalization_sha256"],
            source["source_bc_sha256"], source["data"]["reference_training_fraction"],
            assembly_terminal=source["data"].get("reference_assembly_terminal", False),
            success_threshold=source["data"].get("reference_success_threshold"),
        )
        parts = []
        with torch.no_grad():
            for obs, action, next_obs, done in reference_data.validation_batches(1024, "cpu"):
                reference = (model.q(obs, action).reshape(-1) + model.b(obs).reshape(-1)
                             - source["gamma"] * (1 - done) * model.v(next_obs).reshape(-1))
                parts.append(reference.clamp(-limit, limit).numpy())
        reference_target = np.concatenate(parts)
    paths = sorted(args.run_dir.glob("reward_step_*.pt"),
                   key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    paths.append(args.run_dir / "reward.pt")
    history, best, best_state = [], None, None
    for path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        learner = saved.get("learner", saved)
        for name in ("q", "b", "v"):
            for key, value in source["learner"]["models"][name].items():
                torch.testing.assert_close(learner["models"][name][key], value, rtol=0, atol=0)
        model.load_state_dict(learner)
        prediction = np.concatenate([model.predict(obs, action)
                                     for obs, action, _, _ in data.batches(1024, "cpu")])
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"Nonfinite R checkpoint: {path}")
        row = dict(checkpoint=str(path))
        for label, mask in (("fitted", fitted), ("excluded", ~fitted)):
            mse = float(np.mean((prediction[mask] - target[mask])**2))
            row[label] = dict(rmse=mse**.5, r_squared=1 - mse / max(float(target[mask].var()), 1e-12))
        row["selection_score"] = row["excluded"]["rmse"]**2 / max(float(target[~fitted].var()), 1e-12)
        if reference_data is not None:
            reference_prediction = np.concatenate([
                model.predict(obs, action)
                for obs, action, _, _ in reference_data.validation_batches(1024, "cpu")
            ])
            mse = float(np.mean((reference_prediction - reference_target)**2))
            normalized = mse / max(float(reference_target.var()), 1e-12)
            row["reference_excluded"] = dict(rmse=mse**.5, r_squared=1 - normalized)
            row["selection_score"] = .5 * (row["selection_score"] + normalized)
        history.append(row)
        if best is None or row["selection_score"] < best["selection_score"]:
            best, best_state = row, learner
        print(json.dumps(row), flush=True)
    result = dict(source)
    result["learner"] = best_state
    result["selection"] = dict(
        criterion=("Mean normalized Bellman-target MSE on excluded expert and reference episodes"
                   if reference_data is not None else "Excluded complete expert episode Bellman-target RMSE"),
        caveat=("These episodes are model-selection data, not a blind test. "
                "Native assembly events are used only as episode boundaries."
                if source["data"].get("reference_assembly_terminal", False) else
                "These episodes are model-selection data, not a blind test; assembly labels are not used."),
        selected=best, history=history,
    )
    output = args.run_dir / "selected_reward.pt"
    if output.exists():
        raise FileExistsError(output)
    torch.save(result, output)
    (args.run_dir / "selection.json").write_text(json.dumps(result["selection"], indent=2) + "\n")
    print(f"[Selected LFIRL reward] {output}", flush=True)


if __name__ == "__main__":
    main()
