"""Audit a frozen-policy critic against measured discounted rollout returns.

Default: audit only. Optional updates fit the existing native critic with
ordinary value MSE, excluding complete episodes before selecting a snapshot.
The policy and recovered reward are never optimized.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def discounted_returns(rewards, gamma):
    import numpy as np

    result = np.zeros_like(rewards)
    following = np.zeros(rewards.shape[1], dtype=rewards.dtype)
    for time in reversed(range(len(rewards))):
        following = result[time] = rewards[time] + gamma * following
    return result


def main():
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from agent.finetune.lfirl.reward import LFIRLReward
    from util.reward_scaling import RunningRewardScaler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Select a new output directory to preserve earlier audits")
    if args.updates < 0:
        parser.error("Updates must be nonnegative")
    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    source = args.run_dir / "mechanism/value_1.pt"
    frozen = torch.load(source, map_location="cpu", weights_only=True)
    if not frozen["actor_weights_identical"] or frozen["actor_optimizer_steps"]:
        raise ValueError("Audit requires unchanged actor weights")
    saved = torch.load(cfg.lfirl.reward_checkpoint, map_location="cpu", weights_only=True)
    reward = LFIRLReward(saved["learner"]["settings"], 58, 80, "cpu")
    reward.load_state_dict(saved["learner"])
    scaler = RunningRewardScaler(frozen["environments"])
    scaler.load_state_dict(frozen["reward_scaler"])
    all_obs, all_target, all_training = [], [], []
    for index in range(2):
        with np.load(args.run_dir / f"mechanism/rollout_{index}.npz") as raw:
            obs = raw["observations"].astype(np.float32)
            action = raw["actions"].astype(np.float32)
        shape = obs.shape[:2]
        predictions = np.concatenate([
            reward.predict(obs.reshape(-1, 1, 58)[start:start + 1024], action.reshape(-1, 80)[start:start + 1024])
            for start in range(0, int(np.prod(shape)), 1024)
        ]).reshape(shape)
        scaled = scaler.transform(predictions.T).T
        target = discounted_returns(scaled, frozen["gamma"])
        all_obs.append(obs.reshape(-1, 1, 58))
        all_target.append(target.reshape(-1))
        all_training.append(np.broadcast_to(np.arange(shape[1]) < int(.75 * shape[1]), shape).reshape(-1))
    observations = torch.from_numpy(np.concatenate(all_obs)).to(args.device)
    targets = torch.from_numpy(np.concatenate(all_target)).float().to(args.device)
    training = torch.from_numpy(np.concatenate(all_training)).to(args.device)
    critic = hydra.utils.instantiate(cfg.model.critic).to(args.device)
    critic.load_state_dict(frozen["critic"], strict=True)
    indices = torch.nonzero(training).reshape(-1)
    rng = torch.Generator(device=args.device).manual_seed(2026110141)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.train.critic_lr,
                                  weight_decay=cfg.train.critic_weight_decay) if args.updates else None
    history, best, best_state = [], None, None
    for step in range(args.updates + 1):
        if step % 100 == 0 or step == args.updates:
            with torch.no_grad():
                prediction = torch.cat([critic(batch).reshape(-1) for batch in observations.split(2048)])
            row = dict(updates=step)
            for name, mask in (("fitted_episodes", training), ("excluded_episodes", ~training)):
                error = prediction[mask] - targets[mask]
                mse = float(error.square().mean())
                row[name] = dict(rmse=mse**.5, mean_prediction=float(prediction[mask].mean()),
                                 mean_measured_return=float(targets[mask].mean()),
                                 r_squared=1 - mse / max(float(targets[mask].var(unbiased=False)), 1e-12))
            history.append(row)
            if best is None or row["excluded_episodes"]["rmse"] < best["excluded_episodes"]["rmse"]:
                best = row
                best_state = {key: value.detach().cpu().clone() for key, value in critic.state_dict().items()}
            print(json.dumps(row), flush=True)
        if step == args.updates:
            break
        rows = indices[torch.randint(len(indices), (1024,), generator=rng, device=args.device)]
        loss = .5 * (critic(observations[rows]).reshape(-1) - targets[rows]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite value-regression loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    args.output.mkdir(parents=True)
    report = dict(
        selected=best, history=history, actor_updated=False, reward_updated=False,
        target="Actual finite-horizon discounted recovered rewards, using the final native reward scaler; zero value at the observed timeout.",
        excluded_episode_count=128, fitted_episode_count=384,
        caveat="Exclusion applies to the additional Monte Carlo refit only; the preceding native GAE warmup already used these episodes. This tests value fitting, not blind policy success or true task-reward recovery.",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.updates:
        export = dict(frozen)
        export["critic"] = best_state
        export["value_calibration"] = report
        torch.save(export, args.output / "mc_value.pt")


if __name__ == "__main__":
    main()
