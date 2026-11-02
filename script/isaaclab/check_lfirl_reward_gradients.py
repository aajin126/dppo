"""Rescore real on-policy paths before committing to another policy update."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def peer_baseline(returns, valid):
    """Use other independent environments at the same control index.

    The leave-one-out arithmetic follows https://huggingface.co/docs/trl/rloo_trainer.
    These peers have different reset states; this is a diagnostic time baseline,
    not a reproduction of prompt-group RLOO.
    """
    import numpy as np

    observed = np.where(valid, returns, 0.)
    count = valid.sum(axis=1, keepdims=True)
    baseline = np.divide(observed.sum(axis=1, keepdims=True) - observed, count - 1,
                         out=np.zeros_like(observed), where=count > 1)
    return np.where(valid, returns - baseline, 0.)


def main():
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from agent.finetune.lfirl.episodes import assembly_boundaries, finite_episode_returns
    from agent.finetune.lfirl.reward import LFIRLReward, sha256, validate_reward_provenance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], help="label=/absolute/path/reward.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.audit / "config.yaml")
    cfg.device = cfg.model.device = args.device
    guard = json.loads((args.audit / "mechanism/actor_guard_2.json").read_text())
    if not guard["actor_weights_identical"] or guard["actor_optimizer_steps"]:
        raise ValueError("The audit actor must still equal its original BC initialization")
    checkpoint = args.audit / "checkpoint/state_2.pt"
    weights = torch.load(checkpoint, map_location=args.device, weights_only=True)["model"]
    model = hydra.utils.instantiate(cfg.model).eval()
    model.load_state_dict(weights, strict=True)
    with np.load(args.audit / "mechanism/rollout_2.npz") as raw:
        observations, actions = raw["observations"].copy(), raw["actions"].copy()
        events = raw["assembly_rewards"].copy()
    valid, _ = assembly_boundaries(events, 1)
    goal_returns = finite_episode_returns(events, valid, cfg.train.gamma)
    states = observations.reshape(-1, 1, cfg.obs_dim)
    identities, duplicated = {}, set()
    for index, state in enumerate(states):
        key = state.tobytes()
        if key in identities:
            duplicated.add(key)
        else:
            identities[key] = index
    batches, goals, recorded = [], [], []
    parameters = tuple(model.actor_ft.parameters())

    def gradient(batch, advantages, anchor=False):
        obs, previous, following, denoising, logprobs = batch
        zeros = torch.zeros(len(previous), device=args.device)
        loss = model.loss(
            obs, previous, following, denoising, zeros, zeros,
            torch.as_tensor(advantages, device=args.device).float().clone(), logprobs,
            use_bc_loss=anchor, reward_horizon=cfg.act_steps,
        )
        objective = loss[0] + (cfg.train.bc_loss_coeff * loss[6] if anchor else 0.)
        derivatives = torch.autograd.grad(objective, parameters, allow_unused=True)
        return torch.cat([(value if value is not None else torch.zeros_like(parameter)).detach().flatten()
                          for value, parameter in zip(derivatives, parameters)])

    def cosine(left, right):
        return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-20))

    for path in sorted((args.audit / "mechanism").glob("credit_batch_2_*.npz")):
        with np.load(path) as raw:
            keep = np.array([state.tobytes() not in duplicated for state in raw["states"]])
            indices = np.array([identities[state.tobytes()] for state in raw["states"][keep]])
            np.testing.assert_allclose(goal_returns.reshape(-1)[indices], raw["goal_returns"][keep], rtol=1e-6)
            batch = (
                {"state": torch.as_tensor(raw["states"][keep], device=args.device)},
                *(torch.as_tensor(raw[key][keep], device=args.device)
                  for key in ("previous", "following", "denoising", "old_logprobs")),
            )
            with torch.no_grad():
                current = model.get_logprobs_subsample(
                    batch[0], batch[1], batch[2], batch[3],
                )
                torch.testing.assert_close(current, batch[4], rtol=1e-4, atol=1e-4)
            goals.append(gradient(batch, raw["goal_returns"][keep]))
            recorded.append(raw["learned_advantages"][keep].copy())
            batches.append((batch, indices, int((~keep).sum())))
    if not batches:
        raise ValueError("No captured native denoising-path minibatches")
    cases = {"recorded_gae": recorded}
    rewards = dict(baseline=Path(cfg.lfirl.reward_checkpoint))
    rewards.update({name: Path(path) for name, path in (value.split("=", 1) for value in args.candidate)})
    metadata = {}
    for name, path in rewards.items():
        saved = torch.load(path, map_location=args.device, weights_only=True)
        validate_reward_provenance(saved, sha256(cfg.normalization_path), sha256(cfg.base_policy_path))
        if saved["gamma"] != cfg.train.gamma or saved["horizon_steps"] != cfg.act_steps:
            raise ValueError("Candidate reward time scale differs from this actual rollout")
        reward = LFIRLReward(saved["learner"]["settings"], cfg.obs_dim, cfg.act_steps * cfg.action_dim, args.device)
        reward.load_state_dict(saved["learner"])
        prediction = np.concatenate([
            reward.predict(states[start:start + 1024], actions.reshape(len(states), -1)[start:start + 1024])
            for start in range(0, len(states), 1024)
        ]).reshape(events.shape)
        returns = finite_episode_returns(prediction, valid, cfg.train.gamma)
        peers = peer_baseline(returns, valid)
        cases[f"{name}/finite_zero"] = [returns.reshape(-1)[indices] for _, indices, _ in batches]
        cases[f"{name}/finite_peers"] = [peers.reshape(-1)[indices] for _, indices, _ in batches]
        success = events.sum(0) >= 1
        metadata[name] = dict(path=str(path), sha256=sha256(path), native_successes=int(success.sum()),
                              mean_initial_return_success=float(returns[0, success].mean()),
                              mean_initial_return_failure=float(returns[0, ~success].mean()))
        del reward
    results = {}
    for name, advantages in cases.items():
        rows = []
        for index, ((batch, _, omitted), values, goal) in enumerate(zip(batches, advantages, goals)):
            torch.manual_seed(2026110191 + index)
            raw = gradient(batch, values)
            torch.manual_seed(2026110191 + index)
            combined = gradient(batch, values, anchor=True)
            rows.append(dict(batch=index, omitted_ambiguous_states=omitted,
                             policy_gradient_cosine=cosine(raw, goal),
                             combined_gradient_cosine=cosine(combined, goal)))
        results[name] = dict(batches=rows, mean_combined_cosine=float(np.mean([r["combined_gradient_cosine"] for r in rows])))
        print(json.dumps(dict(case=name, **results[name])), flush=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    report = dict(results=results, rewards=metadata, policy_updated=False,
                  caveat="Exploratory gradient directions on three actual successful episodes; not blind policy evaluation or a guarantee of improvement. Finite returns terminate at native assembly or the recorded budget.")
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
