"""Read-only probes linking recovered values to actual manipulation progress.

Hold-command comparisons are counterfactual network queries, not simulated
outcomes. Immediate reward preference alone is not a policy-improvement test.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from agent.finetune.lfirl.reward import LFIRLReward
    from env.gym_utils.furniture_normalizer import LinearNormalizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (saved["observation_dim"], saved["action_dim"], saved["horizon_steps"]) != (58, 10, 8):
        raise ValueError("This probe requires the existing One-leg observation/action contract")
    reward = LFIRLReward(saved["learner"]["settings"], 58, 80, "cpu")
    reward.load_state_dict(saved["learner"])
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(torch.load(saved["normalization_path"], map_location="cpu", weights_only=True))
    parts = {}
    for path in sorted(args.rollouts.glob("bc_seed_*.npz")):
        with np.load(path, allow_pickle=False) as raw:
            for key in ("pre_states", "actions", "leg_height_gain_m", "tcp_leg_distance_m",
                        "gripper_width_m", "tcp_leg_relative_position_m", "assembly_rewards"):
                parts.setdefault(key, []).append(raw[key].copy())
    if not parts:
        raise ValueError("No frozen BC rollout archives found")
    data = {key: np.concatenate(value) for key, value in parts.items()}
    # t >= 1 has measured pre- and post-action geometry. Omit the final
    # transition whose actual next observation was not recorded.
    states = data["pre_states"][:, 1:-1]
    following = data["pre_states"][:, 2:]
    commands = data["actions"][:, 1:-1]
    shape = states.shape[:2]
    obs = torch.from_numpy(states.reshape(-1, 1, 58)).float()
    next_obs = torch.from_numpy(following.reshape(-1, 1, 58)).float()
    actions = torch.from_numpy(commands.reshape(-1, 8, 10)).float()
    physical = normalizer(obs[:, 0], "observations", forward=False)
    physical_actions = normalizer(actions, "actions", forward=False)
    held = physical_actions.clone()
    held[:, :, :9] = physical[:, None, :9]
    hold_actions = normalizer(held, "actions", forward=True)
    feasible_hold = (hold_actions.abs() <= 1 + 1e-5).all(dim=(1, 2)).numpy().reshape(shape)
    hold_actions = hold_actions.clamp(-1, 1)
    moving = ((physical_actions[:, :, :3] - physical[:, None, :3]).norm(dim=-1).amax(1) > .003).numpy().reshape(shape)
    collected = {name: [] for name in ("reward", "hold_reward", "q", "hold_q", "target")}
    with torch.no_grad():
        for start in range(0, len(obs), 1024):
            end = start + 1024
            state, command = obs[start:end], actions[start:end].reshape(-1, 80)
            hold = hold_actions[start:end].reshape(-1, 80)
            q = reward.q(state, command).reshape(-1) + reward.b(state).reshape(-1)
            hold_q = reward.q(state, hold).reshape(-1) + reward.b(state).reshape(-1)
            target = q - saved["gamma"] * reward.v(next_obs[start:end]).reshape(-1)
            batch = {
                "reward": reward.r(state, command).reshape(-1),
                "hold_reward": reward.r(state, hold).reshape(-1),
                "q": q, "hold_q": hold_q, "target": target,
            }
            for name, value in batch.items():
                collected[name].append(value.numpy())
    scores = {name: np.concatenate(value).reshape(shape) for name, value in collected.items()}
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise FloatingPointError("Nonfinite mechanism probe")
    pre_distance = data["tcp_leg_distance_m"][:, :-2]
    post_distance = data["tcp_leg_distance_m"][:, 1:-1]
    pre_height = data["leg_height_gain_m"][:, :-2]
    post_height = data["leg_height_gain_m"][:, 1:-1]
    width = data["gripper_width_m"][:, 1:-1]
    relative_slip = np.linalg.norm(np.diff(data["tcp_leg_relative_position_m"], axis=1)[:, :-1], axis=-1)
    masks = {
        "all_moving_commands": moving,
        "approach_progress": moving & (pre_distance > .05) & (pre_distance - post_distance > .01),
        "approach_stalled": moving & (pre_distance > .05) & (np.abs(pre_distance - post_distance) < .002),
        "coupled_upward_motion": moving & (post_height - pre_height > .02) & (post_distance < .04)
            & (width > .015) & (width < .045) & (relative_slip < .01),
        "leg_drop": (pre_height > .06) & (pre_height - post_height > .02),
        "assembly_event": data["assembly_rewards"][:, 1:-1] >= 1,
    }
    groups = {}
    for name, mask in masks.items():
        selected = mask & feasible_hold
        row = dict(recorded_transitions=int(mask.sum()), feasible_hold_transitions=int(selected.sum()),
                   episodes=int(mask.any(1).sum()))
        if selected.any():
            for value_name in ("reward", "q"):
                gap = (scores[value_name] - scores["hold_" + value_name])[selected]
                row[value_name + "_actual_over_hold_fraction"] = float(np.mean((gap > 0) + .5 * (gap == 0)))
                row[value_name + "_actual_minus_hold_median"] = float(np.median(gap))
        if mask.any():
            row["actual_reward_mean"] = float(scores["reward"][mask].mean())
        groups[name] = row
    target, predicted = scores["target"].reshape(-1), scores["reward"].reshape(-1)
    mse = float(np.mean((target - predicted)**2))
    report = dict(
        checkpoint=str(args.checkpoint), rollout_directory=str(args.rollouts),
        episodes=shape[0], known_transitions=int(np.prod(shape)),
        target_fit=dict(target_std=float(target.std()), prediction_std=float(predicted.std()),
                        rmse=mse**.5, r_squared=1 - mse / max(float(target.var()), 1e-12)),
        groups=groups,
        policy_updated=False, simulator_run=False,
        caveats=[
            "Geometry categories are descriptive proxies, not annotated skills or verified contact grasps.",
            "Hold commands retain the executed gripper commands and hold all eight TCP poses at the measured pre-action pose.",
            "Out-of-box hold queries are excluded from preferences; no counterfactual is physically simulated.",
            "Immediate reward need not favor every useful action. Frozen Q preference tests the recovered long-term ordering; it is not the current PPO critic or proof of a policy improvement.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
