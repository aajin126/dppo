"""Replay common prefixes, intervene once, and measure frozen-BC continuations.

The hold intervention changes TCP commands for one eight-action chunk only.
It retains the recorded gripper commands. No optimizer is constructed.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def make_plan(archive, horizon=8, per_group=64):
    import numpy as np

    with np.load(archive, allow_pickle=False) as raw:
        height = raw["leg_height_gain_m"]
        distance = raw["tcp_leg_distance_m"]
        width = raw["gripper_width_m"]
        relative = raw["tcp_leg_relative_position_m"]
        assembly = raw["assembly_rewards"]
    before_h, after_h = height[:, :-1], height[:, 1:]
    before_d, after_d = distance[:, :-1], distance[:, 1:]
    stable = np.linalg.norm(np.diff(relative, axis=1), axis=-1) < .01
    before_success = assembly.cumsum(1)[:, :-1] >= 1
    groups = {
        "recorded_height_drop": (before_h > .06) & (before_h - after_h > .02),
        "held_leg": (before_h > .06) & (before_d < .04) & stable
            & (width[:, :-1] > .015) & (width[:, :-1] < .045),
        "approach": (before_d > .06) & (before_d < .20) & (before_d - after_d > .01),
    }
    # A column represents the command at source index column+1.
    available = np.arange(before_h.shape[1]) + 1 + horizon <= height.shape[1] - 1
    rng = np.random.default_rng(2026110131)
    used, rows = set(), []
    for name, mask in groups.items():
        mask = mask & ~before_success & available[None]
        candidates = [int(index) for index in np.flatnonzero(mask.any(1)) if int(index) not in used]
        if len(candidates) < per_group:
            raise ValueError(f"Not enough disjoint episodes for {name}: {len(candidates)}")
        for index in rng.choice(candidates, per_group, replace=False):
            time = int(rng.choice(np.flatnonzero(mask[index]))) + 1
            rows.append(dict(env_id=int(index), step=time, group=name))
            used.add(int(index))
    return dict(
        archive=str(Path(archive).resolve()), horizon=horizon, environments=len(height),
        rows=rows, seed=2026110132,
        selection="Retrospective diagnostic cohorts selected from recorded geometry; not an unbiased policy evaluation or training labels.",
        intervention="Hold all TCP poses at the measured branch-start pose for one chunk; preserve recorded gripper commands; then replan with the same frozen BC.",
    )


def make_hold(normalizer, observation, actions):
    """Return normalized hold commands and flag targets outside policy bounds."""
    import torch

    device = (normalizer.stats["observations"]["min"].device
              if hasattr(normalizer, "stats") else torch.as_tensor(observation).device)
    state = normalizer(torch.as_tensor(observation, device=device).float(), "observations", forward=False)
    command = normalizer(torch.as_tensor(actions, device=device).float(), "actions", forward=False).clone()
    command[..., :9] = state[..., None, :9]
    normalized = normalizer(command, "actions", forward=True)
    feasible = (normalized.abs() <= 1 + 1e-5).all(dim=(-1, -2))
    return normalized.clamp(-1, 1).cpu().numpy(), feasible.cpu().numpy()


def close_grip(normalizer, actions):
    """Diagnostic intervention: retain pose commands and command closed fingers."""
    import torch

    device = (normalizer.stats["actions"]["min"].device
              if hasattr(normalizer, "stats") else torch.as_tensor(actions).device)
    original = torch.as_tensor(actions, device=device).float()
    command = normalizer(original, "actions", forward=False).clone()
    command[..., 9] = 1.
    closed = normalizer(command, "actions", forward=True)
    # Preserve pose commands bit for bit; only the physical grip field changes.
    result = original.clone()
    result[..., 9] = closed[..., 9]
    return result.cpu().numpy()


def measure(env, initial_leg):
    import torch
    from isaaclab.utils.math import subtract_frame_transforms

    leg = env._parts["square_table_leg4"]
    tcp = env.robot.data.body_pos_w[:, env.franka_ee_index]
    relative, _ = subtract_frame_transforms(
        tcp, env.robot.data.body_quat_w[:, env.franka_ee_index],
        leg.data.root_pos_w, leg.data.root_quat_w,
    )
    return dict(
        height=(leg.data.root_pos_w[:, 2] - initial_leg[:, 2]).cpu().numpy(),
        distance=torch.linalg.vector_norm(tcp - leg.data.root_pos_w, dim=-1).cpu().numpy(),
        width=env.gripper_width().reshape(-1).cpu().numpy(),
        relative=relative.cpu().numpy(),
    )


def main():
    from script.isaaclab.run_lab import prepare_cuda_libraries
    prepare_cuda_libraries()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", choices=["actual", "repeat", "hold", "close_grip"], required=True)
    parser.add_argument("--reference", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing branch probes; select a new output directory")
    if (args.branch != "actual") != bool(args.reference):
        parser.error("Repeat and hold branches require the actual branch archive")
    plan = json.loads(args.plan.read_text())
    from omegaconf import OmegaConf, open_dict
    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    with open_dict(cfg):
        cfg.wandb = None
        cfg.logdir = str(args.output / "runtime")
        cfg.device = cfg.model.device = args.device
        cfg.env.n_envs = plan["environments"]
        cfg.env.specific.headless = args.headless
        cfg.env.specific.gpu_id = int(args.device.split(":")[-1])
    args.output.mkdir(parents=True)
    OmegaConf.save(cfg, args.output / "config.yaml")
    sys.path.insert(0, str(ROOT / "external/furniture-bench"))
    launcher = AppLauncher(args)
    agent = None
    report = dict(branch=args.branch, completed=False, policy_updated=False,
                  reward_updated=False, intervention_chunks=plan["horizon"] if args.branch == "close_grip" else 1,
                  pose_replanning_chunks=plan["horizon"] - 1)
    try:
        import numpy as np
        import torch
        from agent.finetune.train_agent import TrainAgent
        from agent.finetune.lfirl.reward import LFIRLReward, sha256
        from script.isaaclab.diagnose_firl import seed_all
        from script.isaaclab.evaluate_lfirl import (
            physical_state_snapshot, restore_reference, paired_initial_observation,
        )
        seed_all(plan["seed"])
        agent = TrainAgent(cfg)
        weights_path = args.run_dir / "checkpoint/state_0.pt"
        weights = torch.load(weights_path, map_location=args.device, weights_only=True)["model"]
        agent.model.load_state_dict(weights, strict=True)
        agent.model.eval()
        saved = torch.load(args.reward, map_location=args.device, weights_only=True)
        reward = LFIRLReward(saved["learner"]["settings"], 58, 80, args.device)
        reward.load_state_dict(saved["learner"])
        with np.load(plan["archive"], allow_pickle=False) as source:
            initial_states = source["initial_states"].copy()
            initial_physics = source["initial_physics"].copy()
            recorded_actions = source["actions"].copy()
        env = agent.venv.unwrapped
        env.last_grasp.fill_(-1.)
        agent.venv.reset()
        observation = restore_reference(agent, initial_physics)
        stats = agent.venv.normalizer.stats["observations"]
        scale = ((stats["max"] - stats["min"]) / 2).cpu().numpy()
        initial, error = paired_initial_observation(
            observation["state"][:, -1], physical_state_snapshot(env),
            (initial_states, initial_physics), restored=True, observation_scale=scale,
        )
        observation = {"state": initial[:, None]}
        initial_leg = env._parts["square_table_leg4"].data.root_pos_w.clone()
        ids = np.asarray([row["env_id"] for row in plan["rows"]])
        times = np.asarray([row["step"] for row in plan["rows"]])
        horizon, count = plan["horizon"], len(ids)
        reference = np.load(args.reference, allow_pickle=False) if args.reference else None
        fields = dict(
            states=np.zeros((count, horizon + 1, 58), dtype=np.float32),
            actions=np.zeros((count, horizon, 8, 10), dtype=np.float32),
            rewards=np.zeros((count, horizon), dtype=np.float32),
            assembly=np.zeros((count, horizon), dtype=np.float32),
            height=np.zeros((count, horizon + 1), dtype=np.float32),
            distance=np.zeros((count, horizon + 1), dtype=np.float32),
            width=np.zeros((count, horizon + 1), dtype=np.float32),
            relative=np.zeros((count, horizon + 1, 3), dtype=np.float32),
            start_physics=np.zeros((count, initial_physics.shape[1]), dtype=np.float32),
            hold_feasible=np.ones(count, dtype=bool),
            prefix_assembly=np.zeros(count, dtype=np.float32),
            start_q=np.zeros(count, dtype=np.float32),
            end_v=np.zeros(count, dtype=np.float32),
        )
        cumulative = np.zeros(env.num_envs)
        with torch.no_grad():
            for time in range(int((times + horizon).max())):
                if not launcher.app.is_running():
                    raise KeyboardInterrupt("Simulator closed during a branch probe")
                seed_all(plan["seed"] + 100000 + time)
                cond = {"state": torch.as_tensor(observation["state"], device=args.device).float()}
                sampled = agent.model(cond=cond, deterministic=True).trajectories[:, :8].cpu().numpy()
                action = recorded_actions[:, time].copy()
                continuing = (time > times) & (time < times + horizon)
                action[ids[continuing]] = sampled[ids[continuing]]
                starting = np.flatnonzero(times == time)
                if len(starting):
                    physics = physical_state_snapshot(env)
                    fields["start_physics"][starting] = physics[ids[starting]]
                    fields["prefix_assembly"][starting] = cumulative[ids[starting]]
                    if reference is not None:
                        # Share the measured model input only at intervention
                        # time. Physical pre-state agreement is checked later.
                        observation["state"][ids[starting], 0] = reference["states"][starting, 0]
                    held, feasible = make_hold(
                        agent.venv.normalizer, observation["state"][ids[starting], 0], action[ids[starting]],
                    )
                    fields["hold_feasible"][starting] = feasible
                    if args.branch == "hold":
                        action[ids[starting]] = held
                active = np.flatnonzero((time >= times) & (time < times + horizon))
                if args.branch == "close_grip" and len(active):
                    action[ids[active]] = close_grip(agent.venv.normalizer, action[ids[active]])
                offsets = time - times[active]
                before = measure(env, initial_leg)
                fields["states"][active, offsets] = observation["state"][ids[active], 0]
                for key in ("height", "distance", "width", "relative"):
                    fields[key][active, offsets] = before[key][ids[active]]
                fields["actions"][active, offsets] = action[ids[active]]
                if len(active):
                    state = observation["state"][ids[active]]
                    fields["rewards"][active, offsets] = reward.predict(state, action[ids[active]].reshape(-1, 80)).reshape(-1)
                if len(starting):
                    state = torch.as_tensor(observation["state"][ids[starting]], device=args.device).float()
                    command = torch.as_tensor(action[ids[starting]], device=args.device).float().reshape(-1, 80)
                    fields["start_q"][starting] = (reward.q(state, command) + reward.b(state)).reshape(-1).cpu().numpy()
                observation, native_reward, terminated, truncated, _ = agent.venv.step(action)
                if (terminated | truncated).any():
                    raise RuntimeError("Unexpected episode boundary inside the branch window")
                cumulative += native_reward
                fields["assembly"][active, offsets] = native_reward[ids[active]]
                after = measure(env, initial_leg)
                fields["states"][active, offsets + 1] = observation["state"][ids[active], 0]
                for key in ("height", "distance", "width", "relative"):
                    fields[key][active, offsets + 1] = after[key][ids[active]]
                ending = np.flatnonzero(time + 1 == times + horizon)
                if len(ending):
                    state = torch.as_tensor(observation["state"][ids[ending]], device=args.device).float()
                    fields["end_v"][ending] = reward.v(state).reshape(-1).cpu().numpy()
                if time % 10 == 0:
                    print(f"[Branch {args.branch}] common-prefix step={time}, completed={(time + 1 >= times + horizon).sum()}/{count}", flush=True)
        np.savez_compressed(args.output / "branches.npz", **fields)
        if reference is not None:
            reference.close()
        report.update(completed=True, probes=count, environments=env.num_envs,
                      initial_measurement_error=error, gamma=saved["gamma"],
                      plan=str(args.plan.resolve()), reward_sha256=sha256(args.reward),
                      policy_sha256=sha256(weights_path))
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if agent is not None:
            agent.venv.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
