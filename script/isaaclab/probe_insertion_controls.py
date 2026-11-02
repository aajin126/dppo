"""Isolate insertion failure causes from identical real held-leg snapshots.

Guided branches are diagnostic motor interventions, never policy performance.
No furniture pose is changed after restoring the recorded starting state.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def bounded_rotation_rows(rotation, lower, upper):
    """Encode the same rotation with positive row scales inside action bounds.

    Native row-6D decoding normalizes each row. Clipping unit rows would change
    the requested rotation, whereas positive row scaling preserves it exactly.
    """
    import torch

    rows = rotation[..., :2, :]
    if not ((lower < 0).all() and (upper > 0).all()):
        raise ValueError("Rotation bounds must contain zero in every coordinate")
    bound = torch.where(rows > 0, upper, lower)
    ratio = torch.where(rows.abs() > 1e-8, bound / rows, torch.full_like(rows, float("inf")))
    scale = ratio.amin(-1, keepdim=True).clamp_max(1.) * (1. - 1e-6)
    return (rows * scale).flatten(-2)


def main():
    from script.isaaclab.run_lab import prepare_cuda_libraries

    prepare_cuda_libraries()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mechanics-extensions", action="store_true",
                        help="Also compare bounded overtravel and both screw directions.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous interventions; choose a new output directory")

    import numpy as np
    from omegaconf import OmegaConf, open_dict

    cases = ["bc", "bc_closed", "hold_closed", "descent_only",
             "position_descent", "rotation_descent", "pose_descent"]
    if args.mechanics_extensions:
        cases.extend(["pose_press4", "pose_press8", "pose_spin_plus", "pose_spin_minus"])
    horizon = 32 if args.mechanics_extensions else 16
    count = json.loads((args.bank / "report.json").read_text())["accepted"]
    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    with open_dict(cfg):
        cfg.wandb = None
        cfg.logdir = str(args.output / "runtime")
        cfg.device = cfg.model.device = args.device
        cfg.env.n_envs = count * len(cases)
        cfg.env.specific.headless = args.headless
        cfg.env.specific.gpu_id = int(args.device.split(":")[-1])
    args.output.mkdir(parents=True)
    OmegaConf.save(cfg, args.output / "config.yaml")
    sys.path.insert(0, str(ROOT / "external/furniture-bench"))
    launcher = AppLauncher(args)
    agent = None
    report = dict(completed=False, policy_updated=False, reward_updated=False,
                  training_data_added=False, full_task_evaluation=False,
                  intervention="Motor commands only; unchanged colliders, gains, native assembly tolerance",
                  cases=cases, unique_real_states=count, horizon_chunks=horizon)
    try:
        import torch
        from agent.finetune.train_agent import TrainAgent
        from agent.finetune.lfirl.reset_bank import load_reset_bank, relocate_snapshots
        from agent.finetune.lfirl.reward import sha256
        from isaaclab.utils.math import (
            axis_angle_from_quat, matrix_from_quat, quat_from_angle_axis,
            quat_from_matrix, quat_inv, quat_mul,
        )
        from script.isaaclab.analyze_one_leg_phases import assembly_targets
        from script.isaaclab.diagnose_firl import seed_all
        from script.isaaclab.evaluate_lfirl import physical_state_snapshot, restore_reference

        def pose(position, quaternion):
            result = torch.eye(4, device=args.device).repeat(len(position), 1, 1)
            result[:, :3, :3] = matrix_from_quat(quaternion)
            result[:, :3, 3] = position
            return result

        seed_all(2026110401)
        agent = TrainAgent(cfg)
        weights = args.run_dir / "checkpoint/state_0.pt"
        agent.model.load_state_dict(torch.load(weights, map_location=args.device,
                                              weights_only=True)["model"], strict=True)
        agent.model.eval()
        env = agent.venv.unwrapped
        roots = ["robot", *env._parts.keys(), *env._obstacles.keys()]
        bank, bank_report = load_reset_bank(
            args.bank, sha256(cfg.normalization_path), sha256(cfg.base_policy_path),
            roots, env.robot.num_joints,
        )
        agent.venv.reset()
        ids = np.tile(np.arange(count), len(cases))
        physical = relocate_snapshots(bank[ids], env.robot.data.root_pos_w.cpu().numpy(),
                                      env.robot.num_joints, len(roots))
        observation = restore_reference(agent, physical)
        error = float(np.abs(physical_state_snapshot(env) - physical).max())
        if error > 1e-5:
            raise ValueError(f"Physical restoration differs: {error}")
        top, leg = env._parts["square_table_top"], env._parts["square_table_leg4"]

        def current_poses():
            data = env.robot.data
            return (pose(top.data.root_pos_w, top.data.root_quat_w),
                    pose(leg.data.root_pos_w, leg.data.root_quat_w),
                    pose(data.body_pos_w[:, env.franka_ee_index],
                         data.body_quat_w[:, env.franka_ee_index]),
                    pose(data.body_pos_w[:, env.franka_base_index],
                         data.body_quat_w[:, env.franka_base_index]))

        initial_top, initial_leg, initial_tcp, _ = current_poses()
        initial_relative = torch.linalg.solve(initial_top, initial_leg)
        targets = torch.as_tensor(assembly_targets(), device=args.device)
        nearest = (initial_relative[:, None, :3, 3] - targets[None, :, :3, 3]).square().sum(-1).argmin(-1)
        goals = targets[nearest]
        normalizer = agent.venv.normalizer
        lower = normalizer.stats["actions"]["min"][3:9].reshape(2, 3)
        upper = normalizer.stats["actions"]["max"][3:9].reshape(2, 3)
        initial_states = observation["state"][:, -1].copy()
        actions, events, traces, tracking, clipping = [], [], [], [], []
        with torch.no_grad():
            for chunk in range(horizon):
                if not launcher.app.is_running():
                    raise KeyboardInterrupt("Simulator closed during insertion intervention")
                top_pose, leg_pose, tcp_pose, base_pose = current_poses()
                # Correct observed slip instead of assuming a rigid attachment.
                # These object measurements are for the diagnostic servo only.
                inverse_grasp = torch.linalg.solve(leg_pose, tcp_pose)
                relative = torch.linalg.solve(top_pose, leg_pose)
                traces.append(relative.cpu().numpy())
                physical_action = torch.zeros(env.num_envs, 8, 10, device=args.device)
                for group, name in enumerate(cases):
                    indices = slice(group * count, (group + 1) * count)
                    if name in ("bc", "bc_closed"):
                        seed_all(2026110402 + chunk)
                        condition = {"state": torch.as_tensor(observation["state"][indices],
                                                               device=args.device).float()}
                        sampled = agent.model(cond=condition, deterministic=True).trajectories[:, :8]
                        physical_action[indices] = normalizer(sampled, "actions", forward=False)
                        if name == "bc_closed":
                            physical_action[indices, :, 9] = 1.
                        continue
                    desired = initial_tcp[indices].clone()
                    if name != "hold_closed":
                        desired_leg = initial_relative[indices].clone()
                        # Shared lift/settle schedule isolates lateral centering
                        # and orientation, rather than changing insertion timing.
                        desired_leg[:, 1, 3] = goals[indices, 1, 3] + .05
                        if chunk >= 2:
                            if name == "position_descent" or name.startswith("pose_"):
                                desired_leg[:, [0, 2], 3] = goals[indices][:, [0, 2], 3]
                            if name == "rotation_descent" or name.startswith("pose_"):
                                desired_leg[:, :3, :3] = goals[indices, :3, :3]
                        if chunk >= 6:
                            desired_leg[:, 1, 3] = goals[indices, 1, 3]
                            overtravel = .008 if name == "pose_press8" else (
                                .004 if name in ("pose_press4", "pose_spin_plus", "pose_spin_minus") else 0.
                            )
                            desired_leg[:, 1, 3] -= overtravel
                            if name in ("pose_spin_plus", "pose_spin_minus"):
                                import math

                                turn = min(chunk - 5, 12) * math.pi / 6
                                if name == "pose_spin_minus":
                                    turn = -turn
                                rotation = torch.tensor([
                                    [math.cos(turn), 0., math.sin(turn)],
                                    [0., 1., 0.],
                                    [-math.sin(turn), 0., math.cos(turn)],
                                ], device=args.device)
                                desired_leg[:, :3, :3] = goals[indices, :3, :3] @ rotation
                        desired = top_pose[indices] @ desired_leg @ inverse_grasp[indices]
                    current = tcp_pose[indices]
                    delta = desired[:, :3, 3] - current[:, :3, 3]
                    current_q = quat_from_matrix(current[:, :3, :3])
                    desired_q = quat_from_matrix(desired[:, :3, :3])
                    difference = quat_mul(desired_q, quat_inv(current_q))
                    difference = torch.where(difference[:, :1] < 0, -difference, difference)
                    axis_angle = axis_angle_from_quat(difference)
                    angle = axis_angle.norm(dim=-1)
                    axis = axis_angle / angle[:, None].clamp_min(1e-8)
                    axis[angle < 1e-8] = torch.tensor([1., 0., 0.], device=args.device)
                    for action_index in range(8):
                        fraction = ((action_index + 1) * .002 / delta.norm(dim=-1).clamp_min(1e-8)).clamp_max(1.)
                        next_position = current[:, :3, 3] + fraction[:, None] * delta
                        step_angle = angle.clamp_max((action_index + 1) * np.deg2rad(4.))
                        next_quaternion = quat_mul(quat_from_angle_axis(step_angle, axis), current_q)
                        world_target = pose(next_position, next_quaternion)
                        local_target = torch.linalg.solve(base_pose[indices], world_target)
                        physical_action[indices, action_index, :3] = local_target[:, :3, 3]
                        physical_action[indices, action_index, 3:9] = bounded_rotation_rows(
                            local_target[:, :3, :3], lower, upper,
                        )
                    physical_action[indices, :, 9] = 1.
                normalized = normalizer(physical_action, "actions", forward=True)
                clipping.append((normalized.abs() > 1. + 1e-5).any(-1).cpu().numpy())
                action = normalized.clamp(-1., 1.).cpu().numpy()
                actions.append(action)
                observation, reward, terminated, truncated, _ = agent.venv.step(action)
                events.append(reward)
                if (terminated | truncated).any():
                    raise RuntimeError("Unexpected reset during matched insertion intervention")
                commanded = normalizer(torch.as_tensor(action, device=args.device), "actions", forward=False)
                actual_tcp = env._current_ee_pose_b()[0]
                tracking.append((commanded[:, -1, :3] - actual_tcp).norm(dim=-1).cpu().numpy())
                print(f"[Insertion intervention] chunk={chunk + 1}/{horizon}; events={np.stack(events).sum():.0f}", flush=True)
        event_array = np.stack(events, axis=1)
        trace = np.stack(traces, axis=1)
        results = {}
        for group, name in enumerate(cases):
            indices = slice(group * count, (group + 1) * count)
            results[name] = dict(native_successes=int((event_array[indices].sum(1) >= 1).sum()),
                                 states=count,
                                 median_tcp_tracking_m=float(np.median(np.stack(tracking, axis=1)[indices])),
                                 command_clipping_fraction=float(np.stack(clipping, axis=1)[indices].mean()))
        np.savez_compressed(args.output / "branches.npz", initial_physics=physical,
                            initial_states=initial_states, actions=np.stack(actions, axis=1),
                            native_events=event_array, relative_poses=trace, donor_ids=ids,
                            tcp_tracking_m=np.stack(tracking, axis=1))
        report.update(completed=True, results=results, bank_sha256=bank_report["bank_sha256"],
                      original_bc_sha256=sha256(weights), maximum_restoration_error=error,
                      guided_target_source="Original native socket poses, diagnostic only",
                      guided_object_feedback="Recompute the measured TCP-to-leg relation each chunk; compensate slip",
                      guided_rotation_encoding="Positive row scaling preserves rotation inside existing bounds",
                      maximum_translation_step_m=.002, maximum_rotation_step_degrees=4.)
        print(json.dumps(results, indent=2), flush=True)
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
