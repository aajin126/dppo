"""Measure native DiffIK/PD tracking without training or changing checkpoints."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from script.isaaclab.run_lab import compose_config, prepare_cuda_libraries


def main():
    prepare_cuda_libraries()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--gains", nargs="+", default=["1000,200", "1000,100", "2000,200", "2000,100", "4000,200"])
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory")
    gains = [tuple(map(float, value.split(","))) for value in args.gains]
    if any(len(pair) != 2 or pair[0] <= 0 or pair[1] < 0 for pair in gains):
        parser.error("Expected positive stiffness and nonnegative damping")
    output.mkdir(parents=True)
    cfg = compose_config("cfg/furniture/finetune/one_leg_low", "ft_ppo_diffusion_mlp", [
        "wandb=null", f"env.n_envs={args.num_envs}", f"logdir={output / 'runtime'}",
    ])
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.device = args.device
        cfg.model.device = args.device
        cfg.env.specific.headless = args.headless
        cfg.env.specific.gpu_id = int(args.device.split(":")[-1])
    sys.path.insert(0, str(ROOT / "external/furniture-bench"))
    launcher = AppLauncher(args)
    agent = None
    report = dict(policy_training=False, conditions=[], trials=[])
    try:
        import numpy as np
        import torch
        from isaaclab.utils.math import matrix_from_quat, quat_from_angle_axis, quat_mul, quat_error_magnitude
        from agent.finetune.train_agent import TrainAgent

        agent = TrainAgent(cfg)
        env = agent.venv.unwrapped
        torch.manual_seed(4242)
        env.reset()
        initial_joints = env.robot.data.joint_pos.clone()
        period = env._control_substeps * env._physics_dt
        if abs(period - .1) > 1e-9:
            raise RuntimeError(f"Unexpected control period: {period}")
        report["control_period_s"] = period
        reference_position = None
        reference_quaternion = None

        def restore_robot():
            env.robot.write_joint_state_to_sim(initial_joints, torch.zeros_like(initial_joints))
            env._position_targets.copy_(initial_joints)
            env._efforts.zero_()
            env._efforts[:, env._finger_ids] = float(env.sim_config["robot"]["gripper_torque"])
            env.last_grasp.fill_(-1.)
            env.robot.set_joint_position_target(env._position_targets)
            env.robot.set_joint_effort_target(env._efforts)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(0.)
            env._sync_state()

        for stiffness, damping in gains:
            env.robot.write_joint_stiffness_to_sim(stiffness, joint_ids=env._arm_ids)
            env.robot.write_joint_damping_to_sim(damping, joint_ids=env._arm_ids)
            env.robot.actuators["arm"].stiffness.fill_(stiffness)
            env.robot.actuators["arm"].damping.fill_(damping)
            measurements = []
            for kind in ("translation", "rotation"):
                for axis in range(3):
                    for sign in (-1, 1):
                        restore_robot()
                        position, quaternion, _ = env._current_ee_pose_b()
                        if reference_position is None:
                            reference_position, reference_quaternion = position.clone(), quaternion.clone()
                        if not torch.allclose(position, reference_position, atol=1e-5, rtol=0):
                            raise RuntimeError("Calibration robot initial position changed")
                        target_position, target_quaternion = position.clone(), quaternion.clone()
                        if kind == "translation":
                            target_position[:, axis] += sign * .015
                        else:
                            direction = torch.zeros(args.num_envs, 3, device=args.device)
                            direction[:, axis] = 1.
                            angle = torch.full((args.num_envs,), sign * np.deg2rad(5), device=args.device)
                            target_quaternion = quat_mul(quat_from_angle_axis(angle, direction), quaternion)
                        rotation = matrix_from_quat(target_quaternion)[:, :2].reshape(args.num_envs, 6)
                        action = torch.cat((target_position, rotation, -torch.ones(args.num_envs, 1, device=args.device)), dim=-1)
                        for tick in range(2):
                            env.step(action)
                            actual_position, actual_quaternion, _ = env._current_ee_pose_b()
                            position_error = (target_position - actual_position).norm(dim=-1) * 1000
                            rotation_error = quat_error_magnitude(target_quaternion, actual_quaternion).rad2deg()
                            for index in range(args.num_envs):
                                measurements.append(dict(
                                    kind=kind, axis=axis, sign=sign, tick=tick + 1,
                                    position_error_mm=float(position_error[index]),
                                    rotation_error_deg=float(rotation_error[index]),
                                ))
            first_translation = [row["position_error_mm"] for row in measurements if row["kind"] == "translation" and row["tick"] == 1]
            first_rotation = [row["rotation_error_deg"] for row in measurements if row["kind"] == "rotation" and row["tick"] == 1]
            summary = dict(
                stiffness=stiffness, damping=damping,
                translation_error_mm_median_100ms=float(np.median(first_translation)),
                translation_error_mm_p95_100ms=float(np.quantile(first_translation, .95)),
                rotation_error_deg_median_100ms=float(np.median(first_rotation)),
                rotation_error_deg_p95_100ms=float(np.quantile(first_rotation, .95)),
            )
            report["conditions"].append(summary)
            report["trials"].append(dict(stiffness=stiffness, damping=damping, measurements=measurements))
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"[Tracking calibration] {json.dumps(summary)}", flush=True)
        report["completed"] = True
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as error:
        import traceback
        traceback.print_exc()
        report["error"] = repr(error)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        sys.stderr.flush()
        raise
    finally:
        if agent is not None:
            agent.venv.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
