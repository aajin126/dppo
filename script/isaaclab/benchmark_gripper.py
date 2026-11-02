"""Measure the native effort-controlled gripper without a learned policy."""

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
    parser.add_argument("--frictions", type=float, nargs="+", default=[15., 0.])
    parser.add_argument("--legacy-joint-friction", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if any(value < 0 for value in args.frictions):
        parser.error("Joint friction must be nonnegative")
    if args.legacy_joint_friction and len(args.frictions) != 1:
        parser.error("Use one coefficient per fresh legacy-model simulator process")
    if args.output.exists():
        parser.error("Use a new output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = compose_config("cfg/furniture/finetune/one_leg_low", "ft_ppo_diffusion_mlp", [
        "wandb=null", "env.n_envs=1", f"logdir={args.output.parent / 'runtime'}",
        *(["+env.specific.legacy_joint_friction=true",
           f"+env.specific.gripper_joint_friction={args.frictions[0]}"]
          if args.legacy_joint_friction else []),
    ])
    cfg.device = cfg.model.device = args.device
    cfg.env.specific.headless = args.headless
    sys.path.insert(0, str(ROOT / "external/furniture-bench"))
    launcher = AppLauncher(args)
    agent = None
    report = {"policy_training": False, "trials": []}
    try:
        import torch
        from isaaclab.utils.math import matrix_from_quat
        from agent.finetune.train_agent import TrainAgent

        agent = TrainAgent(cfg)
        env = agent.venv.unwrapped
        env.reset()
        initial_joints = env.robot.data.joint_pos.clone()
        position, quaternion, _ = env._current_ee_pose_b()
        rotation = matrix_from_quat(quaternion)[:, :2].reshape(1, 6)
        closing = torch.cat((position, rotation, torch.ones(1, 1, device=args.device)), dim=-1)
        for friction in args.frictions:
            env.robot.write_joint_state_to_sim(initial_joints, torch.zeros_like(initial_joints))
            env._position_targets.copy_(initial_joints)
            env.robot.set_joint_position_target(initial_joints)
            if not args.legacy_joint_friction:
                env.robot.write_joint_friction_coefficient_to_sim(friction, joint_ids=env._finger_ids)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(0.)
            env._sync_state()
            widths = [float(env.robot.data.joint_pos[:, env._finger_ids].sum())]
            for _ in range(12):
                env.step(closing)
                widths.append(float(env.robot.data.joint_pos[:, env._finger_ids].sum()))
            result = dict(
                joint_friction=friction, legacy_model=args.legacy_joint_friction,
                time_step_s=.1, width_m=widths,
                legacy_coefficients=env.robot.root_physx_view.get_dof_friction_coefficients()[:, env._finger_ids].cpu().tolist(),
                new_friction_properties=env.robot.root_physx_view.get_dof_friction_properties()[:, env._finger_ids].cpu().tolist(),
            )
            report["trials"].append(result)
            print(f"[Gripper response] {json.dumps(result)}", flush=True)
        report["completed"] = True
    except BaseException as error:
        import traceback
        traceback.print_exc()
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if agent is not None:
            agent.venv.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
