"""Compare the active IsaacLab TCP and Jacobian against the original URDF."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from script.isaaclab.run_lab import prepare_cuda_libraries

    prepare_cuda_libraries()
    from isaaclab.app import AppLauncher
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous checks; use a new output directory")
    args.output.mkdir(parents=True)
    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    cfg.wandb = None
    cfg.device = cfg.model.device = args.device
    cfg.env.n_envs = 1
    cfg.env.specific.headless = args.headless
    cfg.logdir = str(args.output / "runtime")
    sys.path.insert(0, str(ROOT / "external/furniture-bench"))
    launcher = AppLauncher(args)
    agent = None
    report = dict(completed=False, policy_updated=False, physics_parameters_changed=False)
    try:
        import numpy as np
        import pinocchio as pin
        import torch
        from agent.finetune.train_agent import TrainAgent
        from isaaclab.utils.math import matrix_from_quat

        agent = TrainAgent(cfg)
        agent.venv.reset()
        env = agent.venv.unwrapped
        urdf = ROOT / "external/furniture-bench/furniture_bench/assets/franka_description_ros/franka_description/robots/franka_panda.urdf"
        oracle = pin.buildModelFromUrdf(str(urdf))
        frame = oracle.getFrameId("k_ee_link")
        arm_names = [env.robot.joint_names[index] for index in env._arm_ids]
        joint_columns = [oracle.joints[oracle.getJointId(name)].idx_v for name in arm_names]
        original = env.robot.data.joint_pos.clone()
        rows = []

        def pose(q):
            data = oracle.createData()
            pin.framesForwardKinematics(oracle, data, q)
            return data.oMf[frame].translation.copy(), data.oMf[frame].rotation.copy()

        for sample in range(5):
            target = original.clone()
            if sample:
                perturbation = .12 * torch.sin(torch.arange(7, device=env.device) + sample)
                target[:, env._arm_ids] += perturbation
            env.robot.write_joint_state_to_sim(target, torch.zeros_like(target))
            env.robot.set_joint_position_target(target)
            env.robot.set_joint_effort_target(torch.zeros_like(target))
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env._physics_dt)
            actual = env.robot.data.joint_pos[0].cpu().numpy()
            q = pin.neutral(oracle)
            for index, name in enumerate(env.robot.joint_names):
                q[oracle.joints[oracle.getJointId(name)].idx_q] = actual[index]
            position, rotation = pose(q)
            native_position, native_quaternion, base = env._current_ee_pose_b()
            native_rotation = matrix_from_quat(native_quaternion)[0].cpu().numpy()
            native = env.robot.root_physx_view.get_jacobians()[:, env._jacobian_id, :, env._arm_ids][0].clone()
            inverse_base = matrix_from_quat(base)[0].T
            native[:3] = inverse_base @ native[:3]
            native[3:] = inverse_base @ native[3:]
            native = native.cpu().numpy()
            analytic = pin.computeFrameJacobian(
                oracle, oracle.createData(), q, frame, pin.LOCAL_WORLD_ALIGNED,
            )[:, joint_columns]
            numerical = np.empty_like(analytic)
            for column, joint in enumerate(joint_columns):
                plus, minus = q.copy(), q.copy()
                plus[joint] += 1e-6
                minus[joint] -= 1e-6
                p1, r1 = pose(plus)
                p0, r0 = pose(minus)
                numerical[:3, column] = (p1 - p0) / 2e-6
                numerical[3:, column] = pin.log3(r1 @ r0.T) / 2e-6
            rows.append(dict(
                sample=sample,
                tcp_position_error_m=float(np.linalg.norm(native_position[0].cpu().numpy() - position)),
                tcp_orientation_error_rad=float(np.linalg.norm(pin.log3(native_rotation @ rotation.T))),
                jacobian_linear_max_error=float(np.abs(native[:3] - analytic[:3]).max()),
                jacobian_angular_max_error=float(np.abs(native[3:] - analytic[3:]).max()),
                pinocchio_vs_central_difference_max_error=float(np.abs(numerical - analytic).max()),
            ))
        report.update(completed=True, urdf=str(urdf), tcp="k_ee_link", configurations=rows)
        report["passed"] = all(
            row["tcp_position_error_m"] < 1e-4 and row["tcp_orientation_error_rad"] < 1e-3
            and row["jacobian_linear_max_error"] < 1e-3 and row["jacobian_angular_max_error"] < 1e-3
            and row["pinocchio_vs_central_difference_max_error"] < 1e-6 for row in rows
        )
        report["caveat"] = "Kinematics only; this does not establish original-engine contact or controller-dynamics equivalence. Joint states are changed in this isolated diagnostic scene."
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if agent is not None:
            agent.venv.close()
        launcher.app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
