"""Check whether original assembly poses are physically feasible with SDF 512.

This is a mechanics intervention, not a policy rollout or a success benchmark.
Place the leg at/above original socket poses, hold the robot away, then observe
gravity/contact response. Never add these initialized states to training data.
"""

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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    from omegaconf import OmegaConf, open_dict

    offsets = [(0., 0.), (.025, 0.), (.035, 0.), (0., .006), (.025, .006)]
    cases = [dict(socket=socket, extra_height_m=height, lateral_offset_m=lateral)
             for socket in range(4) for height, lateral in offsets]
    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    with open_dict(cfg):
        cfg.wandb = None
        cfg.logdir = str(args.output / "runtime")
        cfg.device = cfg.model.device = args.device
        cfg.env.n_envs = len(cases)
        cfg.env.specific.headless = args.headless
        cfg.env.specific.gpu_id = int(args.device.split(":")[-1])
    args.output.mkdir(parents=True)
    OmegaConf.save(cfg, args.output / "config.yaml")
    sys.path.insert(0, str(ROOT / "external/furniture-bench"))
    launcher = AppLauncher(args)
    agent = None
    report = dict(completed=False, mechanics_intervention=True,
                  policy_performance_evaluation=False, policy_updated=False,
                  training_data_added=False, cases=cases)
    try:
        import numpy as np
        import torch
        from agent.finetune.train_agent import TrainAgent
        from isaaclab.utils.math import quat_apply, subtract_frame_transforms
        from script.isaaclab.analyze_one_leg_phases import assembly_targets
        from script.isaaclab.check_lfirl_branches import make_hold
        from script.isaaclab.diagnose_firl import seed_all
        from script.isaaclab.evaluate_lfirl import physical_state_snapshot, restore_reference

        seed_all(2026110339)
        agent = TrainAgent(cfg)
        observation = agent.venv.reset()
        env = agent.venv.unwrapped
        top = env._parts["square_table_top"]
        leg = env._parts["square_table_leg4"]
        targets = torch.as_tensor(assembly_targets(), device=args.device)
        torch.testing.assert_close(targets[:, :3, :3], torch.eye(3, device=args.device).expand(4, -1, -1))
        goal = targets[[row["socket"] for row in cases], :3, 3].clone()
        starting = goal.clone()
        starting[:, 1] += torch.tensor([row["extra_height_m"] for row in cases], device=args.device)
        starting[:, 0] += torch.tensor([row["lateral_offset_m"] for row in cases], device=args.device)
        world_position = top.data.root_pos_w + quat_apply(top.data.root_quat_w, starting)
        physical = physical_state_snapshot(env)
        roots = ["robot", *env._parts.keys(), *env._obstacles.keys()]
        offset = 2 * env.robot.num_joints + 13 * roots.index("square_table_leg4")
        physical[:, offset:offset + 3] = world_position.cpu().numpy()
        physical[:, offset + 3:offset + 7] = top.data.root_quat_w.cpu().numpy()
        physical[:, offset + 7:offset + 13] = 0.
        observation = restore_reference(agent, physical)
        commands = np.zeros((env.num_envs, agent.act_steps, agent.action_dim), dtype=np.float32)
        held, feasible = make_hold(agent.venv.normalizer, observation["state"][:, -1], commands)
        if not feasible.all():
            raise ValueError("Initial robot hold pose is outside original action bounds")
        records = []
        for chunk in range(4):
            relative, _ = subtract_frame_transforms(
                top.data.root_pos_w, top.data.root_quat_w,
                leg.data.root_pos_w, leg.data.root_quat_w,
            )
            records.append(relative.cpu().numpy().copy())
            if chunk < 3:
                observation, _, terminated, truncated, _ = agent.venv.step(held)
                if (terminated | truncated).any():
                    raise RuntimeError("Unexpected reset in the mechanics probe")
        trace = np.stack(records, axis=1)
        np.savez_compressed(args.output / "mechanics.npz", relative_positions=trace,
                            native_goal_positions=goal.cpu().numpy())
        for index, row in enumerate(cases):
            row.update(relative_height_trace_m=trace[index, :, 1].tolist(),
                       final_relative_position_m=trace[index, -1].tolist(),
                       final_native_goal_position_error_m=(trace[index, -1] - goal[index].cpu().numpy()).tolist())
        report.update(completed=True, measured_times_s=[0., .8, 1.6, 2.4],
                      collider_approximation="SDF", sdf_resolution=512,
                      native_assembly_tolerance_changed=False)
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
