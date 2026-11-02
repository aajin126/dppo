"""Evaluate a frozen policy from real curriculum states, never full-task starts."""

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
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026110305)
    parser.add_argument("--training-sampler", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists() or args.horizon < 1:
        parser.error("Use a new output directory and a positive horizon")

    from omegaconf import OmegaConf, open_dict

    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    with open_dict(cfg):
        cfg.wandb = None
        cfg.logdir = str(args.output / "runtime")
        cfg.device = cfg.model.device = args.device
        cfg.seed = args.seed
        cfg.env.n_envs = args.num_envs
        cfg.env.max_episode_steps = args.horizon * cfg.act_steps
        cfg.env.specific.headless = args.headless
        cfg.env.specific.gpu_id = int(args.device.split(":")[-1])
    args.output.mkdir(parents=True)
    OmegaConf.save(cfg, args.output / "config.yaml")
    sys.path.insert(0, str(ROOT / "external/furniture-bench"))
    launcher = AppLauncher(args)
    agent = None
    report = dict(completed=False, full_task_evaluation=False, policy_updated=False,
                  reward_updated=False, horizon_macro_chunks=args.horizon, seed=args.seed,
                  policy_noise_seed=args.seed + 100000,
                  native_sampler_mode="training" if args.training_sampler else "evaluation")
    try:
        import numpy as np
        import torch
        from agent.finetune.train_agent import TrainAgent
        from agent.finetune.lfirl.reward import sha256
        from agent.finetune.lfirl.reset_bank import load_reset_bank, relocate_snapshots
        from script.isaaclab.diagnose_firl import seed_all
        from script.isaaclab.evaluate_lfirl import restore_reference, physical_state_snapshot

        seed_all(args.seed)
        agent = TrainAgent(cfg)
        checkpoint = args.checkpoint or args.run_dir / "checkpoint/state_0.pt"
        weights = torch.load(checkpoint, map_location=args.device, weights_only=True)["model"]
        agent.model.load_state_dict(weights, strict=True)
        agent.model.eval()
        env = agent.venv.unwrapped
        roots = ["robot", *env._parts.keys(), *env._obstacles.keys()]
        bank, bank_report = load_reset_bank(args.bank, sha256(cfg.normalization_path),
                                            sha256(cfg.base_policy_path), roots, env.robot.num_joints)
        agent.venv.reset()
        ids = np.arange(env.num_envs) % len(bank)
        restored = relocate_snapshots(bank[ids], env.robot.data.root_pos_w.cpu().numpy(),
                                     env.robot.num_joints, len(roots))
        observation = restore_reference(agent, restored)
        initial_physics = physical_state_snapshot(env)
        restoration_error = float(np.abs(initial_physics - restored).max())
        if restoration_error > 1e-5:
            raise ValueError(f"Physical restoration error: {restoration_error}")
        initial_states = observation["state"][:, -1].copy()
        # Separate policy noise from constructor and reset RNG consumption.
        seed_all(args.seed + 100000)
        states = np.empty((env.num_envs, args.horizon, agent.obs_dim), dtype=np.float32)
        actions = np.empty((env.num_envs, args.horizon, agent.act_steps, agent.action_dim), dtype=np.float32)
        events = np.zeros((env.num_envs, args.horizon), dtype=np.float32)
        with torch.no_grad():
            for time in range(args.horizon):
                if not launcher.app.is_running():
                    raise KeyboardInterrupt("Simulator closed during reset-bank diagnostic")
                states[:, time] = observation["state"][:, -1]
                condition = {"state": torch.as_tensor(observation["state"], device=args.device).float()}
                action = agent.model(cond=condition, deterministic=not args.training_sampler).trajectories[:, :agent.act_steps].cpu().numpy()
                actions[:, time] = action
                observation, reward, terminated, _, _ = agent.venv.step(action)
                events[:, time] = reward
                if terminated.any() and time != args.horizon - 1:
                    raise RuntimeError("Unexpected reset inside the diagnostic horizon")
                if time % 10 == 0:
                    print(f"[Bank diagnostic] chunk={time}/{args.horizon}; native successes={(events.sum(1) >= 1).sum()}", flush=True)
        np.savez_compressed(args.output / "rollout.npz", donor_ids=ids, pre_states=states,
                            actions=actions, assembly_rewards=events,
                            initial_states=initial_states, initial_physics=initial_physics)
        report.update(completed=True, native_assembly_successes=int((events.sum(1) >= 1).sum()),
                      environments=env.num_envs, bank_sha256=bank_report["bank_sha256"],
                      checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                      maximum_physical_restoration_error=restoration_error)
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
