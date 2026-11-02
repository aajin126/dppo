"""Capture real, replay-validated held-leg states for a training reset curriculum.

Recorded commands are replayed without optimization. Proposed states are accepted
only when the live simulation still holds the leg near a socket before assembly.
This changes neither motor commands nor the native assembly criterion.
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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new directory; preserve existing reset banks")

    import numpy as np
    from omegaconf import OmegaConf, open_dict

    plan = json.loads(args.plan.read_text())
    rows = plan["proposed_curriculum_rows"]
    near_completion = plan.get("mode") == "near_completion"
    archives = sorted({row["trajectory"] for row in rows})
    counts = []
    for path in archives:
        with np.load(path, allow_pickle=False) as raw:
            counts.append(len(raw["initial_physics"]))
    if len(set(counts)) != 1:
        raise ValueError("Replay sources must have the same physical scene layout")
    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    with open_dict(cfg):
        cfg.wandb = None
        cfg.logdir = str(args.output / "runtime")
        cfg.device = cfg.model.device = args.device
        cfg.env.n_envs = counts[0]
        cfg.env.specific.headless = args.headless
        cfg.env.specific.gpu_id = int(args.device.split(":")[-1])
    args.output.mkdir(parents=True)
    OmegaConf.save(cfg, args.output / "config.yaml")
    sys.path.insert(0, str(ROOT / "external/furniture-bench"))
    launcher = AppLauncher(args)
    agent = None
    report = dict(completed=False, policy_updated=False, reward_updated=False,
                  purpose="Training reset states only; not a full-task performance evaluation",
                  rows=[], accepted=0)
    try:
        import torch
        import furniture_bench.controllers.control_utils as C
        from agent.finetune.train_agent import TrainAgent
        from agent.finetune.lfirl.reward import sha256
        from agent.finetune.lfirl.reset_bank import relocate_snapshots
        from script.isaaclab.analyze_one_leg_phases import assembly_targets
        from script.isaaclab.check_lfirl_branches import measure
        from script.isaaclab.diagnose_firl import seed_all
        from script.isaaclab.evaluate_lfirl import physical_state_snapshot, restore_reference

        seed_all(2026110302)
        agent = TrainAgent(cfg)
        weights = torch.load(args.run_dir / "checkpoint/state_0.pt", map_location=args.device,
                             weights_only=True)["model"]
        agent.model.load_state_dict(weights, strict=True)
        agent.model.eval()
        env = agent.venv.unwrapped
        targets = torch.as_tensor(assembly_targets(), device=args.device)
        bank_physics, bank_obs, bank_rows = [], [], []
        normalizer = agent.venv.normalizer
        scale = ((normalizer.stats["observations"]["max"]
                  - normalizer.stats["observations"]["min"]) / 2).cpu().numpy()
        for path in archives:
            selected = [row for row in rows if row["trajectory"] == path]
            with np.load(path, allow_pickle=False) as raw:
                physics = raw["initial_physics"].copy()
                commands = raw["actions"].copy()
                original_obs = raw["pre_states"].copy()
            seed_all(selected[0]["seed"])
            agent.venv.reset()
            observation = restore_reference(agent, physics)
            initial_leg = env._parts["square_table_leg4"].data.root_pos_w.clone()
            cumulative = np.zeros(counts[0], dtype=np.float32)
            relative_history = []
            replay_end = max(row["source_success_chunk"] for row in selected) + 1 if near_completion else max(row["step"] for row in selected)
            for time in range(replay_end + 1):
                if not launcher.app.is_running():
                    raise KeyboardInterrupt("Simulator closed during prefix replay")
                geometry = measure(env, initial_leg)
                relative_history.append(geometry["relative"].copy())
                starting = [row for row in selected if row["step"] == time]
                if starting:
                    physical = normalizer(torch.as_tensor(observation["state"][:, -1],
                        device=args.device).float(), "observations", forward=False)
                    top = C.pose_from_vector(physical[:, 16:23])
                    leg = C.pose_from_vector(physical[:, 44:51])
                    relative_pose = torch.linalg.inv(top) @ leg
                    goal_error = (relative_pose[:, None, :3, 3] - targets[None, :, :3, 3]).abs().amax(-1).amin(-1).cpu().numpy()
                    snapshots = physical_state_snapshot(env)
                    already_assembled = np.zeros(env.num_envs, dtype=bool)
                    if near_completion:
                        parts, founds = env._get_parts_poses(sim_coord=True)
                        for row in starting:
                            index = row["env_id"]
                            already_assembled[index] = any(
                                env.furnitures[index].is_assembled_idx(
                                    first, second, parts[index].cpu().numpy(), founds[index].cpu().numpy(),
                                ) for first, second in env.furnitures[index].should_be_assembled
                            )
                    for row in starting:
                        index = row["env_id"]
                        recent = np.stack(relative_history[-4:])[:, index]
                        stable = len(recent) == 4 and bool((np.linalg.norm(np.diff(recent, axis=0), axis=-1) < .01).all())
                        held = (geometry["height"][index] > .06
                                and geometry["distance"][index] < .04
                                and .015 < geometry["width"][index] < .045 and stable)
                        known = bool((np.abs(observation["state"][index, -1, np.r_[16:23, 44:51]]) < 4.999).all())
                        minimum_error = 0. if near_completion else .006
                        # Insertion lowers the leg, so a lifted-height proxy
                        # would incorrectly reject genuine pre-completion states.
                        contact_candidate = near_completion or held
                        accepted = bool(contact_candidate and known and cumulative[index] < 1
                                        and minimum_error < goal_error[index] < .08
                                        and not already_assembled[index])
                        measured = dict(row, accepted=accepted, held=bool(held),
                            previous_native_assembly=float(cumulative[index]),
                            replay_position_error_m=float(goal_error[index]),
                            native_goal_already_satisfied=bool(already_assembled[index]),
                            replay_observation_max_physical_error=float(np.max(np.abs(
                                observation["state"][index, -1] - original_obs[index, time]) * scale)))
                        report["rows"].append(measured)
                        if accepted:
                            bank_physics.append(snapshots[index])
                            bank_obs.append(observation["state"][index, -1].copy())
                            bank_rows.append(measured)
                if time < replay_end:
                    observation, reward, terminated, truncated, _ = agent.venv.step(commands[:, time])
                    cumulative += reward
                    if (terminated | truncated).any():
                        raise RuntimeError("Unexpected reset during physical prefix replay")
                if time % 10 == 0:
                    print(f"[Reset bank] archive={Path(path).name} chunk={time}; accepted={len(bank_physics)}", flush=True)
            if near_completion:
                report.setdefault("source_replays", []).append(dict(
                    trajectory=path,
                    native_successful_environments=int((cumulative >= 1).sum()),
                    checked_source_environments=sorted({int(row["env_id"]) for row in selected}),
                    verified_source_environments=[int(index) for index in
                                                 sorted({row["env_id"] for row in selected}) if cumulative[index] >= 1],
                ))
                for row in bank_rows:
                    if row["trajectory"] == path:
                        verified = bool(cumulative[row["env_id"]] >= 1)
                        row["recorded_continuation_native_success_verified"] = verified
                        row["accepted"] = row["accepted"] and verified
        if near_completion:
            keep = [index for index, row in enumerate(bank_rows) if row["accepted"]]
            bank_physics = [bank_physics[index] for index in keep]
            bank_obs = [bank_obs[index] for index in keep]
        report["accepted"] = len(bank_physics)
        report["mode"] = plan.get("mode", "held_leg")
        if len(bank_physics) < 16:
            raise ValueError(f"Too few replay-validated held states: {len(bank_physics)}; require 16")
        np.savez_compressed(args.output / "states.npz", physical=np.stack(bank_physics),
                            observations=np.stack(bank_obs))
        # Test restoration itself and the opportunity for learning before any
        # curriculum optimizer is allowed to run. These are diagnostic starts,
        # never reported as full-task assembly performance.
        seed_all(2026110304)
        agent.venv.reset()
        donor_ids = np.arange(env.num_envs) % len(bank_physics)
        destinations = env.robot.data.root_pos_w.cpu().numpy()
        restored = relocate_snapshots(np.stack(bank_physics)[donor_ids], destinations,
                                     env.robot.num_joints, 1 + len(env._parts) + len(env._obstacles))
        observation = restore_reference(agent, restored)
        restoration_error = float(np.abs(physical_state_snapshot(env) - restored).max())
        if restoration_error > 1e-5:
            raise ValueError(f"Restored curriculum state differs: {restoration_error}")
        diagnostic_events = np.zeros((env.num_envs, 16), dtype=np.float32)
        with torch.no_grad():
            for time in range(16):
                condition = {"state": torch.as_tensor(observation["state"], device=args.device).float()}
                action = agent.model(cond=condition, deterministic=True).trajectories[:, :agent.act_steps].cpu().numpy()
                observation, reward, _, _, _ = agent.venv.step(action)
                diagnostic_events[:, time] = reward
        np.savez_compressed(args.output / "restoration_diagnostic.npz",
                            donor_ids=donor_ids, native_assembly=diagnostic_events)
        report["restoration_diagnostic"] = dict(
            environments=env.num_envs, macro_chunks=16,
            native_assembly_successes=int((diagnostic_events.sum(1) >= 1).sum()),
            maximum_physical_restoration_error=restoration_error,
            full_task_evaluation=False, policy="Unchanged original BC",
        )
        report.update(completed=True, normalizer_sha256=sha256(cfg.normalization_path),
                      source_bc_sha256=sha256(cfg.base_policy_path),
                      config_sha256=sha256(args.output / "config.yaml"),
                      bank_sha256=sha256(args.output / "states.npz"),
                      root_objects=["robot", *env._parts.keys(), *env._obstacles.keys()],
                      joint_count=env.robot.num_joints)
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
