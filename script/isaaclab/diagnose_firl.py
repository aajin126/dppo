"""Audit saved f-IRL rewards with paired rollouts and a frozen-reward DPPO test.

This is an opt-in diagnostic. It does not overwrite the source run or change
the policy, sampler, reward loss, or environment implementation.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from script.isaaclab.run_lab import (
    compose_config,
    enable_progress_logging,
    prepare_cuda_libraries,
)


def describe(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p05": float(np.quantile(values, .05)),
        "p95": float(np.quantile(values, .95)),
    }


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def offline_audit(agent, source, output):
    import numpy as np
    import torch

    initial = torch.load(source / "checkpoint/state_0.pt", map_location="cpu", weights_only=True)["model"]
    final = torch.load(source / "checkpoint/state_99.pt", map_location="cpu", weights_only=True)["model"]
    saved_reward = torch.load(source / "checkpoint/firl_99.pt", map_location="cpu", weights_only=True)
    bc = torch.load(saved_reward["source_bc_checkpoint"], map_location="cpu", weights_only=True)["ema"]
    identical = all(
        torch.equal(value, bc["network." + name[len("actor_ft."):]])
        for name, value in initial.items() if name.startswith("actor_ft.")
    )
    changes = {
        prefix: max(
            float((final[name] - value).abs().max())
            for name, value in initial.items() if name.startswith(prefix + ".")
        ) for prefix in ("actor", "actor_ft", "critic")
    }
    expert = agent.expert.states
    rewards = agent.reward_learner.predict(expert)
    logits = agent.reward_learner.disc.log_density_ratio(expert).cpu().numpy()
    np.savez_compressed(output / "expert_scores.npz", reward=rewards, discriminator_logit=logits)
    return {
        "initial_actor_identical_to_source_bc_ema": identical,
        "maximum_parameter_change": changes,
        "expert_evaluation_scope": "training data only; no held-out expert demonstrations were available",
        "expert_states": len(expert),
        "expert_reward": describe(rewards),
        "expert_discriminator_logit": describe(logits),
        "expert_classification_accuracy": float((logits > 0).mean()),
        "expert_reward_clipped_fraction": float((np.abs(rewards) >= 9.999).mean()),
        "reward_checkpoint": str(source / "checkpoint/firl_99.pt"),
    }


def rollout(agent, app, output, label, seed, stochastic=False):
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    device_index = torch.device(agent.device).index or 0
    try:
        with torch.random.fork_rng(devices=[device_index]), torch.no_grad():
            seed_all(seed)
            obs = agent.venv.reset()
            initial = obs["state"][:, -1].copy()
            env = agent.venv.unwrapped
            leg = env._parts["square_table_leg4"]
            initial_leg = leg.data.root_pos_w.detach().clone()
            seed_all(seed + 100000)
            agent.model.eval()
            states, actions, assembly = [], [], []
            heights, distances = [], []
            for step in range(agent.n_steps):
                if not app.is_running():
                    raise KeyboardInterrupt("Simulation closed during diagnostic")
                cond = {"state": torch.as_tensor(obs["state"], device=agent.device, dtype=torch.float32)}
                samples = agent.model(cond=cond, deterministic=not stochastic)
                action = samples.trajectories[:, :agent.act_steps].cpu().numpy()
                obs, reward, terminated, truncated, _ = agent.venv.step(action)
                states.append(obs["state"][:, -1].copy())
                actions.append(action)
                assembly.append(reward.copy())
                leg_position = leg.data.root_pos_w
                tcp = env.robot.data.body_pos_w[:, env.franka_ee_index]
                heights.append((leg_position[:, 2] - initial_leg[:, 2]).cpu().numpy())
                distances.append(torch.linalg.vector_norm(tcp - leg_position, dim=-1).cpu().numpy())
                if step % 20 == 0:
                    print(f"[Diagnostic] {label} seed={seed} chunk={step}/{agent.n_steps}", flush=True)
                if step < agent.n_steps - 1 and (terminated | truncated).any():
                    raise RuntimeError("Unexpected internal episode termination")
            states = np.stack(states).transpose(1, 0, 2)
            actions = np.stack(actions).transpose(1, 0, 2, 3)
            assembly = np.stack(assembly).T
            heights = np.stack(heights).T
            distances = np.stack(distances).T
            flat = states.reshape(-1, agent.obs_dim)
            learned = agent.reward_learner.predict(flat).reshape(states.shape[:2])
            logits = agent.reward_learner.disc.log_density_ratio(flat).cpu().numpy().reshape(states.shape[:2])
            max_runs = np.zeros(agent.n_envs, dtype=int)
            runs = np.zeros_like(max_runs)
            for above in (heights > .03).T:
                runs = np.where(above, runs + 1, 0)
                max_runs = np.maximum(max_runs, runs)
            path = output / f"{label}_seed_{seed}.npz"
            np.savez_compressed(
                path, initial_states=initial, states=states, actions=actions,
                assembly_rewards=assembly, frozen_rewards=learned,
                discriminator_logits=logits, leg_height_gain_m=heights,
                tcp_leg_distance_m=distances,
            )
            result = {
                "label": label,
                "seed": seed,
                "sampler_mode": "native_training_stochastic" if stochastic else "native_evaluation",
                "episodes": agent.n_envs,
                "assembly_returns": assembly.sum(1).tolist(),
                "assembly_successes": (assembly.sum(1) >= agent.best_reward_threshold_for_success).tolist(),
                "frozen_reward_returns": learned.sum(1).tolist(),
                "max_leg_height_gain_m": heights.max(1).tolist(),
                "leg_lift_3cm_sustained_3_chunks": (max_runs >= 3).tolist(),
                "minimum_tcp_leg_distance_m": distances.min(1).tolist(),
                "policy_discriminator_logit": describe(logits),
                "policy_classification_accuracy": float((logits < 0).mean()),
                "trajectory_path": str(path),
            }
            print("[Diagnostic result] " + json.dumps(result), flush=True)
            return result, initial
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def main():
    prepare_cuda_libraries()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--frozen-iterations", type=int, default=10)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory to preserve prior diagnostic artifacts")
    args.output.mkdir(parents=True)
    source = args.run_dir.resolve()
    reward_path = source / "checkpoint/firl_99.pt"
    cfg = compose_config("cfg/furniture/finetune/one_leg_low", "ft_firl_diffusion_mlp", [
        f"env.n_envs={args.num_envs}",
        f"train.batch_size={args.num_envs * 64}",
        f"train.n_train_itr={args.frozen_iterations}",
        "train.save_model_freq=10", "+train.force_train=true",
        "firl.freeze_reward=true", f"firl.reward_checkpoint={reward_path}",
        f"logdir={args.output.resolve() / 'frozen_training'}",
    ])
    from omegaconf import OmegaConf, open_dict

    with open_dict(cfg):
        cfg.device = args.device
        cfg.model.device = args.device
        cfg.env.specific.headless = args.headless
        cfg.env.specific.gpu_id = int(args.device.split(":")[-1])
    OmegaConf.save(cfg, args.output / "diagnostic_config.yaml")
    sys.path.insert(0, str(ROOT / "external/furniture-bench"))
    launcher = AppLauncher(args)
    agent = None
    report = {"source_run": str(source), "rollouts": [], "initial_state_comparisons": []}
    report_path = args.output / "report.json"
    try:
        import hydra
        import numpy as np
        import torch

        agent = hydra.utils.get_class(cfg._target_)(cfg)
        enable_progress_logging()
        report["offline"] = offline_audit(agent, source, args.output)
        save_report(report_path, report)
        references = {}
        for iteration in (0, 60, 99):
            state = torch.load(source / f"checkpoint/state_{iteration}.pt", map_location=agent.device, weights_only=True)
            agent.model.load_state_dict(state["model"], strict=True)
            for seed in args.seeds:
                result, initial = rollout(agent, launcher.app, args.output, f"state_{iteration}", seed)
                report["rollouts"].append(result)
                if iteration == 0:
                    references[seed] = initial
                else:
                    report["initial_state_comparisons"].append({
                        "policy": iteration, "seed": seed,
                        "max_abs_normalized_difference": float(np.abs(initial - references[seed]).max()),
                    })
                save_report(report_path, report)
            result, _ = rollout(agent, launcher.app, args.output, f"state_{iteration}_stochastic", args.seeds[0], True)
            report["rollouts"].append(result)
            save_report(report_path, report)
        state = torch.load(source / "checkpoint/state_0.pt", map_location=agent.device, weights_only=True)
        agent.model.load_state_dict(state["model"], strict=True)
        before = {k: v.detach().clone() for k, v in agent.reward_learner.reward.state_dict().items()}
        seed_all(314159)
        if args.frozen_iterations:
            agent.run()
            report["frozen_reward_unchanged"] = all(
                torch.equal(value, agent.reward_learner.reward.state_dict()[name])
                for name, value in before.items()
            )
            assert report["frozen_reward_unchanged"]
            for seed in args.seeds:
                result, initial = rollout(agent, launcher.app, args.output, "frozen_trained", seed)
                report["rollouts"].append(result)
                report["initial_state_comparisons"].append({
                    "policy": "frozen_trained", "seed": seed,
                    "max_abs_normalized_difference": float(np.abs(initial - references[seed]).max()),
                })
                save_report(report_path, report)
        report["completed"] = True
        save_report(report_path, report)
    finally:
        if agent is not None:
            agent.venv.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
