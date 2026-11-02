"""Run official LFIRL stages on local Furniture NPZ and native diffusion BC."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.isaaclab.run_lab import compose_config, prepare_cuda_libraries


def main():
    prepare_cuda_libraries()
    import hydra
    import numpy as np
    import torch
    from agent.finetune.lfirl.dataset import MacroTransitions, ReferenceRewardTransitions
    from agent.finetune.lfirl.policy import MacroDiffusionPolicy, NativeActionPolicy
    from agent.finetune.lfirl.reward import LFIRLReward, default_settings, sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["one_leg", "lamp", "round_table"], default="one_leg")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches-per-stage", type=int, default=0)
    parser.add_argument("--max-expert-episodes", type=int, default=0)
    parser.add_argument("--qhat-passes", type=int)
    parser.add_argument("--q-lr", type=float)
    parser.add_argument("--q-score-noise-samples", type=int, default=1)
    parser.add_argument("--resume-reward", type=Path)
    parser.add_argument("--reset-reward-head", action="store_true")
    parser.add_argument("--reset-calibration-heads", action="store_true")
    parser.add_argument("--calibration-lr", type=float)
    parser.add_argument("--v-passes", type=int)
    parser.add_argument("--v-loss", choices=["gumbel", "expectile"])
    parser.add_argument("--v-random-action-samples", type=int)
    parser.add_argument("--b-passes", type=int)
    parser.add_argument("--r-passes", type=int)
    parser.add_argument("--r-lr", type=float)
    parser.add_argument("--r-loss", choices=["huber", "mse"])
    parser.add_argument("--r-include-action", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--shuffle-r", action="store_true")
    parser.add_argument("--shuffle-calibration", action="store_true")
    parser.add_argument("--reference-rollouts", type=Path)
    parser.add_argument("--reference-training-fraction", type=float, default=.75)
    parser.add_argument("--reference-assembly-terminal", action="store_true")
    parser.add_argument("--reference-success-threshold", type=float)
    parser.add_argument("--discount", type=float)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--window-stride", type=int)
    parser.add_argument("--resume-stage", choices=["QHAT", "V_INIT", "B", "V_CALIBRATED", "R"], default="V_INIT")
    parser.add_argument("--reward-horizon", type=int)
    parser.add_argument(
        "--reward-observation-normalization",
        choices=["expert_standard", "policy"],
        help="Use expert standardization, or retain the existing DPPO NPZ normalization.",
    )
    parser.add_argument(
        "--terminal-mode", choices=["truncate", "demonstration_end"], default="truncate"
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_batches_per_stage < 0 or args.max_expert_episodes < 0:
        parser.error("Batch size must be positive and limits must be nonnegative")
    if args.qhat_passes is not None and args.qhat_passes < 1:
        parser.error("--qhat-passes must be positive")
    if args.q_lr is not None and (not np.isfinite(args.q_lr) or args.q_lr <= 0):
        parser.error("--q-lr must be finite and positive")
    if args.q_score_noise_samples < 1:
        parser.error("--q-score-noise-samples must be positive")
    if args.v_random_action_samples is not None and args.v_random_action_samples < 0:
        parser.error("--v-random-action-samples must be nonnegative")
    if args.reset_calibration_heads and (not args.resume_reward or args.resume_stage != "V_INIT"):
        parser.error("--reset-calibration-heads requires --resume-reward and --resume-stage V_INIT")
    if args.reference_rollouts and (not args.resume_reward or args.resume_stage == "QHAT"):
        parser.error("Reference transitions require a resumed, frozen Q; Q imitation remains expert-only")
    if args.reference_assembly_terminal and (
            not args.reference_rollouts or args.reference_success_threshold is None
            or not np.isfinite(args.reference_success_threshold) or args.reference_success_threshold <= 0):
        parser.error("Native reference terminals require --reference-rollouts and an explicit positive --reference-success-threshold")
    if args.reference_success_threshold is not None and not args.reference_assembly_terminal:
        parser.error("--reference-success-threshold requires --reference-assembly-terminal")
    if args.calibration_lr is not None and (not np.isfinite(args.calibration_lr) or args.calibration_lr <= 0):
        parser.error("--calibration-lr must be finite and positive")
    if args.r_lr is not None and (not np.isfinite(args.r_lr) or args.r_lr <= 0):
        parser.error("--r-lr must be finite and positive")
    if args.discount is not None and not 0 < args.discount < 1:
        parser.error("--discount must be between zero and one")
    if args.save_every < 0:
        parser.error("--save-every must be nonnegative")
    if args.window_stride is not None and args.window_stride < 1:
        parser.error("--window-stride must be positive")
    for name in ("v_passes", "b_passes", "r_passes"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("data", "checkpoint", "output", "resume_reward", "reference_rollouts"):
        path = getattr(args, name)
        if path is not None:
            path = path.expanduser()
            setattr(args, name, path if path.is_absolute() else ROOT / path)
    np.random.seed(42)
    torch.manual_seed(42)
    if (args.output / "reward.pt").exists():
        parser.error("Use a new output directory to preserve existing reward checkpoints")
    cfg = compose_config(
        f"cfg/furniture/finetune/{args.task}_low", "ft_ppo_diffusion_mlp", []
    )
    cfg.model.device = args.device
    if args.discount is not None:
        cfg.train.gamma = args.discount
    if args.checkpoint:
        cfg.model.network_path = str(args.checkpoint.resolve())
    reward_horizon = cfg.act_steps if args.reward_horizon is None else args.reward_horizon
    if reward_horizon not in (1, cfg.act_steps):
        parser.error("--reward-horizon must be 1 or match the native action chunk")
    data_path = (args.data or ROOT / f"data/furniture/{args.task}_low/train.npz").resolve()
    normalization = data_path.with_name("normalization.pth")
    if sha256(normalization) != sha256(cfg.normalization_path):
        raise ValueError("Expert NPZ and native policy normalization must match")
    transitions = MacroTransitions(
        data_path, reward_horizon, args.terminal_mode, args.max_expert_episodes, args.window_stride
    )
    if args.reference_rollouts:
        transitions = ReferenceRewardTransitions(
            transitions, args.reference_rollouts, sha256(normalization),
            sha256(cfg.model.network_path), args.reference_training_fraction,
            assembly_terminal=args.reference_assembly_terminal,
            success_threshold=args.reference_success_threshold,
        )
        transitions.summary["reference_usage"] = (
            "R Bellman regression only" if args.resume_stage == "R"
            else "V and state-offset calibration, then R Bellman regression"
        )
    native = hydra.utils.instantiate(cfg.model).eval()
    policy_type = NativeActionPolicy if reward_horizon == 1 else MacroDiffusionPolicy
    policy = policy_type(native, cfg.cond_steps, args.q_score_noise_samples)
    settings = default_settings()
    resumed = None
    if args.resume_reward:
        resumed = torch.load(args.resume_reward, map_location=args.device, weights_only=True)
        if resumed["normalization_sha256"] != sha256(normalization):
            raise ValueError("Resumed LFIRL checkpoint has a different normalization")
        if resumed["source_bc_sha256"] != sha256(cfg.model.network_path):
            raise ValueError("Resumed LFIRL Q was recovered from a different BC checkpoint")
        if (resumed["observation_dim"], resumed["action_dim"], resumed["horizon_steps"]) != (
            cfg.obs_dim, cfg.action_dim, reward_horizon
        ):
            raise ValueError("Resumed LFIRL dimensions differ")
        settings = resumed["learner"]["settings"]
    previous_normalization = (
        resumed.get("reward_observation_normalization", "expert_standard")
        if resumed else "expert_standard"
    )
    reward_normalization = args.reward_observation_normalization or previous_normalization
    if resumed and reward_normalization != previous_normalization:
        raise ValueError("Changing reward input normalization requires fitting Q from scratch")
    settings["irl"]["gamma"] = float(cfg.train.gamma)
    settings["irl"]["batch_size"] = args.batch_size
    if args.qhat_passes is not None:
        settings["irl"]["qhat_passes"] = args.qhat_passes
    if args.q_lr is not None:
        settings["q_net"]["lr"] = args.q_lr
    if args.calibration_lr is not None:
        for name in ("b_net", "v_net", "r_net"):
            settings[name]["lr"] = args.calibration_lr
    if args.v_loss is not None:
        settings["v_net"]["loss"]["type"] = args.v_loss
    if args.r_lr is not None:
        settings["r_net"]["lr"] = args.r_lr
    if args.r_loss is not None:
        settings["r_net"]["loss"]["use_huber"] = args.r_loss == "huber"
    if args.r_include_action is not None:
        changed = bool(settings["r_net"].get("include_action", True)) != args.r_include_action
        if resumed and changed and not (args.reset_reward_head or args.reset_calibration_heads):
            parser.error("Changing reward inputs requires an explicit reward-head reset")
        settings["r_net"]["include_action"] = args.r_include_action
    if args.v_random_action_samples is not None:
        settings["v_net"]["loss"]["sample_random_times"] = args.v_random_action_samples
    for name in ("v", "b", "r"):
        value = getattr(args, f"{name}_passes")
        if value is not None:
            if name == "v":
                settings["irl"]["initial_v_passes"] = value
                settings["irl"]["calibrated_v_passes"] = value
            else:
                settings["irl"][f"{name}_passes"] = value
    reward = LFIRLReward(settings, cfg.obs_dim, reward_horizon * cfg.action_dim, args.device)
    if resumed:
        reward.load_state_dict(resumed["learner"],
                               load_reward_head=not (args.reset_reward_head or args.reset_calibration_heads))
    if args.reset_calibration_heads:
        reward.reset_calibration_heads()
    elif args.reset_reward_head:
        reward.reset_reward_head()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[LFIRL data] {json.dumps(transitions.summary)}", flush=True)
    print(
        "[LFIRL] Frozen BC score -> Qhat -> initial V -> offset b -> calibrated V "
        "-> reward. No simulator or f-IRL discriminator.",
        flush=True,
    )
    history = reward.fit(
        transitions, policy, args.output, args.max_batches_per_stage,
        start_stage=args.resume_stage if resumed else "QHAT",
        checkpoint_interval=args.save_every,
        update_normalization=not resumed and reward_normalization == "expert_standard",
        shuffle_reward=args.shuffle_r,
        shuffle_calibration=args.shuffle_calibration,
    )
    terminal_assumption = (
        "Only known interior transitions; no terminal labels invented"
        if args.terminal_mode == "truncate" else
        "Demonstration endpoints assumed terminal, matching upstream missing-done fallback"
    )
    payload = dict(
        algorithm=("LFIRL upstream first-action score with native DPPO BC"
                   if reward_horizon == 1 else "LFIRL with native DPPO macro-action adapter"),
        learner=reward.state_dict(),
        stages=history,
        data=transitions.summary,
        normalization_sha256=sha256(normalization),
        source_bc_checkpoint=str(cfg.model.network_path),
        source_bc_sha256=sha256(cfg.model.network_path),
        horizon_steps=reward_horizon,
        policy_action_horizon=cfg.act_steps,
        normalization_path=str(normalization),
        reward_observation_normalization=reward_normalization,
        expert_dataset_path=str(data_path),
        action_dim=cfg.action_dim,
        observation_dim=cfg.obs_dim,
        gamma=float(cfg.train.gamma),
        source_commit="bce7509b789f1ac361d37618a85ded256af3d5d4",
        limited_training=bool(args.max_batches_per_stage),
        terminal_assumption=terminal_assumption,
        q_training_overrides=(resumed.get("q_training_overrides")
                              if resumed and args.resume_stage != "QHAT" else dict(
                                  passes=args.qhat_passes, learning_rate=args.q_lr,
                                  score_noise_samples=args.q_score_noise_samples,
                              )),
        resumed_reward=str(args.resume_reward) if args.resume_reward else None,
        inherited_q_refinement=(resumed.get("q_refinement") or resumed.get("inherited_q_refinement")) if resumed else None,
        resumed_stage=args.resume_stage if args.resume_reward else None,
        reset_reward_head=args.reset_reward_head,
        reward_head_overrides=dict(learning_rate=args.r_lr, loss=args.r_loss,
                                   shuffle=args.shuffle_r, include_action=args.r_include_action),
        reward_reference_rollouts=str(args.reference_rollouts.resolve()) if args.reference_rollouts else None,
        reference_scope=("calibration_and_reward" if args.reference_rollouts and args.resume_stage != "R"
                         else "reward_only" if args.reference_rollouts else None),
        calibration_overrides=dict(
            learning_rate=args.calibration_lr,
            v_passes=args.v_passes, b_passes=args.b_passes,
            r_passes=args.r_passes, v_loss=args.v_loss,
            random_action_samples=args.v_random_action_samples,
            reset_calibration_heads=args.reset_calibration_heads,
            shuffled=args.shuffle_calibration,
        ),
    )
    torch.save(payload, args.output / "reward.pt")
    report = {key: value for key, value in payload.items() if key != "learner"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[LFIRL saved] {args.output / 'reward.pt'}", flush=True)


if __name__ == "__main__":
    main()
