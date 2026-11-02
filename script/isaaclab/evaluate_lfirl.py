"""Compare BC, selected and final LFIRL/DPPO policies on matched resets.

Evaluation only: no policy/reward optimizer steps and no success replay writes.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from script.isaaclab.run_lab import compose_config, prepare_cuda_libraries
from script.isaaclab.diagnose_firl import seed_all, save_report


def wilson_interval(successes, count):
    z = 1.959963984540054
    rate = successes / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def grasp_lift_proxy(height, distance, width, relative_position):
    """Conservative motion proxy, separate from the original assembly metric."""
    import numpy as np

    if (height.shape != distance.shape or height.shape != width.shape
            or relative_position.shape != (*height.shape, 3)):
        raise ValueError("Grasp diagnostic time axes differ")
    stable = np.ones_like(height, dtype=bool)
    stable[:, 1:] = np.linalg.norm(np.diff(relative_position, axis=1), axis=-1) < .01
    plausible = ((height > .06) & (distance < .04)
                 & (width > .015) & (width < .045) & stable)
    current = np.zeros(len(height), dtype=int)
    longest = current.copy()
    for condition in plausible.T:
        current = np.where(condition, current + 1, 0)
        longest = np.maximum(longest, current)
    return (longest >= 3).tolist()


def restore_reference(agent, physical):
    """Restore recorded physical/control state while retaining candidate gains."""
    import torch
    from env.gym_utils.wrapper.furniture import stack_last_n_obs_dict

    env = agent.venv.unwrapped
    values = torch.as_tensor(physical, device=agent.device, dtype=torch.float32)
    width = env.robot.num_joints
    position = values[:, :width]
    velocity = values[:, width:2 * width]
    cursor = 2 * width
    for obj in [env.robot, *env._parts.values(), *env._obstacles.values()]:
        obj.write_root_state_to_sim(values[:, cursor:cursor + 13])
        cursor += 13
    env.robot.write_joint_state_to_sim(position, velocity)
    env.last_grasp.copy_(values[:, cursor])
    cursor += 1
    env._efforts.copy_(values[:, cursor:cursor + width])
    cursor += width
    env._position_targets.copy_(values[:, cursor:cursor + width])
    cursor += width
    if cursor != values.shape[1]:
        raise ValueError("Physical snapshot schema differs from this environment")
    env.robot.set_joint_position_target(env._position_targets)
    env.robot.set_joint_effort_target(env._efforts)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.)
    env._sync_state()
    agent.venv.obs.clear()
    agent.venv.obs.append(env._get_observation())
    stacked = stack_last_n_obs_dict(agent.venv.obs, agent.n_cond_step)
    return {"state": agent.venv.process_obs(stacked)}


def physical_observation_tolerances(dimension):
    """Numerical tolerances for 16D proprioception and 7D part poses."""
    import numpy as np

    if dimension < 16 or (dimension - 16) % 7:
        raise ValueError("Unknown Furniture observation schema")
    tolerances = np.full(dimension, 1e-5)
    # Positions/width: 10 micrometers. Row-6D rotations: 1e-5.
    # Linear/angular velocities: 1e-4 m/s and rad/s, respectively.
    tolerances[9:15] = 1e-4
    # The original fork converts part rotation matrices with sqrt(1 +/- trace).
    # Cancellation near a zero quaternion component amplifies float32 roundoff
    # to O(sqrt(eps)). Reserve eight eps inside that formula, including its 1/2
    # factor. This applies only to derived part/obstacle quaternion components;
    # restored physical root quaternions still have the independent 1e-6 gate.
    quaternion_roundoff = .5 * np.sqrt(8 * np.finfo(np.float32).eps)
    for start in range(16, dimension, 7):
        tolerances[start + 3:start + 7] = quaternion_roundoff
    return tolerances


def paired_initial_observation(observation, physics, reference, restored=False,
                               observation_scale=None):
    """Match the physical reset and share its initial measurement when restored.

    Recomputing FK/quaternions after PhysX state restoration can introduce small
    float32 differences, magnified by min/max normalization. Record those
    differences while giving both policies the exact same initial measurement.
    """
    import numpy as np

    reference_observation, reference_physics = reference
    physics_difference = float(np.abs(physics - reference_physics).max())
    observation_difference = float(np.abs(observation - reference_observation).max())
    physics_tolerance = 1e-6 if restored else 1e-4
    observation_tolerance = 1e-3 if restored else 1e-4
    invalid_observation = observation_difference > observation_tolerance
    physical_observation_ratio = None
    largest_physical_errors = None
    if restored and observation_scale is not None:
        scale = np.asarray(observation_scale)
        if (scale.shape != observation.shape[-1:] or not np.isfinite(scale).all()
                or (scale <= 0).any()):
            raise ValueError("Expected positive physical scale for every observation feature")
        physical_error = np.abs(observation - reference_observation) * scale
        tolerances = physical_observation_tolerances(observation.shape[-1])
        physical_observation_ratio = float((physical_error / tolerances).max())
        feature_errors = physical_error.reshape(-1, physical_error.shape[-1]).max(axis=0)
        largest_physical_errors = [
            {"feature": int(index), "physical_error": float(feature_errors[index])}
            for index in np.argsort(feature_errors / tolerances)[-5:][::-1]
        ]
        invalid_observation = physical_observation_ratio > 1.
    if physics_difference > physics_tolerance or invalid_observation:
        raise RuntimeError(
            f"Paired reset differs before rollout: physics={physics_difference}, "
            f"observation={observation_difference}, "
            f"physical_observation_tolerance_ratio={physical_observation_ratio}, "
            f"largest_physical_errors={largest_physical_errors}"
        )
    measurement = reference_observation.copy() if restored else observation
    return measurement, observation_difference


def physical_state_snapshot(env):
    """Capture the physical/control fields restored by restore_reference."""
    import torch

    objects = [env.robot, *env._parts.values(), *env._obstacles.values()]
    return torch.cat([
        env.robot.data.joint_pos, env.robot.data.joint_vel,
        *[obj.data.root_state_w for obj in objects],
        env.last_grasp[:, None], env._efforts, env._position_targets,
    ], dim=-1).cpu().numpy().copy()


def read_surface_materials(asset):
    """Optional read-only diagnostic; never modifies simulator materials."""
    try:
        values = asset.root_physx_view.get_material_properties().detach().cpu()
        return {
            "columns": ["static_friction", "dynamic_friction", "restitution"],
            "buffer_shape": list(values.shape),
            "first_environment_shapes": values[0].tolist(),
            "minimum_over_all_shapes": values.reshape(-1, 3).amin(0).tolist(),
            "maximum_over_all_shapes": values.reshape(-1, 3).amax(0).tolist(),
        }
    except Exception as error:
        # An optional material query must not hide the assembly evaluation.
        # Preserve the error explicitly instead of inventing default values.
        return {"measurement_error": repr(error)}


def collect(agent, reward_model, app, output, label, seed, reference=None, restore=False,
            restore_self=False, training_sampler=False):
    import numpy as np
    import torch
    from agent.finetune.lfirl.reward import discounted_action_rewards
    from isaaclab.utils.math import subtract_frame_transforms

    seed_all(seed)
    env = agent.venv.unwrapped
    # Explicitly match a fresh environment's initial open command. The current
    # RL environment also resets this memory in its _reset_franka override.
    env.last_grasp.fill_(-1.)
    observation = agent.venv.reset()
    if restore_self:
        if reference is not None or restore:
            raise ValueError("Self restoration and external reference restoration are exclusive")
        reference = (observation["state"][:, -1].copy(), physical_state_snapshot(env))
        restore = True
    if restore:
        if reference is None:
            raise ValueError("Snapshot restoration requires a baseline reference")
        observation = restore_reference(agent, reference[1])
    initial_observation = observation["state"][:, -1].copy()
    initial_physics = physical_state_snapshot(env)
    recomputed_observation_difference = 0.
    if reference is not None:
        statistics = agent.venv.normalizer.stats["observations"]
        observation_scale = ((statistics["max"] - statistics["min"]) / 2).detach().cpu().numpy()
        try:
            initial_observation, recomputed_observation_difference = paired_initial_observation(
                initial_observation, initial_physics, reference, restored=restore,
                observation_scale=observation_scale,
            )
        except RuntimeError:
            np.savez_compressed(
                output / f"{label}_seed_{seed}_reset_mismatch.npz",
                observation=initial_observation, reference_observation=reference[0],
                physics=initial_physics, reference_physics=reference[1],
                observation_scale=observation_scale,
            )
            raise
        if restore:
            if agent.n_cond_step != 1:
                raise ValueError("Common initial measurement requires one observation step")
            observation = {"state": initial_observation[:, None].copy()}
    leg = env._parts["square_table_leg4"]
    initial_leg = leg.data.root_pos_w.clone()
    seed_all(seed + 100000)
    agent.model.eval()
    pre_states, actions, assembly, heights, distances = [], [], [], [], []
    widths, relative_positions = [], []
    micro_states, micro_returns = [], []
    single_action_reward = reward_model.action_dim == agent.action_dim and agent.act_steps > 1
    with torch.no_grad():
        for step in range(agent.n_steps):
            if not app.is_running():
                raise KeyboardInterrupt("Simulator closed during evaluation")
            pre_states.append(observation["state"][:, -1].copy())
            condition = {"state": torch.as_tensor(
                observation["state"], device=agent.device, dtype=torch.float32
            )}
            samples = agent.model(cond=condition, deterministic=not training_sampler)
            action = samples.trajectories[:, :agent.act_steps].cpu().numpy()
            observation, actual_reward, terminated, truncated, info = agent.venv.step(action)
            if single_action_reward:
                micro = np.concatenate((pre_states[-1][:, None], info["action_observations"][:, :-1]), axis=1)
                values = reward_model.predict(
                    micro.reshape(-1, 1, agent.obs_dim), action.reshape(-1, agent.action_dim),
                ).reshape(agent.n_envs, agent.act_steps)
                micro_states.append(micro)
                micro_returns.append(discounted_action_rewards(values, reward_model.settings["irl"]["gamma"]))
            actions.append(action)
            assembly.append(actual_reward.copy())
            leg_position = leg.data.root_pos_w
            tcp = env.robot.data.body_pos_w[:, env.franka_ee_index]
            heights.append((leg_position[:, 2] - initial_leg[:, 2]).cpu().numpy())
            distances.append(torch.linalg.vector_norm(tcp - leg_position, dim=-1).cpu().numpy())
            relative_position, _ = subtract_frame_transforms(
                tcp, env.robot.data.body_quat_w[:, env.franka_ee_index],
                leg_position, leg.data.root_quat_w,
            )
            widths.append(env.gripper_width().reshape(-1).cpu().numpy())
            relative_positions.append(relative_position.cpu().numpy())
            if step % 20 == 0:
                print(f"[Evaluation] {label} seed={seed} chunk={step}/{agent.n_steps}", flush=True)
            if step < agent.n_steps - 1 and (terminated | truncated).any():
                raise RuntimeError("Unexpected reset inside a paired evaluation episode")

    states = np.stack(pre_states).transpose(1, 0, 2)
    commands = np.stack(actions).transpose(1, 0, 2, 3)
    actual = np.stack(assembly).T
    height = np.stack(heights).T
    distance = np.stack(distances).T
    width = np.stack(widths).T
    relative_position = np.stack(relative_positions).transpose(1, 0, 2)
    if single_action_reward:
        learned = np.stack(micro_returns).T
    else:
        learned = reward_model.predict(
            states.reshape(-1, 1, agent.obs_dim),
            commands.reshape(-1, agent.horizon_steps * agent.action_dim),
        ).reshape(states.shape[:2])
    if not np.isfinite(learned).all():
        raise FloatingPointError("Nonfinite LFIRL evaluation reward")

    current_poses, founds = env._get_parts_poses(sim_coord=True)
    end_matches = [
        all(furniture.is_assembled_idx(
            first, second, current_poses[index].cpu().numpy(), founds[index].cpu().numpy()
        ) for first, second in furniture.should_be_assembled)
        for index, furniture in enumerate(env.furnitures)
    ]
    current_run = np.zeros(agent.n_envs, dtype=int)
    longest_run = current_run.copy()
    for above in (height > .03).T:
        current_run = np.where(above, current_run + 1, 0)
        longest_run = np.maximum(longest_run, current_run)
    trajectory = output / f"{label}_seed_{seed}.npz"
    extra = {"micro_pre_states": np.stack(micro_states).transpose(1, 0, 2, 3)} if micro_states else {}
    np.savez_compressed(
        trajectory, initial_states=initial_observation, initial_physics=initial_physics,
        pre_states=states, actions=commands, assembly_rewards=actual,
        lfirl_rewards=learned, leg_height_gain_m=height, tcp_leg_distance_m=distance,
        gripper_width_m=width, tcp_leg_relative_position_m=relative_position,
        **extra,
    )
    result = dict(
        policy=label, seed=seed, episodes=agent.n_envs,
        recomputed_initial_observation_max_difference=recomputed_observation_difference,
        common_initial_measurement=bool(restore),
        assembly_successes=(actual.sum(1) >= agent.best_reward_threshold_for_success).tolist(),
        final_pose_matches=[bool(value) for value in end_matches],
        sustained_leg_lifts=(longest_run >= 3).tolist(),
        coupled_grasp_lift_proxy=grasp_lift_proxy(height, distance, width, relative_position),
        minimum_tcp_leg_distance_m=distance.min(1).tolist(),
        maximum_leg_height_gain_m=height.max(1).tolist(),
        mean_lfirl_rewards=learned.mean(1).tolist(),
        trajectory=str(trajectory),
    )
    print(f"[Evaluation result] {label} seed={seed}: "
          f"success={sum(result['assembly_successes'])}/{agent.n_envs}, "
          f"end_pose={sum(end_matches)}/{agent.n_envs}", flush=True)
    return result, initial_observation, initial_physics


def summarize(rollouts):
    import numpy as np

    summaries = {}
    for label in dict.fromkeys(row["policy"] for row in rollouts):
        rows = [row for row in rollouts if row["policy"] == label]
        successes = np.concatenate([row["assembly_successes"] for row in rows])
        summaries[label] = dict(
            episodes=len(successes), successes=int(successes.sum()),
            success_rate=float(successes.mean()),
            success_rate_wilson_95=wilson_interval(int(successes.sum()), len(successes)),
            final_pose_matches=sum(sum(row["final_pose_matches"]) for row in rows),
            sustained_leg_lifts=sum(sum(row["sustained_leg_lifts"]) for row in rows),
            coupled_grasp_lift_proxy=sum(sum(row["coupled_grasp_lift_proxy"]) for row in rows),
            mean_minimum_tcp_leg_distance_m=float(np.mean([
                value for row in rows for value in row["minimum_tcp_leg_distance_m"]
            ])),
            mean_lfirl_reward=float(np.mean([
                value for row in rows for value in row["mean_lfirl_rewards"]
            ])),
        )
    return summaries


def fixed_eta_for_probe(value, device):
    """Represent a frozen native eta, including exactly one, with finite state."""
    import math
    from model.diffusion.eta import EtaFixed

    if not math.isfinite(value) or not 0. < value <= 1.:
        raise ValueError("Frozen eta must be positive and at most one")
    if value == 1.:
        # EtaFixed's inverse tanh is infinite at its upper bound. The original
        # formula with equal fixed bounds returns exactly one for any logit.
        eta = EtaFixed(base_eta=.5, min_eta=0., max_eta=1.)
        eta.min = eta.max = 1.
    else:
        eta = EtaFixed(base_eta=value, min_eta=0., max_eta=1.)
    return eta.to(device)


def main():
    prepare_cuda_libraries()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[4242, 10001])
    parser.add_argument("--policy", choices=["all", "bc", "best", "final"], default="all")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--restore-reference", action="store_true")
    parser.add_argument(
        "--restore-self", action="store_true",
        help="Restore the baseline's own snapshot so both policies use the same restoration procedure.",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--training-sampler", action="store_true",
        help="Probe the native stochastic training sampler without optimizer updates.",
    )
    parser.add_argument(
        "--fixed-eta", type=float,
        help="Explicit frozen-sampler probe override; applied after checkpoint loading.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.fixed_eta is not None:
        if not args.training_sampler or not 0. < args.fixed_eta <= 1.:
            parser.error("--fixed-eta requires --training-sampler and a finite value in (0, 1]")
    if args.restore_self and (args.restore_reference or args.reference_dir):
        parser.error("--restore-self cannot be combined with an external reference")
    if args.restore_self and args.policy != "bc":
        parser.error("--restore-self requires a standalone --policy bc baseline")
    source = args.run_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory to preserve earlier evaluations")
    checkpoints = {
        "bc": source / "checkpoint/state_0.pt",
        "best": source / "checkpoint/best.pt",
        "final": source / f"checkpoint/state_{args.iterations - 1}.pt",
    }
    labels = list(checkpoints) if args.policy == "all" else [args.policy]
    reward_name = {"bc": "lfirl_0.pt", "best": "lfirl_best.pt"}.get(
        args.policy, f"lfirl_{args.iterations - 1}.pt"
    )
    reward_path = source / "checkpoint" / reward_name
    for path in (checkpoints["bc"], *(checkpoints[label] for label in labels), reward_path):
        if not path.is_file():
            parser.error(f"Required evaluation checkpoint missing: {path}")
    output.mkdir(parents=True)
    cfg = compose_config("cfg/furniture/finetune/one_leg_low", "ft_lfirl_diffusion_mlp", [
        "wandb=null", f"env.n_envs={args.num_envs}", f"logdir={output / 'runtime'}",
        *args.override,
    ])
    from omegaconf import OmegaConf, open_dict

    if (source / "config.yaml").is_file():
        cfg = OmegaConf.merge(
            OmegaConf.load(source / "config.yaml"),
            OmegaConf.from_dotlist([
                "wandb=null", f"env.n_envs={args.num_envs}", f"logdir={output / 'runtime'}",
                *(value.lstrip("+") for value in args.override),
            ]),
        )

    with open_dict(cfg):
        cfg.device = args.device
        cfg.model.device = args.device
        cfg.env.specific.headless = args.headless
        cfg.env.specific.gpu_id = int(args.device.split(":")[-1])
        if args.fixed_eta is not None:
            # The exact probe value is installed after checkpoint loading.
            cfg.model.eta.base_eta = .5 if args.fixed_eta == 1. else args.fixed_eta
            cfg.model.eta.min_eta = 0.
            cfg.model.eta.max_eta = 1.
    OmegaConf.save(cfg, output / "evaluation_config.yaml")
    sys.path.insert(0, str(ROOT / "external/furniture-bench"))
    launcher = AppLauncher(args)
    agent = None
    report = dict(
        source_run=str(source), original_best=json.loads((source / "checkpoint/best.json").read_text()),
        evaluation_reward_checkpoint=str(reward_path),
        native_sampler_mode="training_probe" if args.training_sampler else "evaluation",
        fixed_eta_override=args.fixed_eta, seeds=args.seeds, environments=args.num_envs,
        reset_protocol="Native seeded reset, with gripper command memory initialized open for every episode",
        configuration_overrides=args.override,
        physical_snapshot_restored=args.restore_reference or args.restore_self,
        self_snapshot_restoration=args.restore_self,
        restored_observation_tolerances=physical_observation_tolerances(cfg.obs_dim).tolist(),
        restored_observation_tolerance_units="m for positions/width, m/s for linear velocity, rad/s for angular velocity, dimensionless for rotation encodings",
        training_during_evaluation=False, rollouts=[], reset_comparisons=[],
        caveats=[
            "Assembly success uses the original one-time pose event, not tightening torque.",
            "End-pose matching is reported separately; gripper release is not required.",
            "Leg lift is a diagnostic proxy, not proof of a stable grasp.",
            "The coupled lift proxy is a motion diagnostic, not measured grasp contact; it does not affect reward or checkpoint selection.",
        ],
        coupled_lift_proxy_thresholds=dict(
            height_gain_m=.06, maximum_tcp_distance_m=.04,
            gripper_width_range_m=[.015, .045],
            maximum_relative_position_change_m=.01,
            minimum_consecutive_post_chunk_samples=3,
        ),
    )
    provenance_path = source / "evaluation_provenance.json"
    if provenance_path.is_file():
        report["source_experiment_provenance"] = json.loads(provenance_path.read_text())
    try:
        import numpy as np
        import torch
        from agent.finetune.train_agent import TrainAgent
        from agent.finetune.lfirl.reward import LFIRLReward

        agent = TrainAgent(cfg)
        environment = agent.venv.unwrapped
        if str(args.device).startswith("cuda"):
            free_memory, total_memory = torch.cuda.mem_get_info(args.device)
            report["gpu_memory_after_scene_init_bytes"] = dict(
                free=free_memory, total=total_memory,
                used_global=total_memory - free_memory,
                scope="All GPU processes at measurement time, not per-process usage",
            )
        report["applied_arm_gains"] = {
            "stiffness": environment.robot.data.joint_stiffness[:, environment._arm_ids].cpu().tolist(),
            "damping": environment.robot.data.joint_damping[:, environment._arm_ids].cpu().tolist(),
        }
        report["applied_finger_joint_friction"] = (
            environment.robot.data.joint_friction_coeff[:, environment._finger_ids].cpu().tolist()
        )
        report["applied_joint_friction_model"] = (
            "legacy_force_scaled_coefficient" if environment.legacy_joint_friction
            else "static_effort"
        )
        report["applied_finger_friction_efforts"] = (
            environment.robot.root_physx_view.get_dof_friction_properties()[:, environment._finger_ids].cpu().tolist()
        )
        if environment.legacy_joint_friction:
            report["applied_finger_legacy_friction_coefficients"] = (
                environment.robot.root_physx_view.get_dof_friction_coefficients()[:, environment._finger_ids].cpu().tolist()
            )
        report["applied_joint_armature"] = environment.robot.data.joint_armature.cpu().tolist()
        report["applied_gripper_command_memory"] = (
            "upstream_raw_action" if environment.upstream_gripper_memory else "accepted_command"
        )
        report["applied_finger_velocity_limit"] = (
            environment.robot.data.joint_vel_limits[:, environment._finger_ids].cpu().tolist()
        )
        report["applied_body_inertias"] = environment.robot.root_physx_view.get_inertias().cpu().tolist()
        try:
            masses = environment.robot.root_physx_view.get_masses().detach().cpu()
            report["runtime_robot_mass_diagnostic"] = dict(
                body_names=environment.robot.body_names,
                first_environment_masses_kg=masses[0].tolist(),
                minimum_per_body_kg=masses.amin(dim=0).tolist(),
                maximum_per_body_kg=masses.amax(dim=0).tolist(),
                physics_changed_by_query=False,
                original_gym_runtime_mass_equivalence_verified=False,
            )
        except Exception as error:
            report["runtime_robot_mass_diagnostic"] = dict(measurement_error=repr(error))
        try:
            part_masses = {}
            for name, asset in environment._parts.items():
                values = asset.root_physx_view.get_masses().detach().cpu().reshape(-1)
                part_masses[name] = dict(first_environment_mass_kg=float(values[0]),
                                         minimum_mass_kg=float(values.min()),
                                         maximum_mass_kg=float(values.max()))
            report["runtime_furniture_mass_diagnostic"] = dict(parts=part_masses,
                                                               physics_changed_by_query=False)
        except Exception as error:
            report["runtime_furniture_mass_diagnostic"] = dict(measurement_error=repr(error))
        report["surface_material_diagnostic"] = {
            "robot": read_surface_materials(environment.robot),
            "parts": {
                name: read_surface_materials(asset)
                for name, asset in environment._parts.items()
            },
            "obstacles": {
                name: read_surface_materials(asset)
                for name, asset in environment._obstacles.items()
            },
            "physics_changed_by_query": False,
            "original_Gym_asset_material_equivalence_verified": False,
        }
        saved_reward = torch.load(reward_path, map_location=args.device, weights_only=True)
        reward_model = LFIRLReward(
            saved_reward["learner"]["settings"], saved_reward["observation_dim"],
            saved_reward["horizon_steps"] * saved_reward["action_dim"], args.device,
        )
        reward_model.load_state_dict(saved_reward["learner"])
        agent.venv.record_action_observations = saved_reward["horizon_steps"] == 1 and agent.act_steps > 1
        original = torch.load(saved_reward["source_bc_checkpoint"], map_location="cpu", weights_only=True)
        initial = torch.load(checkpoints["bc"], map_location="cpu", weights_only=True)["model"]
        report["bc_matches_original_checkpoint"] = all(
            torch.equal(value, initial["actor_ft." + name[len("network."):]])
            for name, value in original["ema"].items() if name.startswith("network.")
        )
        if not report["bc_matches_original_checkpoint"]:
            raise RuntimeError("The reference policy does not match the original BC")
        report["checkpoint_sha256"] = {
            label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in checkpoints.items() if label in labels or label == "bc"
        }
        report["checkpoint_paths"] = {
            label: str(path) for label, path in checkpoints.items()
            if label in labels or label == "bc"
        }
        references = {}
        if args.reference_dir:
            for seed in args.seeds:
                with np.load(args.reference_dir / f"bc_seed_{seed}.npz") as reference:
                    references[seed] = (
                        reference["initial_states"].copy(), reference["initial_physics"].copy()
                    )
        for label, path in checkpoints.items():
            if args.policy != "all" and label != args.policy:
                continue
            saved = torch.load(path, map_location=args.device, weights_only=True)
            agent.model.load_state_dict(saved["model"], strict=True)
            if args.fixed_eta is not None:
                # Loading a checkpoint restores eta_logit too. Replace only the
                # native fixed-eta module after loading; actor weights stay fixed.
                agent.model.eta = fixed_eta_for_probe(args.fixed_eta, args.device)
                if not torch.isfinite(agent.model.eta.eta_logit).all():
                    raise ValueError("Requested eta is too close to a floating-point endpoint")
            configured_eta = float(agent.model.eta({
                "state": torch.zeros(1, 1, cfg.obs_dim, device=args.device),
            }).item())
            report.setdefault("applied_sampler", {})[label] = {
                "eta": configured_eta if args.training_sampler else 0.,
                "configured_eta": configured_eta,
                "minimum_denoising_std": float(agent.model.get_min_sampling_denoising_std()),
                "deterministic": not args.training_sampler,
            }
            for seed in args.seeds:
                reference = references.get(seed)
                result, initial_obs, initial_physics = collect(
                    agent, reward_model, launcher.app, output, label, seed,
                    reference=reference,
                    restore=args.restore_reference,
                    restore_self=args.restore_self,
                    training_sampler=args.training_sampler,
                )
                if reference is None:
                    references[seed] = (initial_obs, initial_physics)
                else:
                    reference_obs, reference_physics = reference
                    comparison = dict(
                        policy=label, seed=seed,
                        maximum_observation_difference=float(np.abs(initial_obs - reference_obs).max()),
                        maximum_physics_difference=float(np.abs(initial_physics - reference_physics).max()),
                    )
                    report["reset_comparisons"].append(comparison)
                    if comparison["maximum_physics_difference"] > 1e-4:
                        raise RuntimeError(f"Paired resets differ physically: {comparison}")
                report["rollouts"].append(result)
                report["summary"] = summarize(report["rollouts"])
                save_report(output / "report.json", report)
        with (output / "episodes.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["policy", "seed", "env", "success", "end_pose", "sustained_lift", "mean_lfirl_reward"])
            for row in report["rollouts"]:
                for index in range(row["episodes"]):
                    writer.writerow([
                        row["policy"], row["seed"], index, row["assembly_successes"][index],
                        row["final_pose_matches"][index], row["sustained_leg_lifts"][index],
                        row["mean_lfirl_rewards"][index],
                    ])
        report["completed"] = True
        save_report(output / "report.json", report)
        print(json.dumps(report["summary"], indent=2), flush=True)
    except BaseException as error:
        import traceback

        traceback.print_exc()
        report["completed"] = False
        report["error"] = repr(error)
        save_report(output / "report.json", report)
        sys.stderr.flush()
        raise
    finally:
        if agent is not None:
            agent.venv.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
