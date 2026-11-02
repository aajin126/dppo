"""View every recorded Furniture NPZ demonstration in Isaac Sim.

State mode displays recorded object poses and reconstructs the arm from its
recorded TCP pose. Joint angles are not present in the NPZ. Action mode sends
recorded commands through the existing physical environment without a policy.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.isaaclab.run_lab import prepare_cuda_libraries


class Demonstrations:
    def __init__(self, path, normalizer):
        import numpy as np
        import torch

        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as source:
            self.states = source["states"].astype(np.float32)
            self.actions = source["actions"].astype(np.float32)
            lengths = source["traj_lengths"].astype(np.int64)
        if lengths.sum() != len(self.states) or len(self.actions) != len(self.states):
            raise ValueError("NPZ trajectory lengths, states and actions do not align")
        stats = torch.load(normalizer, map_location="cpu", weights_only=True)
        self.observation_min = stats["stats.observations.min"].numpy()
        self.observation_span = stats["stats.observations.max"].numpy() - self.observation_min
        self.action_min = stats["stats.actions.min"].numpy()
        self.action_span = stats["stats.actions.max"].numpy() - self.action_min
        valid = ~((self.states == 0).all(-1) & (self.actions == 0).all(-1))
        boundaries = np.r_[0, lengths.cumsum()]
        self.rows = [np.flatnonzero(valid[start:end]) + start for start, end in zip(boundaries[:-1], boundaries[1:])]
        self.playable = [i for i, rows in enumerate(self.rows) if len(rows)]
        if not self.playable:
            raise ValueError("NPZ contains no nonpadding demonstrations")
        self.summary = dict(
            dataset=str(self.path), declared_demonstrations=len(lengths),
            playable_demonstrations=len(self.playable), valid_frames=int(valid.sum()),
            empty_demonstrations=[i + 1 for i, rows in enumerate(self.rows) if not len(rows)],
            stored_fields=["states", "actions", "traj_lengths"],
            recorded_joint_angles_available=False,
        )

    def frame(self, demo, frame):
        row = self.rows[demo][frame]
        state = (self.states[row] + 1) * .5 * self.observation_span + self.observation_min
        action = (self.actions[row] + 1) * .5 * self.action_span + self.action_min
        return state, action


class Viewer:
    def __init__(self, env, dataset, app, args):
        import numpy as np
        import torch
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from pxr import Gf, UsdGeom

        self.env, self.dataset, self.app, self.args = env, dataset, app, args
        expected = 16 + 7 * (env.furniture.num_parts + 1)
        if dataset.states.shape[1] != expected or dataset.actions.shape[1] != 10:
            raise ValueError(f"Selected task expects state/action dimensions {expected}/10")
        self.demos = dataset.playable[args.start_demo - 1:]
        if args.demo_limit:
            self.demos = self.demos[:args.demo_limit]
        if not self.demos:
            raise ValueError("No demonstrations selected by --start-demo")
        self.demo_index, self.frame_index = 0, 0
        self.paused, self.finished, self.pending_demo = False, False, 0
        self.needs_reset = True
        self.total_frames, self.episode_return = 0, 0.
        self.metrics, self.episode_metrics = [], []
        self.label = None
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1, device=env.device,
        )
        self.marker = UsdGeom.Sphere.Define(env.sim.stage, "/World/RecordedTCP")
        self.marker.CreateRadiusAttr(.004)
        self.marker.CreateDisplayColorAttr([Gf.Vec3f(1., .15, .05)])
        self.marker_translation = UsdGeom.Xformable(self.marker).AddTranslateOp()
        self.nominal_obstacles = {name: np.asarray(pose) for name, pose in env._fixed_poses.items()}
        if not args.headless:
            env.sim.set_camera_view((1.05, -.85, 1.25), (-.10, 0., .63))
            import omni.ui as ui
            self.window = ui.Window("Recorded NPZ demonstrations", width=530, height=270)
            with self.window.frame:
                with ui.VStack(spacing=5):
                    ui.Label(f"Dataset: {dataset.path.parent.name}/{dataset.path.name}")
                    if args.mode == "states":
                        ui.Label("RECORDED STATES: objects are set from the NPZ.", word_wrap=True)
                        ui.Label("Arm joints reconstructed by IK; no physical success test.", word_wrap=True)
                    else:
                        ui.Label("RECORDED ACTIONS: physics rollout, no learned policy.", word_wrap=True)
                        ui.Label("Initial arm joints reconstructed; object velocities unavailable.", word_wrap=True)
                    ui.Label("Red sphere = recorded TCP" if args.mode == "states" else "Red sphere = recorded action TCP target")
                    self.label = ui.Label("", word_wrap=True)
                    with ui.HStack():
                        ui.Button("Previous", clicked_fn=lambda: self.request_demo(-1))
                        ui.Button("Pause / Resume", clicked_fn=self.toggle_pause)
                        ui.Button("Next", clicked_fn=lambda: self.request_demo(1))
                        ui.Button("Restart", clicked_fn=lambda: self.request_demo(0))
        self.last_error = (0., 0.)

    def toggle_pause(self):
        self.paused = not self.paused

    def request_demo(self, delta):
        self.pending_demo = delta
        self.needs_reset = True
        self.finished = False

    def forward(self):
        self.env.sim.forward()
        self.env.scene.update(self.env._physics_dt)

    def set_objects(self, state):
        import numpy as np
        import torch
        import furniture_bench.controllers.control_utils as control

        count = self.env.furniture.num_parts
        self.env._reset_parts(0, state[16:16 + count * 7])
        supplied = state[16 + count * 7:16 + (count + 1) * 7]
        front = self.env._part_reset_pose(None, supplied)
        nominal = self.nominal_obstacles["obstacle_front"]

        def matrix(pose):
            tensor = torch.as_tensor(pose, device=self.env.device, dtype=torch.float32)
            return control.pose2mat(tensor[:3], tensor[3:7], device=self.env.device)

        delta = matrix(front) @ torch.linalg.inv(matrix(nominal))
        for name, obstacle in self.env._obstacles.items():
            position, quaternion = control.mat2pose(delta @ matrix(self.nominal_obstacles[name]))
            pose = torch.cat((position, quaternion)).cpu().numpy()
            obstacle.write_root_pose_to_sim(self.env._pose_to_world(pose, [0]))
            obstacle.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.env.device))
        self.forward()

    def target(self, state):
        import torch
        from isaaclab.utils.math import quat_from_matrix
        from furniture_bench.isaaclab.control_utils_lab import rotation_6d_to_matrix

        robot = self.env.robot
        position = torch.as_tensor(state[:3], device=self.env.device)[None]
        position = position + robot.data.body_pos_w[:, self.env.franka_base_index]
        rotation = torch.as_tensor(state[3:9], device=self.env.device)[None]
        quaternion = quat_from_matrix(rotation_6d_to_matrix(rotation))
        self.marker_translation.Set(tuple(float(x) for x in position[0]))
        return position, quaternion

    def pose_error(self, position, quaternion):
        import torch
        from isaaclab.utils.math import quat_error_magnitude

        data = self.env.robot.data
        p = data.body_pos_w[:, self.env.franka_ee_index]
        q = data.body_quat_w[:, self.env.franka_ee_index]
        return float(torch.linalg.vector_norm(position - p)), float(quat_error_magnitude(quaternion, q))

    def reconstruct_arm(self, state, first=False):
        import torch

        robot = self.env.robot
        target_position, target_quaternion = self.target(state)
        self.ik.set_command(torch.cat((target_position, target_quaternion), -1))
        bounds = robot.data.joint_pos_limits[:, self.env._arm_ids]
        for _ in range(120 if first else self.args.ik_iterations):
            data = robot.data
            position = data.body_pos_w[:, self.env.franka_ee_index]
            quaternion = data.body_quat_w[:, self.env.franka_ee_index]
            error = self.pose_error(target_position, target_quaternion)
            if error[0] < .0005 and error[1] < .01:
                break
            jacobian = robot.root_physx_view.get_jacobians()[:, self.env._jacobian_id, :, self.env._arm_ids]
            current = data.joint_pos[:, self.env._arm_ids]
            desired = self.ik.compute(position, quaternion, jacobian, current)
            desired = current + (desired - current).clamp(-.15, .15)
            desired = desired.clamp(bounds[..., 0] + .001, bounds[..., 1] - .001)
            full = data.joint_pos.clone()
            full[:, self.env._arm_ids] = desired
            full[:, self.env._finger_ids] = float(state[15]) / 2
            robot.write_joint_state_to_sim(full, torch.zeros_like(full))
            self.env._position_targets[:] = full
            robot.set_joint_position_target(full)
            self.forward()
        full = robot.data.joint_pos.clone()
        full[:, self.env._finger_ids] = float(state[15]) / 2
        robot.write_joint_state_to_sim(full, torch.zeros_like(full))
        self.env._position_targets[:] = full
        robot.set_joint_position_target(full)
        self.forward()
        self.env._sync_state()
        return self.pose_error(target_position, target_quaternion)

    def reset_demo(self):
        self.demo_index = max(0, min(len(self.demos) - 1, self.demo_index + self.pending_demo))
        self.pending_demo = 0
        self.frame_index, self.episode_return = 0, 0.
        self.episode_metrics = []
        self.env.reset()
        demo = self.demos[self.demo_index]
        state, _ = self.dataset.frame(demo, 0)
        self.set_objects(state)
        error = self.reconstruct_arm(state, first=True)
        if self.env.furniture_name == "lamp":
            import torch
            bulb = self.env._parts["lamp_bulb"].data
            self.env._bulb_rest_pose = torch.cat((bulb.root_pos_w, bulb.root_quat_w), -1).clone()
        self.needs_reset = False
        print(f"[NPZ] demo={demo + 1}/{len(self.dataset.rows)} frames={len(self.dataset.rows[demo])} mode={self.args.mode} initial_TCP_error_mm={error[0] * 1000:.3f}", flush=True)

    def show_frame(self):
        import numpy as np
        import torch

        demo = self.demos[self.demo_index]
        state, action = self.dataset.frame(demo, self.frame_index)
        if self.args.mode == "states":
            self.set_objects(state)
            error = self.reconstruct_arm(state)
        else:
            _, reward, _, _ = self.env.step(torch.as_tensor(action, device=self.env.device)[None])
            error = self.pose_error(*self.target(action))
            self.episode_return += float(reward.sum())
        self.last_error = error
        self.episode_metrics.append(error)
        self.total_frames += 1
        if self.label is not None:
            self.label.text = (
                f"Demo {demo + 1}/{len(self.dataset.rows)} | playable {self.demo_index + 1}/{len(self.demos)}\n"
                f"Frame {self.frame_index + 1}/{len(self.dataset.rows[demo])} | speed {self.args.speed:g}x\n"
                f"TCP reconstruction/tracking error: {error[0] * 1000:.2f} mm, {np.rad2deg(error[1]):.2f} deg"
            )
        self.env.sim.render()
        if self.args.snapshot and not self.args.headless and self.total_frames == 10:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
            self.args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            capture_viewport_to_file(get_active_viewport(), str(self.args.snapshot))
        self.frame_index += 1
        if self.frame_index == len(self.dataset.rows[demo]):
            errors = np.asarray(self.episode_metrics)
            result = dict(
                demo=demo + 1, frames=len(errors), mode=self.args.mode,
                mean_tcp_error_mm=float(errors[:, 0].mean() * 1000),
                max_tcp_error_mm=float(errors[:, 0].max() * 1000),
                mean_rotation_error_deg=float(np.rad2deg(errors[:, 1]).mean()),
                environment_assembly_return=self.episode_return if self.args.mode == "actions" else None,
            )
            self.metrics.append(result)
            print(f"[NPZ completed] {json.dumps(result)}", flush=True)
            if self.demo_index + 1 < len(self.demos):
                self.pending_demo = 1
                self.needs_reset = True
            elif self.args.loop:
                self.pending_demo = -self.demo_index
                self.needs_reset = True
            else:
                self.finished = True
                if self.label is not None:
                    self.label.text += "\nAll selected demonstrations finished."

    def run(self):
        import torch

        deadline = time.monotonic()
        with torch.no_grad():
            while self.app.is_running():
                if self.args.max_frames and self.total_frames >= self.args.max_frames:
                    break
                if self.finished and (self.args.headless or self.args.exit_after_playback):
                    break
                if self.needs_reset:
                    self.reset_demo()
                    deadline = time.monotonic()
                if not self.paused and not self.finished and self.env.sim.is_playing() and time.monotonic() >= deadline:
                    self.show_frame()
                    deadline = max(deadline + 1 / (self.args.fps * self.args.speed), time.monotonic())
                else:
                    self.env.sim.render()
                    time.sleep(.005)


def main():
    prepare_cuda_libraries()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/furniture/one_leg_low/train.npz")
    parser.add_argument("--task", choices=["one_leg", "lamp", "round_table"], default="one_leg")
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--mode", choices=["states", "actions"], default="states")
    parser.add_argument("--start-demo", type=int, default=1, help="One-based index among nonempty demonstrations")
    parser.add_argument("--demo-limit", type=int, default=0, help="Zero plays every nonempty demonstration")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional automated-check limit")
    parser.add_argument("--fps", type=float, default=10.)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--ik-iterations", type=int, default=24)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--exit-after-playback", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--snapshot", type=Path, help="Optional viewport image after ten displayed frames")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.fps, args.speed, args.ik_iterations, args.start_demo) <= 0:
        parser.error("Playback speed, FPS, IK iterations and start demo must be positive")
    if args.demo_limit < 0 or args.max_frames < 0:
        parser.error("Demo and frame limits must be nonnegative")
    args.data = args.data if args.data.is_absolute() else ROOT / args.data
    normalization = args.normalization or args.data.with_name("normalization.pth")
    if not normalization.is_absolute():
        normalization = ROOT / normalization
    sys.path.insert(0, str(ROOT / "external/furniture-bench"))
    launcher = AppLauncher(args)
    env, viewer = None, None
    try:
        from furniture_bench.isaaclab.furniture_rl_sim_env_lab import FurnitureRLSimEnv

        dataset = Demonstrations(args.data, normalization)
        print(f"[NPZ dataset] {json.dumps(dataset.summary)}", flush=True)
        env = FurnitureRLSimEnv(
            furniture=args.task, num_envs=1, headless=args.headless,
            gpu_id=int(args.device.split(":")[-1]), max_env_steps=100000,
        )
        viewer = Viewer(env, dataset, launcher.app, args)
        viewer.run()
    finally:
        if args.report and viewer is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(dict(
                **viewer.dataset.summary, mode=args.mode, displayed_frames=viewer.total_frames,
                completed_demonstrations=viewer.metrics,
                last_tcp_error_mm=viewer.last_error[0] * 1000,
            ), indent=2) + "\n")
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
