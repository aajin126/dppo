"""Verify source-faithful DPPO batched state conversion without a simulator."""

import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external/furniture-bench"))
from furniture_bench.isaaclab.furniture_rl_sim_env_lab import FurnitureRLSimEnv
import furniture_bench.controllers.control_utils as C
from furniture_bench.furniture import furniture_factory
import furniture_bench.utils.transform as T


class BatchedObservationTests(unittest.TestCase):
    def test_gripper_raw_memory_matches_original_rl_source_including_deadband(self):
        from furniture_bench.isaaclab.control_utils_lab import upstream_gripper_command

        source = ROOT / "external/furniture-bench/furniture_bench/envs/furniture_rl_sim_env.py"
        tree = ast.parse(source.read_text())
        # This module defines its own FurnitureSimEnv; FurnitureRLSimEnv
        # inherits simulate_step from that local class, not the other file.
        original = next(method for node in tree.body if isinstance(node, ast.ClassDef)
                        for method in node.body if isinstance(method, ast.FunctionDef)
                        and method.name == "simulate_step")
        selected = [copy.deepcopy(node) for node in original.body if isinstance(node, ast.Assign)
                    and ast.unparse(node.targets[0]) in ("grasp", "grip_sep", "self.last_grasp")]
        self.assertEqual(len(selected), 3)
        function = ast.FunctionDef(
            name="reference", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self"), ast.arg(arg="action")],
                                                 kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=selected + [ast.Return(value=ast.Tuple(elts=[ast.Name(id="grip_sep", ctx=ast.Load()),
                                ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()),
                                              attr="last_grasp", ctx=ast.Load())], ctx=ast.Load()))],
            decorator_list=[],
        )
        namespace = {"torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     str(source), "exec"), namespace)
        state = SimpleNamespace(last_grasp=-torch.ones(11), grasp_margin=.019, max_gripper_width=.065)
        memory = state.last_grasp.clone()
        rng = torch.Generator().manual_seed(2026110604)
        commands = torch.cat((torch.zeros(4, 11), torch.full((4, 11), -.005),
                              torch.full((4, 11), .005), torch.rand(128, 11, generator=rng) * 2 - 1))
        for grasp in commands:
            action = torch.zeros(11, 10)
            action[:, -1] = grasp
            expected, expected_memory = namespace["reference"](state, action)
            actual, memory = upstream_gripper_command(grasp, memory, .019, .065)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(memory, expected_memory, rtol=0, atol=0)

    def test_repeated_small_opposite_grip_uses_raw_previous_action(self):
        from furniture_bench.isaaclab.control_utils_lab import upstream_gripper_command

        separation, memory = upstream_gripper_command(torch.tensor([-.005]), torch.ones(1), .019, .065)
        self.assertEqual(float(separation[0]), 0.)
        separation, memory = upstream_gripper_command(torch.tensor([-.005]), memory, .019, .065)
        torch.testing.assert_close(separation, torch.tensor([.065]))
        torch.testing.assert_close(memory, torch.tensor([-.005]))

    def test_empty_frame_selection_excludes_geometry_inertia_and_moving_links(self):
        import tempfile
        from furniture_bench.isaaclab.assets_lab import empty_fixed_frame_names

        xml = """<robot name="frames">
        <link name="base"/><link name="frame"/>
        <link name="physical"><inertial/></link>
        <link name="geometry"><collision/></link><link name="moving"/>
        <joint name="f" type="fixed"><parent link="base"/><child link="frame"/></joint>
        <joint name="p" type="fixed"><parent link="base"/><child link="physical"/></joint>
        <joint name="g" type="fixed"><parent link="base"/><child link="geometry"/></joint>
        <joint name="m" type="revolute"><parent link="base"/><child link="moving"/></joint>
        </robot>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.urdf"
            path.write_text(xml)
            self.assertEqual(empty_fixed_frame_names(path), ("frame",))
            self.assertEqual(path.read_text(), xml)

    def test_original_robot_has_nine_empty_fixed_frames(self):
        from furniture_bench.isaaclab.assets_lab import empty_fixed_frame_names

        source = ROOT / "external/furniture-bench/furniture_bench/assets/franka_description_ros/franka_description/robots/franka_panda.urdf"
        self.assertEqual(set(empty_fixed_frame_names(source)),
                         {*(f"panda_link{index}_sc" for index in range(8)), "k_ee_link"})

    def test_empty_frame_mass_probe_preserves_real_links_and_input_properties(self):
        from furniture_bench.isaaclab.assets_lab import regularize_empty_frame_properties

        masses = torch.tensor([[2., 1.], [3., 1.]])
        inertias = torch.eye(3).reshape(1, 1, 9).repeat(2, 2, 1) * .0001
        original_masses, original_inertias = masses.clone(), inertias.clone()
        changed_mass, changed_inertia = regularize_empty_frame_properties(
            ["physical", "frame"], masses, inertias, ("frame",), .001)
        torch.testing.assert_close(masses, original_masses, rtol=0, atol=0)
        torch.testing.assert_close(inertias, original_inertias, rtol=0, atol=0)
        torch.testing.assert_close(changed_mass[:, 0], masses[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(changed_inertia[:, 0], inertias[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(changed_mass[:, 1], torch.full((2,), .001))
        self.assertTrue((torch.linalg.eigvalsh(changed_inertia.reshape(2, 2, 3, 3)) > 0).all())
        for mass in (0., -1., 1., float("nan")):
            with self.assertRaises(ValueError):
                regularize_empty_frame_properties(["physical", "frame"], masses, inertias, ("frame",), mass)

    def test_gym_friction_coefficient_is_not_written_as_static_effort(self):
        class View:
            def __init__(self):
                self.coefficients = torch.zeros(2, 9)
                self.properties = torch.zeros(2, 9, 3)
                self.properties[:, 7:] = 15.

            def get_dof_friction_coefficients(self):
                return self.coefficients

            def set_dof_friction_coefficients(self, values, indices):
                self.coefficients[indices.long()] = values[indices.long()]

            def get_dof_friction_properties(self):
                return self.properties

        view = View()
        def write_new_model(static, dynamic, viscous, joint_ids):
            view.properties[:, joint_ids] = torch.tensor([static, dynamic, viscous])

        env = FurnitureRLSimEnv.__new__(FurnitureRLSimEnv)
        env.num_envs, env._finger_ids = 2, [7, 8]
        env.sim_config = {"robot": {"gripper_frictions": [15., 15.]}}
        env.gripper_joint_friction = None
        env.robot = SimpleNamespace(
            root_physx_view=view,
            write_joint_friction_coefficient_to_sim=write_new_model,
        )
        env._configure_legacy_joint_friction()
        torch.testing.assert_close(view.coefficients[:, 7:], torch.full((2, 2), 15.))
        self.assertFalse(view.properties[:, 7:].any())
        self.assertFalse(view.coefficients[:, :7].any())

    def test_original_tensor_assembly_events_match_existing_one_leg_metric(self):
        from furniture_bench.isaaclab.upstream_rl import native_reward, SOURCE

        self.assertEqual(Path(native_reward.__code__.co_filename), SOURCE)
        furniture = furniture_factory("one_leg")
        count = 64
        poses = np.zeros((count, 5, 7), dtype=np.float32)
        poses[..., 6] = 1.
        rng = np.random.default_rng(14)
        for index in range(count):
            target = furniture.assembled_rel_poses[(0, 4)][index % 4].copy()
            target[:3, 3] += rng.uniform(-.008, .008, 3)
            angle = .1 if index % 2 else .7
            rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                                 [0, np.sin(angle), np.cos(angle)]])
            target[:3, :3] = target[:3, :3] @ rotation
            poses[index, 4, :3] = target[:3, 3]
            poses[index, 4, 3:] = T.mat2quat(target[:3, :3])
        expected = [furniture.is_assembled_idx(0, 4, row.reshape(-1), np.ones(5)) for row in poses]
        env = FurnitureRLSimEnv.__new__(FurnitureRLSimEnv)
        env.furniture, env.num_envs, env.device = furniture, count, torch.device("cpu")
        env.furnitures = [copy.deepcopy(furniture) for _ in range(count)]
        env.batched_rewards, env.manual_label, env.manual_done = True, False, False
        env.assembled_rel_poses = torch.tensor(np.asarray([furniture.assembled_rel_poses[(0, 4)]]))
        env.already_assembled = torch.zeros(count, 1, dtype=torch.bool)
        env._get_parts_poses = lambda sim_coord=False: (torch.from_numpy(poses.reshape(count, -1)), torch.ones(count, 5))
        actual = env._reward()[:, 0].bool().tolist()
        self.assertEqual(actual, expected)
        self.assertTrue(any(actual))
        self.assertFalse(all(actual))
        self.assertFalse(env._reward().any())
        self.assertEqual([item.all_assembled() for item in env.furnitures], expected)

    def test_transform_is_reused_from_native_rl_environment(self):
        paths = [
            ROOT / "external/furniture-bench/furniture_bench/envs/furniture_rl_sim_env.py",
            ROOT / "external/furniture-bench/furniture_bench/isaaclab/furniture_rl_sim_env_lab.py",
        ]
        bodies = []
        for path in paths:
            tree = ast.parse(path.read_text())
            method = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
                          for node in cls.body if isinstance(node, ast.FunctionDef)
                          and node.name == "sim_pose_to_april_pose")
            bodies.append(ast.dump(method, include_attributes=False))
        self.assertEqual(*bodies)

    def test_pose_order_and_founds_match_original_batch_operations(self):
        torch.manual_seed(17)
        env = FurnitureRLSimEnv.__new__(FurnitureRLSimEnv)
        env.num_envs, env.device, env.batched_observations = 8, torch.device("cpu"), True
        env.furniture = SimpleNamespace(parts=[SimpleNamespace(name=f"part{i}") for i in range(5)])
        env.pose_dim = 7
        env.franka_from_origin_mat = np.eye(4, dtype=np.float32)
        env.base_tag_from_robot_mat = np.eye(4, dtype=np.float32)
        env.rb_states = torch.randn(48, 13)
        env.rb_states[:, 3:7] = torch.nn.functional.normalize(env.rb_states[:, 3:7], dim=-1)
        env.part_idxs = {f"part{i}": (torch.arange(8) * 6 + i).tolist() for i in range(5)}
        env.part_idxs["obstacle_front"] = (torch.arange(8) * 6 + 5).tolist()
        raw = torch.stack([env.rb_states[env.part_idxs[part.name], :7] for part in env.furniture.parts], 1)
        transforms = C.pose2mat_batched(raw[..., :3], raw[..., 3:7], device=env.device)
        expected = torch.cat(C.mat2pose_batched(transforms), -1).reshape(8, -1)
        actual, founds = env._get_parts_poses()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(founds, torch.ones(8, 5)))
        torch.testing.assert_close(env._get_parts_poses(True)[0], raw.reshape(8, -1), rtol=0, atol=0)
        self.assertEqual(tuple(env.get_obstacle_pose().shape), (8, 7))


if __name__ == "__main__":
    unittest.main()
