"""Verify that controller overrides reach the Furniture backend."""

from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env.gym_utils import make_async


class FurnitureFactoryTests(unittest.TestCase):
    def test_launcher_preserves_yaml_environment_count_and_explicit_overrides(self):
        from script.isaaclab.run_lab import compose_config

        directory = "cfg/furniture/finetune/one_leg_low"
        original = compose_config(directory, "ft_ppo_diffusion_mlp", [])
        self.assertEqual(original.env.n_envs, 1000)
        lfirl = compose_config(directory, "ft_lfirl_diffusion_mlp", [])
        self.assertEqual(lfirl.env.n_envs, 4)
        overridden = compose_config(directory, "ft_ppo_diffusion_mlp", ["env.n_envs=32"])
        self.assertEqual(overridden.env.n_envs, 32)

    def test_controller_gains_reach_each_backend(self):
        for backend in ("isaacgym", "isaaclab"):
            cases = [{}, {"stiffness": 4000., "damping": 100.}]
            if backend == "isaaclab":
                cases.append({"gripper_joint_friction": 0.})
                cases.append({"legacy_link_armature": .01})
                cases.append({"empty_frame_mass": .001})
                cases.append({"upstream_gripper_memory": True})
                cases.append({"gripper_velocity_limit": .06})
                cases.append({"batched_observations": True})
                cases.append({"ground_usd": "/local/original_grid.usd"})
            for overrides in cases:
                with self.subTest(backend=backend, overrides=overrides):
                    observation = ModuleType("furniture_bench.envs.observation")
                    observation.DEFAULT_STATE_OBS = ["robot_state", "parts_poses"]
                    wrapper = ModuleType("env.gym_utils.wrapper.furniture")
                    wrapper.FurnitureRLSimEnvMultiStepWrapper = Mock()
                    module_name = (
                        "furniture_bench.isaaclab.furniture_rl_sim_env_lab"
                        if backend == "isaaclab"
                        else "furniture_bench.envs.furniture_rl_sim_env"
                    )
                    module = ModuleType(module_name)
                    module.FurnitureRLSimEnv = Mock()
                    with patch.dict(sys.modules, {
                        observation.__name__: observation,
                        wrapper.__name__: wrapper,
                        module_name: module,
                    }):
                        make_async("one_leg", env_type="furniture", backend=backend, **overrides)
                    arguments = module.FurnitureRLSimEnv.call_args.kwargs
                    self.assertEqual(arguments["stiffness"], overrides.get("stiffness", 1000))
                    self.assertEqual(arguments["damping"], overrides.get("damping", 200))
                    if "gripper_joint_friction" in overrides:
                        self.assertEqual(arguments["gripper_joint_friction"], 0.)
                    else:
                        self.assertNotIn("gripper_joint_friction", arguments)
                    if "legacy_link_armature" in overrides:
                        self.assertEqual(arguments["legacy_link_armature"], .01)
                    else:
                        self.assertNotIn("legacy_link_armature", arguments)
                    if "empty_frame_mass" in overrides:
                        self.assertEqual(arguments["empty_frame_mass"], .001)
                    else:
                        self.assertNotIn("empty_frame_mass", arguments)
                    if "upstream_gripper_memory" in overrides:
                        self.assertTrue(arguments["upstream_gripper_memory"])
                    else:
                        self.assertNotIn("upstream_gripper_memory", arguments)
                    if "gripper_velocity_limit" in overrides:
                        self.assertEqual(arguments["gripper_velocity_limit"], .06)
                    else:
                        self.assertNotIn("gripper_velocity_limit", arguments)
                    if "batched_observations" in overrides:
                        self.assertTrue(arguments["batched_observations"])
                    if "ground_usd" in overrides:
                        self.assertEqual(arguments["ground_usd"], overrides["ground_usd"])
                    else:
                        self.assertNotIn("ground_usd", arguments)

    def test_gripper_calibration_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "require the IsaacLab backend"):
            make_async("one_leg", env_type="furniture", backend="isaacgym", gripper_joint_friction=0.)


if __name__ == "__main__":
    unittest.main()
