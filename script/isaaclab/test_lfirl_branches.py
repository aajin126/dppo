"""Check intervention construction and paired branch-plan alignment."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.isaaclab.check_lfirl_branches import make_hold, make_plan, close_grip
from script.isaaclab.analyze_lfirl_branches import matched_physics, preference


class BranchTests(unittest.TestCase):
    def test_measured_returns_stop_at_timeout_and_never_mix_environments(self):
        from script.isaaclab.calibrate_lfirl_value import discounted_returns

        rewards = np.array([[1., 10.], [2., 20.], [4., 40.]])
        np.testing.assert_array_equal(discounted_returns(rewards, .5),
                                      np.array([[3., 30.], [4., 40.], [4., 40.]]))

    def test_value_probe_refuses_a_configuration_that_could_update_the_actor(self):
        from omegaconf import OmegaConf
        from script.isaaclab.probe_lfirl_advantages import ProbeLFIRLAdvantages

        cfg = OmegaConf.create(dict(train=dict(n_train_itr=3, n_critic_warmup_itr=2)))
        with self.assertRaisesRegex(ValueError, "never permit actor updates"):
            ProbeLFIRLAdvantages(cfg)

    def test_closed_grip_intervention_preserves_pose_commands_exactly(self):
        class Identity:
            def __call__(self, values, key, forward=True):
                return values

        actions = torch.linspace(-.8, .8, 160).reshape(2, 8, 10)
        original = actions.clone()
        closed = close_grip(Identity(), actions)
        np.testing.assert_array_equal(closed[:, :, :9], original[:, :, :9].numpy())
        np.testing.assert_array_equal(closed[:, :, 9], np.ones((2, 8)))
        torch.testing.assert_close(actions, original, rtol=0, atol=0)
    def test_physical_pairing_rejects_different_positions_and_forces(self):
        state = np.zeros((3, 154), dtype=np.float32)
        root = state[:, 18:135].reshape(3, 9, 13)
        root[:, :, 6] = 1.
        other = state.copy()
        other[0, 18] = .0001
        other[1, 136] = .01
        other[2, 18:135].reshape(9, 13)[:, 3:7] *= -1
        valid, errors, _ = matched_physics(state, other)
        self.assertEqual(valid.tolist(), [False, False, True])
        self.assertGreater(errors["root_position_m"], 5e-5)

    def test_direction_check_ignores_small_and_nonrepeatable_physical_effects(self):
        score = np.array([1., -1., 1.])
        effects = np.array([.02, .02, .001])
        result = preference(score, effects, np.zeros(3), .005)
        self.assertEqual(result["informative"], 2)
        self.assertEqual(result["direction_agreement"], .5)
        result = preference(score, effects, np.array([.02, 0., 0.]), .005)
        self.assertEqual(result["informative"], 1)
        self.assertEqual(result["direction_agreement"], 0.)

    def test_hold_changes_only_pose_and_preserves_all_gripper_commands(self):
        class Identity:
            def __call__(self, values, key, forward=True):
                return values

        states = torch.linspace(-.8, .8, 116).reshape(2, 58)
        actions = torch.linspace(-.9, .9, 160).reshape(2, 8, 10)
        original = actions.clone()
        held, feasible = make_hold(Identity(), states, actions)
        np.testing.assert_array_equal(held[:, :, :9], states[:, None, :9].expand(-1, 8, -1).numpy())
        np.testing.assert_array_equal(held[:, :, 9], actions[:, :, 9].numpy())
        torch.testing.assert_close(actions, original, rtol=0, atol=0)
        self.assertTrue(feasible.all())
        states[0, 0] = 2.
        _, feasible = make_hold(Identity(), states, actions)
        self.assertEqual(feasible.tolist(), [False, True])

    def test_plan_uses_pre_state_and_excludes_successful_prefixes_and_timeouts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            height = np.zeros((6, 12), dtype=np.float32)
            distance = np.full((6, 12), .1, dtype=np.float32)
            height[:2, 2:4] = .10
            height[:2, 4:] = .02
            height[2:4, 2:] = .10
            distance[2:4] = .02
            distance[4:, 3:] = .07
            assembly = np.zeros_like(height)
            assembly[0, 0] = 1.
            np.savez(path, leg_height_gain_m=height, tcp_leg_distance_m=distance,
                     gripper_width_m=np.full_like(height, .03),
                     tcp_leg_relative_position_m=np.zeros((6, 12, 3)), assembly_rewards=assembly)
            plan = make_plan(path, horizon=3, per_group=1)
            self.assertEqual(len(plan["rows"]), 3)
            self.assertEqual(len({row["env_id"] for row in plan["rows"]}), 3)
            drop = next(row for row in plan["rows"] if row["group"] == "recorded_height_drop")
            self.assertEqual(drop["env_id"], 1)
            self.assertEqual(drop["step"], 4)
            for row in plan["rows"]:
                self.assertLessEqual(row["step"] + plan["horizon"], 11)
                self.assertEqual(float(assembly[row["env_id"], :row["step"]].sum()), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA normalizer")
    def test_cpu_commands_with_cuda_environment_normalizer(self):
        from env.gym_utils.furniture_normalizer import LinearNormalizer

        normalizer = LinearNormalizer()
        normalizer.load_state_dict({
            "stats.observations.min": torch.full((58,), -1., device="cuda"),
            "stats.observations.max": torch.ones(58, device="cuda"),
            "stats.actions.min": torch.full((10,), -1., device="cuda"),
            "stats.actions.max": torch.ones(10, device="cuda"),
        })
        observation = np.full((2, 58), .2, dtype=np.float32)
        action = np.full((2, 8, 10), -.3, dtype=np.float32)
        held, feasible = make_hold(normalizer, observation, action)
        np.testing.assert_allclose(held[:, :, :9], .2, atol=1e-6)
        np.testing.assert_allclose(held[:, :, 9], -.3, atol=1e-6)
        self.assertTrue(feasible.all())


if __name__ == "__main__":
    unittest.main()
