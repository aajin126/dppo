"""Focused checks for upstream f-IRL reuse, reward learning and data boundaries."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.finetune.firl.dataset import FIRLExpertStates
from agent.finetune.firl.learner import FIRLRewardLearner, f_div_disc_loss
from agent.finetune.train_firl_diffusion_agent import TrainFIRLDiffusionAgent
from firl.divs.f_div_disc import f_div_disc_loss as upstream_loss
from firl.models.reward import MLPReward


def settings():
    return OmegaConf.create(dict(
        discriminator=dict(
            hidden_size=16, batch_size=32, learning_rate=.003, updates=40
        ),
        reward=dict(
            hidden_sizes=[16, 16], clamp_magnitude=10,
            learning_rate=.003, weight_decay=0, updates=4,
        ),
    ))


class FIRLTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_original_loss_and_reward_are_used(self):
        self.assertIs(f_div_disc_loss, upstream_loss)
        learner = FIRLRewardLearner(2, "cpu", settings())
        self.assertIsInstance(learner.reward, MLPReward)
        trajectories = np.random.normal(size=(8, 5, 2)).astype(np.float32)
        # None action/log-probability entries exercise the upstream IS=False
        # branch without inventing an executable-action likelihood.
        loss, _ = upstream_loss(
            "js", False, (trajectories, None, None),
            learner.disc, learner.reward, "cpu",
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in learner.reward.parameters()))

    def test_separate_reward_learns_expert_preference(self):
        learner = FIRLRewardLearner(2, "cpu", settings())
        expert = np.random.normal([1, 0], .05, size=(256, 2)).astype(np.float32)
        trajectories = np.random.normal(size=(16, 4, 2)).astype(np.float32) * .05
        trajectories[:8, :, 0] += 1
        trajectories[8:, :, 0] -= 1
        probes = np.array([[1, 0], [-1, 0]], dtype=np.float32)
        before = np.diff(learner.predict(probes))[0]
        for _ in range(4):
            metrics = learner.update(expert, trajectories)
        after = np.diff(learner.predict(probes))[0]
        self.assertLess(after, before)
        self.assertGreater(metrics["reward_gradient_norm"], 0)
        self.assertEqual(metrics["reward_updates"], 16)
        self.assertEqual(metrics["discriminator_updates"], 160)
        self.assertEqual(metrics["policy_trajectories"], 16)

    def test_covariance_rejects_one_trajectory(self):
        learner = FIRLRewardLearner(2, "cpu", settings())
        with self.assertRaisesRegex(ValueError, "two policy trajectories"):
            learner.update(np.ones((4, 2)), np.ones((1, 4, 2)))

    def test_checkpoint_restores_reward_and_update_counts(self):
        learner = FIRLRewardLearner(2, "cpu", settings())
        trajectories = np.random.normal(size=(4, 3, 2)).astype(np.float32)
        learner.update(np.ones((8, 2)), trajectories)
        restored = FIRLRewardLearner(2, "cpu", settings())
        restored.load_state_dict(learner.state_dict())
        np.testing.assert_array_equal(
            learner.predict(trajectories.reshape(-1, 2)),
            restored.predict(trajectories.reshape(-1, 2)),
        )
        self.assertEqual(learner.reward_updates, restored.reward_updates)

    def test_few_shot_selection_uses_whole_valid_demonstrations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            states = np.array([[0, 0], [1, 1], [1, 2], [2, 1], [2, 2], [3, 1]], dtype=np.float32)
            actions = np.ones((6, 1), dtype=np.float32)
            actions[0] = 0
            np.savez(path, states=states, actions=actions, traj_lengths=np.array([1, 2, 2, 1]))
            all_data = FIRLExpertStates(path, 2)
            selected = FIRLExpertStates(path, 2, max_trajectories=1)
            self.assertEqual(len(all_data.states), 5)
            self.assertNotIn(0, selected.episode_ids)
            self.assertEqual(len(selected.episode_ids), 1)
            self.assertTrue((selected.states[:, 0] == selected.episode_ids[0]).all())

    def test_frozen_reward_uses_post_states_and_preserves_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = TrainFIRLDiffusionAgent.__new__(TrainFIRLDiffusionAgent)
            agent.settings = OmegaConf.create(dict(freeze_reward=True))
            agent.obs_dim = 2
            agent.firl_dir = Path(directory)
            agent.itr = 0
            agent.use_wandb = False
            agent.reward_learner = FIRLRewardLearner(2, "cpu", settings())
            before = {
                name: value.clone()
                for name, value in agent.reward_learner.reward.state_dict().items()
            }
            after_states = np.arange(12, dtype=np.float32).reshape(3, 2, 1, 2) / 10
            environment_reward = np.full((3, 2), 999.0)
            expected = agent.reward_learner.predict(after_states.reshape(-1, 2)).reshape(3, 2)
            actual = agent.get_training_rewards(
                np.zeros_like(after_states), after_states, None,
                environment_reward, np.zeros((3, 2)),
            )
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(agent.reward_learner.reward_updates, 0)
            for name, value in agent.reward_learner.reward.state_dict().items():
                torch.testing.assert_close(before[name], value, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
