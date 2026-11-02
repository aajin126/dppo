"""Regression checks for success replay, frozen-BC anchoring and best selection."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.finetune.firl.success import SuccessfulRolloutBuffer
from agent.finetune.train_firl_diffusion_agent import TrainFIRLDiffusionAgent
from model.common.critic import CriticObs
from model.diffusion.diffusion_ppo import PPODiffusion
from model.diffusion.mlp_diffusion import DiffusionMLP


def buffer(directory, capacity=2, normalizer="test"):
    return SuccessfulRolloutBuffer(directory, capacity, (1, 2), (2, 1), "test", normalizer)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        torch.manual_seed(7)

    def test_success_recording_can_run_without_auxiliary_bc_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = TrainFIRLDiffusionAgent.__new__(TrainFIRLDiffusionAgent)
            agent.settings = OmegaConf.create(dict(success_replay=dict(
                enabled=True, capacity=2, batch_size=2, updates=0, coefficient=0., checkpoint=None,
            )))
            agent.retention_dir = Path(directory)
            agent.normalizer_hash = "test"
            agent.act_steps = agent.horizon_steps = 2
            agent.n_cond_step, agent.obs_dim, agent.action_dim = 1, 2, 1
            agent.initialize_retention(OmegaConf.create(dict(env_name="test")))
            agent.success_replay.add(np.zeros((2, 1, 2)), np.zeros((2, 2, 1)), {})
            agent.itr, agent.n_critic_warmup_itr = 10, 1
            self.assertEqual(len(agent.success_replay), 1)
            self.assertEqual(agent.auxiliary_policy_loss(), 0.)

    def test_only_actual_success_is_stored_and_stops_at_first_success(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = TrainFIRLDiffusionAgent.__new__(TrainFIRLDiffusionAgent)
            agent.success_replay = buffer(directory)
            agent.itr = 4
            agent.best_reward_threshold_for_success = 1
            obs = np.arange(12, dtype=np.float32).reshape(3, 2, 1, 2)
            chains = np.arange(36, dtype=np.float32).reshape(3, 2, 3, 2, 1)
            original = chains.copy()
            reward = np.array([[0, 0], [1, 0], [0, 0]], dtype=np.float32)
            agent.on_rollout_collected(obs, chains, reward, [(0, 0, 2), (1, 0, 2)], False)
            self.assertEqual(len(agent.success_replay), 1)
            episode = agent.success_replay.episodes[0]
            np.testing.assert_array_equal(episode["observations"], obs[:2, 0])
            np.testing.assert_array_equal(episode["actions"], chains[:2, 0, -1])
            np.testing.assert_array_equal(chains, original)
            self.assertEqual(episode["metadata"]["success_chunk"], 1)
            self.assertEqual(episode["metadata"]["environment_return"], 1)

    def test_capacity_persistence_and_schema_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = buffer(Path(directory) / "original")
            for i in range(3):
                replay.add(np.full((2, 1, 2), i), np.full((2, 2, 1), i), dict(iteration=i))
            self.assertEqual([e["metadata"]["iteration"] for e in replay.episodes], [1, 2])
            self.assertEqual(len(list(replay.directory.glob("episode_*.npz"))), 3)
            path = Path(directory) / "replay.pt"
            torch.save(replay.state_dict(), path)
            saved = torch.load(path, weights_only=True)
            restored = buffer(Path(directory) / "restored")
            restored.load_state_dict(saved)
            self.assertEqual(len(restored), 2)
            self.assertEqual(restored.next_id, 3)
            obs, actions = restored.sample(8, "cpu")
            self.assertEqual(tuple(obs.shape), (8, 1, 2))
            self.assertEqual(tuple(actions.shape), (8, 2, 1))
            with self.assertRaisesRegex(ValueError, "normalization"):
                buffer(Path(directory) / "wrong", normalizer="other").load_state_dict(saved)

    def test_best_selection_uses_assembly_and_keeps_earlier_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = TrainFIRLDiffusionAgent.__new__(TrainFIRLDiffusionAgent)
            agent.settings = OmegaConf.create(dict(save_best=True))
            agent.checkpoint_dir = directory
            agent.best_evaluation = None
            agent.best_reward_threshold_for_success = 1

            def save_model():
                for name in ("state", "firl"):
                    torch.save({"itr": agent.itr}, Path(directory) / f"{name}_{agent.itr}.pt")

            agent.save_model = save_model
            for iteration, returns in enumerate(([0, 0], [0, 0], [0, 1], [0, 0])):
                agent.itr = iteration
                agent._maybe_save_best(returns)
            self.assertEqual(torch.load(Path(directory) / "best.pt", weights_only=True)["itr"], 2)
            self.assertEqual(torch.load(Path(directory) / "firl_best.pt", weights_only=True)["itr"], 2)
            metadata = json.loads((Path(directory) / "best.json").read_text())
            self.assertEqual(metadata["success_rate"], .5)
            self.assertEqual(metadata["episodes"], 2)

    def test_success_bc_updates_only_trainable_actor_and_respects_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = TrainFIRLDiffusionAgent.__new__(TrainFIRLDiffusionAgent)
            agent.success_replay = buffer(directory)
            agent.success_replay.add(np.ones((3, 1, 2)), np.zeros((3, 2, 1)), {})
            agent.success_settings = OmegaConf.create(dict(batch_size=4, updates=1, coefficient=.1))
            agent._success_bc_losses = []
            agent.device = "cpu"
            agent.itr = 1
            agent.n_critic_warmup_itr = 1
            agent.model = PPODiffusion(
                actor=DiffusionMLP(action_dim=1, horizon_steps=2, cond_dim=2,
                                   time_dim=8, mlp_dims=[16, 16]),
                critic=CriticObs(cond_dim=2, mlp_dims=[16, 16]),
                horizon_steps=2, obs_dim=2, action_dim=1, denoising_steps=4,
                ft_denoising_steps=2, use_ddim=True, ddim_steps=2,
                gamma_denoising=.9, clip_ploss_coef=.01, device="cpu",
            )
            frozen = {k: v.clone() for k, v in agent.model.actor.state_dict().items()}
            trained = {k: v.clone() for k, v in agent.model.actor_ft.state_dict().items()}
            optimizer = torch.optim.Adam(agent.model.actor_ft.parameters(), lr=.01)
            loss = agent.auxiliary_policy_loss()
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.model.actor_ft.parameters()))
            self.assertTrue(all(p.grad is None for p in agent.model.actor.parameters()))
            optimizer.step()
            self.assertTrue(any(not torch.equal(v, agent.model.actor_ft.state_dict()[k]) for k, v in trained.items()))
            self.assertTrue(all(torch.equal(v, agent.model.actor.state_dict()[k]) for k, v in frozen.items()))
            self.assertEqual(agent.auxiliary_policy_loss(), 0.0)
            agent._success_bc_losses = []
            agent.itr = 0
            self.assertEqual(agent.auxiliary_policy_loss(), 0.0)


if __name__ == "__main__":
    unittest.main()
