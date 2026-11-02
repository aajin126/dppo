"""Check that absorbing assembly boundaries preserve the native reward event."""

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.finetune.lfirl.episodes import assembly_boundaries, finite_episode_returns
from util.reward_scaling import RunningRewardScaler


class AssemblyEpisodeTests(unittest.TestCase):
    def test_finite_returns_stop_at_completion_and_at_the_observed_budget(self):
        rewards = np.array([[-2., -1.], [-1., -2.], [-999., -3.]])
        valid = np.array([[True, True], [True, True], [False, True]])
        actual = finite_episode_returns(rewards, valid, .9)
        np.testing.assert_allclose(actual, [[-2.9, -5.23], [-1., -4.7], [0., -3.]])

    def test_completion_chunk_is_retained_and_other_environments_continue(self):
        native = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
        original = native.copy()
        valid, terminal = assembly_boundaries(native, 1)
        np.testing.assert_array_equal(valid, [[True, True, True], [True, False, True], [False, False, True]])
        np.testing.assert_array_equal(terminal, [[False, True, False], [True, False, False], [False, False, False]])
        np.testing.assert_array_equal(native, original)

    def test_multicomponent_goal_and_last_chunk_success(self):
        valid, terminal = assembly_boundaries(np.array([[1., 0.], [0., 0.], [1., 2.]]), 2)
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(terminal, [[False, False], [False, False], [True, True]])

    def test_terminal_credit_cannot_depend_on_later_cost_or_bootstrap(self):
        valid, terminal = assembly_boundaries(np.array([[0.], [1.], [0.]]), 1)
        for post_value in (0., 1000.):
            rewards = np.array([-2., -1., -1000.])
            values = np.array([0., 0., post_value])
            advantage = np.zeros(3)
            following = 0.
            for time in reversed(range(3)):
                next_value = values[time + 1] if time < 2 else 9999.
                continuing = 1 - terminal[time, 0]
                delta = rewards[time] + .9 * continuing * next_value - values[time]
                following = advantage[time] = delta + .9 * continuing * following
            np.testing.assert_allclose(advantage[valid[:, 0]], [-2.9, -1.])

    def test_masked_scaler_excludes_absorbed_rows_and_all_valid_is_identical(self):
        reward = np.array([[-2., -1., 0.], [-1., -2., -3.]])
        first = np.array([[1., 0., 0.], [1., 0., 0.]])
        baseline, same = RunningRewardScaler(2), RunningRewardScaler(2)
        np.testing.assert_array_equal(baseline(reward, first), same(reward, first, valid=np.ones_like(reward, dtype=bool)))
        self.assertEqual(baseline.ret_rms.count, same.ret_rms.count)
        masked = RunningRewardScaler(2)
        valid = np.array([[True, True, False], [True, True, True]])
        count_before = masked.ret_rms.count
        masked(reward, first, valid=valid)
        self.assertEqual(masked.ret_rms.count - count_before, int(valid.sum()))

    def test_invalid_events_are_rejected(self):
        with self.assertRaises(ValueError):
            assembly_boundaries(np.array([[np.nan]]), 1)
        with self.assertRaises(ValueError):
            assembly_boundaries(np.array([[-1.]]), 1)
        with self.assertRaises(ValueError):
            assembly_boundaries(np.zeros((2, 2)), 0)

    def test_actor_probe_rejects_long_or_ungated_training_configurations(self):
        from omegaconf import OmegaConf
        from script.isaaclab.probe_lfirl_advantages import ProbeOnePolicyIteration

        for iterations, warmup, forced, assembly in ((5, 4, True, True),
                                                    (3, 2, False, True),
                                                    (3, 2, True, False),
                                                    (4, 2, True, True)):
            cfg = OmegaConf.create(dict(train=dict(n_train_itr=iterations, n_critic_warmup_itr=warmup,
                                                   force_train=forced), lfirl=dict(assembly_terminal=assembly)))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                ProbeOnePolicyIteration(cfg)

    def test_credit_audit_preserves_policy_gradients_and_random_state(self):
        import json
        import tempfile
        from types import SimpleNamespace
        import torch
        from model.common.critic import CriticObs
        from model.diffusion.diffusion_ppo import PPODiffusion
        from model.diffusion.eta import EtaFixed
        from model.diffusion.mlp_diffusion import DiffusionMLP
        from script.isaaclab.probe_lfirl_advantages import ProbeLFIRLAdvantages

        model = PPODiffusion(
            actor=DiffusionMLP(action_dim=2, horizon_steps=2, cond_dim=3,
                               time_dim=8, mlp_dims=[16, 16]),
            critic=CriticObs(cond_dim=3, mlp_dims=[16, 16]),
            eta=EtaFixed(base_eta=.2), device="cpu", ft_denoising_steps=2,
            horizon_steps=2, action_dim=2, obs_dim=3, denoising_steps=10,
            use_ddim=True, ddim_steps=2, gamma_denoising=.9, clip_ploss_coef=.001,
        )
        observations = torch.randn(4, 8, 1, 3)
        with torch.no_grad():
            chains = model(cond={"state": observations.reshape(32, 1, 3)},
                           return_chain=True).chains.reshape(4, 8, 3, 2, 2).numpy()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        for parameter in model.actor_ft.parameters():
            parameter.grad = torch.ones_like(parameter)
        rng = torch.get_rng_state().clone()
        events = np.zeros((4, 8), dtype=np.float32)
        events[-1, :4] = 1
        with tempfile.TemporaryDirectory() as directory:
            probe = SimpleNamespace(
                model=model, probe_native_rewards=events, n_envs=8, n_steps=4,
                gamma=.9, device="cpu", batch_size=16, use_bc_loss=True,
                bc_loss_coeff=.1, reward_horizon=2, itr=0, probe_directory=Path(directory),
                reward_scale_const=1.,
            )
            ProbeLFIRLAdvantages.audit_credit_gradients(
                probe, observations, chains, np.arange(32).reshape(4, 8).astype(np.float32),
                np.ones((4, 8), dtype=bool),
                -np.ones((4, 8), dtype=np.float32), np.zeros((4, 8)),
            )
            report = json.loads((Path(directory) / "gradient_credit_0.json").read_text())
            self.assertEqual(len(report["batches"]), 3)
            self.assertEqual(report["optimizer_steps_by_audit"], 0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        for parameter in model.actor_ft.parameters():
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
