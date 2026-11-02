"""Check sampling/score consistency without changing the pretrained sampler."""

import copy
import unittest

import torch

from model.common.critic import CriticObs
from model.diffusion.diffusion_ppo import PPODiffusion
from model.diffusion.eta import EtaFixed
from model.diffusion.mlp_diffusion import DiffusionMLP


def make_model(**kwargs):
    return PPODiffusion(
        actor=DiffusionMLP(action_dim=2, horizon_steps=2, cond_dim=3,
                           time_dim=8, mlp_dims=[16, 16]),
        critic=CriticObs(cond_dim=3, mlp_dims=[16, 16]),
        eta=EtaFixed(base_eta=.2), device='cpu', ft_denoising_steps=2,
        horizon_steps=2, action_dim=2, obs_dim=3, denoising_steps=10,
        use_ddim=True, ddim_steps=2, gamma_denoising=.9, clip_ploss_coef=.001,
        min_sampling_denoising_std=.005, **kwargs,
    )


class ScoreContractTests(unittest.TestCase):
    def test_original_clipping_defaults_remain_exact(self):
        model = make_model()
        values = torch.linspace(-10, 10, 100)
        torch.testing.assert_close(model.clip_logprobs(values), values.clamp(-5, 2), rtol=0, atol=0)

    def test_small_noise_score_retains_gradient_with_matching_upper_bound(self):
        mean = torch.tensor([.001], requires_grad=True)
        action = torch.tensor([.002])
        score = torch.distributions.Normal(mean, .005).log_prob(action)
        old = make_model().clip_logprobs(score)
        old_gradient = torch.autograd.grad(old.sum(), mean, retain_graph=True)[0]
        corrected = make_model(logprob_clip_max=5).clip_logprobs(score)
        corrected_gradient = torch.autograd.grad(corrected.sum(), mean)[0]
        torch.testing.assert_close(old_gradient, torch.zeros_like(old_gradient))
        torch.testing.assert_close(corrected_gradient, (action - mean.detach()) / .005**2)

    def test_score_changes_leave_training_and_evaluation_sampler_exact(self):
        original = make_model()
        corrected = copy.deepcopy(original)
        corrected.min_logprob_denoising_std = .005
        corrected.logprob_clip_max = 5.
        condition = {'state': torch.randn(8, 1, 3)}
        for deterministic in (False, True):
            torch.manual_seed(123)
            first = original(cond=condition, deterministic=deterministic, return_chain=True)
            torch.manual_seed(123)
            second = corrected(cond=condition, deterministic=deterministic, return_chain=True)
            torch.testing.assert_close(first.trajectories, second.trajectories, rtol=0, atol=0)
            torch.testing.assert_close(first.chains, second.chains, rtol=0, atol=0)

    def test_matched_transition_scores_equal_the_sampling_gaussian(self):
        model = make_model(min_logprob_denoising_std=.005, logprob_clip_max=5.)
        condition = {'state': torch.randn(8, 1, 3)}
        sample = model(cond=condition, deterministic=False, return_chain=True)
        indices = torch.ones(8, dtype=torch.long)
        previous, following = sample.chains[:, 1], sample.chains[:, 2]
        timesteps = model.ddim_t[-model.ft_denoising_steps:][indices]
        mean, logvar, _ = model.p_mean_var(previous, timesteps, cond=condition, index=indices)
        actual_std = (.5 * logvar).exp().clamp_min(model.get_min_sampling_denoising_std())
        expected = torch.distributions.Normal(mean, actual_std).log_prob(following)
        measured = model.get_logprobs_subsample(condition, previous, following, indices)
        torch.testing.assert_close(expected, measured, rtol=0, atol=0)

    def test_invalid_clipping_bounds_are_rejected(self):
        for low, high in ((2, 2), (3, -1), (float('nan'), 5), (-5, float('inf'))):
            with self.assertRaisesRegex(ValueError, 'bounds'):
                make_model(logprob_clip_min=low, logprob_clip_max=high)


if __name__ == '__main__':
    unittest.main()
