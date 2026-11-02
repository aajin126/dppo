"""Focused CPU checks for the AIRL bridge and mixed final-action likelihood."""

import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from agent.finetune.airl.density import CalibratedActionDensity, MixedActionDensity
from agent.finetune.airl.reward import DPPOAIRLReward
from agent.finetune.airl.validation import (
    DensityValidationError,
    ValidatedActionDensity,
    ks_uniform,
)
from imitation.algorithms.adversarial.airl import AIRL


class AIRLReuseTests(unittest.TestCase):
    def test_upstream_methods_are_reused_and_update_reward(self):
        self.assertIs(DPPOAIRLReward.train_disc, AIRL.train_disc)
        self.assertIs(DPPOAIRLReward.logits_expert_is_high, AIRL.logits_expert_is_high)
        with tempfile.TemporaryDirectory() as directory:
            adapter = DPPOAIRLReward(3, (2, 2), .99, "cpu", directory, 8)
            try:
                expert = dict(
                    obs=torch.randn(8, 3),
                    acts=torch.randn(8, 2, 2),
                    next_obs=torch.randn(8, 3),
                    dones=torch.zeros(8),
                    logp=torch.randn(8),
                )
                generated = {name: value.clone() + .1 for name, value in expert.items()}
                generated["dones"].zero_()
                expected = adapter.reward_train(
                    expert["obs"], expert["acts"], expert["next_obs"], expert["dones"]
                ) - expert["logp"]
                actual = adapter.logits_expert_is_high(
                    expert["obs"], expert["acts"], expert["next_obs"],
                    expert["dones"], expert["logp"],
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                before = next(adapter.reward_train.parameters()).detach().clone()
                metrics = adapter.train_disc(expert_samples=expert, gen_samples=generated)
                self.assertTrue(np.isfinite(metrics["disc_loss"]))
                self.assertFalse(torch.equal(before, next(adapter.reward_train.parameters())))
                self.assertEqual(adapter._disc_step, 1)
            finally:
                adapter.close()


class MixedDensityTests(unittest.TestCase):
    def model(self, dimensions=1):
        return MixedActionDensity(2, dimensions, hidden_features=16, mixtures=4)

    def test_continuous_density_and_atom_masses_sum_to_one(self):
        model = self.model()
        grid = torch.linspace(-model.bound, model.bound, 20001)[:, None]
        observations = torch.zeros(len(grid), 2)
        with torch.no_grad():
            values, means, scales, weights, masses = model.parameters_for(grid, observations)
            continuous, _ = model._continuous_terms(values, means, scales, weights)
            density = (masses[..., 4] + continuous).exp()[:, 0]
            integral = torch.trapezoid(density, grid[:, 0])
            integral += masses[0, 0, :4].exp().sum()
        self.assertAlmostEqual(float(integral), 1.0, places=4)
        logp = model.log_prob(model.atoms[:, None], torch.zeros(4, 2))
        torch.testing.assert_close(logp, masses[0, 0, :4])

    def test_autoregressive_parameters_do_not_see_current_or_future_action(self):
        model = self.model(4)
        torch.nn.init.normal_(model.network.final_layer.weight, std=.1)
        first = torch.randn(2, 4) * .1
        second = first.clone()
        second[:, 1:] += .3
        context = torch.randn(2, 2)
        first_parameters = model.parameters_for(first, context)[1:]
        second_parameters = model.parameters_for(second, context)[1:]
        for left, right in zip(first_parameters, second_parameters):
            torch.testing.assert_close(left[:, :2], right[:, :2], rtol=0, atol=0)

    def test_randomized_pit_includes_atoms(self):
        torch.manual_seed(42)
        model = self.model()
        count = 10000
        with torch.no_grad():
            _, means, scales, weights, masses = model.parameters_for(
                torch.zeros(1, 1), torch.zeros(1, 2)
            )
            probabilities = masses[0, 0].exp()
            modes = torch.multinomial(probabilities, count, replacement=True)
            components = torch.multinomial(weights[0, 0].exp(), count, replacement=True)
            mean = means[0, 0, components]
            scale = scales[0, 0, components]
            lower = torch.special.ndtr((-model.bound - mean) / scale)
            upper = torch.special.ndtr((model.bound - mean) / scale)
            uniform = lower + torch.rand(count) * (upper - lower)
            actions = mean + scale * torch.special.ndtri(uniform.clamp(1e-7, 1 - 1e-7))
            atoms = modes < 4
            actions[atoms] = model.atoms[modes[atoms]]
            pit = model.randomized_pit(actions[:, None], torch.zeros(count, 2))
        self.assertLess(float(ks_uniform(pit).max()), .025)

    def test_first_action_coordinate_can_depend_on_observation(self):
        model = self.model(4)
        torch.nn.init.normal_(model.context_output.weight, std=.1)
        actions = torch.zeros(2, 4)
        observations = torch.tensor([[0., 0.], [1., 1.]])
        means = model.parameters_for(actions, observations)[1]
        self.assertFalse(torch.equal(means[0, 0], means[1, 0]))

    def test_calibrated_continuous_density_and_atoms_remain_normalized(self):
        torch.manual_seed(17)
        base = self.model()
        observations = torch.zeros(512, 2)
        development = (observations, torch.empty(512, 1).uniform_(-1, 1))
        model = CalibratedActionDensity(base, observations[:1], development, 512)
        with torch.no_grad():
            model.means.fill_(.3)
            model.scales.fill_(1.2)
            grid = torch.linspace(-base.bound, base.bound, 40001)[:, None]
            state = torch.zeros(len(grid), 2)
            terms, left, _ = base.distribution_terms(grid, state)
            mean, scale = model._parameters_for_states(state)
            _, derivative = model._transform(left, mean, scale)
            continuous_density = (terms + derivative).exp()[:, 0]
            # Grid points coinciding with atoms use masses, not a PDF value.
            atom_points = base.atom_matches(grid).any(-1).squeeze(-1)
            continuous_density[atom_points] = 0
            integral = torch.trapezoid(continuous_density, grid[:, 0])
            integral += model.log_prob(base.atoms[:, None], torch.zeros(4, 2)).exp().sum()
        self.assertAlmostEqual(float(integral), 1.0, places=3)

    def test_calibration_rejects_an_unregistered_context(self):
        base = self.model()
        states = torch.zeros(64, 2)
        model = CalibratedActionDensity(base, states[:1], (states, torch.zeros(64, 1)), 64)
        with self.assertRaisesRegex(ValueError, "registered observation"):
            model.log_prob(torch.zeros(1, 1), torch.ones(1, 2))

    def test_unvalidated_density_cannot_reach_airl(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = ValidatedActionDensity({}, directory)
            with self.assertRaises(DensityValidationError):
                gate.log_prob(torch.zeros(1, 1), torch.zeros(1, 2))
            gate.models = [self.model(), self.model()]
            gate.registered_states = torch.zeros(1, 2)
            gate.validated = True
            self.assertTrue(torch.isfinite(gate.log_prob(torch.zeros(1, 1), torch.zeros(1, 2))).all())
            with self.assertRaises(DensityValidationError):
                gate.log_prob(torch.zeros(1, 1), torch.ones(1, 2))
            gate.invalidate()
            with self.assertRaises(DensityValidationError):
                gate.log_prob(torch.zeros(1, 1), torch.zeros(1, 2))


if __name__ == "__main__":
    unittest.main()
