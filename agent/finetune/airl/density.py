"""Conditional final-action density; never substitutes a denoising-path density.

The unchanged DDIM sampler has a mixed distribution: clipping the reconstructed
mean and the final Gaussian noise creates four possible atom locations in each
coordinate. An ordinary continuous normalizing flow would miss these masses.
An upstream nflows MADE supplies the autoregressive network. This adapter adds
the discrete masses and a bounded Gaussian mixture for the continuous part.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from agent.finetune.airl.sources import enable_sources

enable_sources()

from nflows.nn.nde.made import MADE


class MixedActionDensity(nn.Module):
    """Density/mass with respect to Lebesgue plus the known clipping atoms."""

    def __init__(
        self,
        observation_dim,
        action_features,
        mean_bound=1.0,
        noise_bound=3.0,
        final_std=0.04,
        hidden_features=512,
        mixtures=16,
        reverse=False,
    ):
        super().__init__()
        self.features = action_features
        self.mixtures = mixtures
        self.output_width = 3 * mixtures + 5
        self.bound = float(mean_bound + noise_bound * final_std)
        inner = float(mean_bound - noise_bound * final_std)
        if not 0 < inner < self.bound:
            raise ValueError("Expected distinct DDIM clipping atoms")
        self.register_buffer(
            "atoms", torch.tensor([-self.bound, -inner, inner, self.bound])
        )
        permutation = torch.arange(action_features)
        self.register_buffer("order", permutation.flip(0) if reverse else permutation)
        self.network = MADE(
            features=action_features,
            hidden_features=hidden_features,
            context_features=observation_dim,
            output_multiplier=self.output_width,
            num_blocks=2,
            use_batch_norm=False,
            dropout_probability=0.0,
        )
        # The upstream MADE's first output mask has no input-dependent hidden
        # units when features > 1. A context-only projection makes every output,
        # including the first coordinate, conditional on the observation.
        self.context_output = nn.Linear(
            hidden_features, action_features * self.output_width, bias=False
        )
        nn.init.zeros_(self.context_output.weight)
        with torch.no_grad():
            self.network.final_layer.weight.zero_()
            bias = self.network.final_layer.bias.reshape(action_features, -1)
            bias.zero_()
            grid = torch.linspace(-mean_bound, mean_bound, mixtures)
            bias[:, mixtures:2 * mixtures] = torch.atanh(grid / self.bound)
            bias[:, 2 * mixtures:3 * mixtures] = math.log(math.expm1(0.075))
            bias[:, 3 * mixtures:3 * mixtures + 4] = -7.0

    def parameters_for(self, actions, observations):
        actions = actions.flatten(1)[:, self.order]
        context = observations.flatten(1)
        output = self.network(actions, context)
        output += self.context_output(F.relu(self.network.context_layer(context)))
        output = output.reshape(len(actions), self.features, self.output_width)
        logits, raw_mean, raw_scale = output[..., :3 * self.mixtures].split(
            self.mixtures, dim=-1
        )
        means = self.bound * raw_mean.tanh()
        scales = (F.softplus(raw_scale) + 0.005).clamp_max(2 * self.bound)
        log_weights = logits.log_softmax(-1)
        log_masses = output[..., 3 * self.mixtures:].log_softmax(-1)
        return actions, means, scales, log_weights, log_masses

    def _continuous_terms(self, actions, means, scales, log_weights):
        low = (-self.bound - means) / scales
        high = (self.bound - means) / scales
        cdf_low = torch.special.ndtr(low)
        normalizer = (torch.special.ndtr(high) - cdf_low).clamp_min(1e-12)
        z = (actions[..., None] - means) / scales
        log_pdf = -0.5 * z.square() - scales.log() - 0.5 * math.log(2 * math.pi)
        log_pdf -= normalizer.log()
        log_pdf = torch.logsumexp(log_weights + log_pdf, dim=-1)
        cdf = (torch.special.ndtr(z) - cdf_low) / normalizer
        cdf = (log_weights.exp() * cdf.clamp(0, 1)).sum(-1)
        return log_pdf, cdf

    def atom_matches(self, actions):
        # The existing sampler operates in float32. Only rounding-level equality
        # is accepted; values near an atom are not assigned its probability mass.
        tolerance = 2 * torch.finfo(actions.dtype).eps
        return (actions[..., None] - self.atoms).abs() <= tolerance

    def log_prob(self, actions, observations):
        terms, _, _ = self.distribution_terms(actions, observations)
        return terms.sum(-1)

    def distribution_terms(self, actions, observations):
        values, means, scales, weights, masses = self.parameters_for(actions, observations)
        if (values.abs() > self.bound + 1e-6).any():
            raise ValueError("Action is outside the unchanged DDIM sampler's support")
        continuous, continuous_cdf = self._continuous_terms(values, means, scales, weights)
        matches = self.atom_matches(values)
        atom_log_prob = torch.where(
            matches, masses[..., :4], torch.full_like(masses[..., :4], -torch.inf)
        ).logsumexp(-1)
        terms = torch.where(matches.any(-1), atom_log_prob, masses[..., 4] + continuous)
        probabilities = masses.exp()
        less = self.atoms < values[..., None] - 2 * torch.finfo(values.dtype).eps
        left = probabilities[..., 4] * continuous_cdf
        left += (probabilities[..., :4] * less).sum(-1)
        jumps = (probabilities[..., :4] * matches).sum(-1)
        return terms, left, jumps

    @torch.no_grad()
    def randomized_pit(self, actions, observations, generator=None):
        """Randomized Rosenblatt transform, including the discrete atom masses."""
        _, left, jumps = self.distribution_terms(actions, observations)
        uniform = torch.rand(
            left.shape, device=left.device, dtype=left.dtype, generator=generator
        )
        return (left + uniform * jumps).clamp(0, 1)


class CalibratedActionDensity(nn.Module):
    """A monotone CDF calibration fitted on development samples, never test samples.

    G(F) is a proper CDF. Continuous density uses p*G'(F), while atoms use
    G(F_right)-G(F_left), so their probability masses remain well-defined.
    The context table restricts qualification to explicitly sampled states.
    """

    def __init__(self, base, states, development, draws):
        super().__init__()
        self.base = base
        self.features = base.features
        self.register_buffer("states", states.detach().clone())
        with torch.no_grad():
            pits = torch.cat([
                base.randomized_pit(
                    development[1][start:start + 512],
                    development[0][start:start + 512],
                )
                for start in range(0, len(development[0]), 512)
            ])
            normal = torch.special.ndtri(pits.clamp(1e-6, 1 - 1e-6))
            normal = normal.reshape(len(states), draws, base.features)
        self.register_buffer("means", normal.mean(1).clamp(-1, 1))
        self.register_buffer("scales", normal.std(1).clamp(.5, 2))

    def _parameters_for_states(self, states):
        distances = (states[:, None] - self.states[None]).abs().amax(-1)
        values, ids = distances.min(1)
        if (values > 1e-6).any():
            raise ValueError("CDF calibration requires a registered observation")
        return self.means[ids], self.scales[ids]

    def _transform(self, uniform, mean, scale):
        uniform = uniform.to(torch.float64)
        normal = torch.special.ndtri(uniform.clamp(1e-10, 1 - 1e-10))
        standardized = (normal - mean) / scale
        cdf = torch.special.ndtr(standardized)
        cdf = torch.where(uniform <= 0, torch.zeros_like(cdf), cdf)
        cdf = torch.where(uniform >= 1, torch.ones_like(cdf), cdf)
        log_derivative = .5 * (normal.square() - standardized.square()) - scale.log()
        return cdf, log_derivative

    def log_prob(self, actions, observations):
        terms, left, jumps = self.base.distribution_terms(actions, observations)
        mean, scale = self._parameters_for_states(observations.flatten(1))
        calibrated_left, derivative = self._transform(left, mean, scale)
        calibrated_right, _ = self._transform(left + jumps, mean, scale)
        atom_logp = (calibrated_right - calibrated_left).clamp_min(1e-30).log()
        return torch.where(jumps > 0, atom_logp, terms + derivative).sum(-1).float()

    @torch.no_grad()
    def randomized_pit(self, actions, observations):
        _, left, jumps = self.base.distribution_terms(actions, observations)
        mean, scale = self._parameters_for_states(observations.flatten(1))
        calibrated_left, _ = self._transform(left, mean, scale)
        calibrated_right, _ = self._transform(left + jumps, mean, scale)
        uniform = torch.rand_like(calibrated_left)
        return (calibrated_left + uniform * (calibrated_right - calibrated_left)).float()


def sampler_specification(policy):
    """Verify the support of the actual, unmodified stochastic DDIM sampler."""
    if not policy.use_ddim or policy.final_action_clip_value is not None:
        raise ValueError("This density adapter requires DDIM without final-action clipping")
    if policy.denoised_clip_value is None or not math.isfinite(policy.randn_clip_value):
        raise ValueError("Expected the existing clipped-mean/clipped-noise DDIM sampler")
    return dict(
        observation_dim=policy.obs_dim,
        action_features=policy.horizon_steps * policy.action_dim,
        mean_bound=float(policy.denoised_clip_value),
        noise_bound=float(policy.randn_clip_value),
        final_std=float(policy.get_min_sampling_denoising_std()),
    )
