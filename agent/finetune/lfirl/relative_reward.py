"""Learned object-relative correction to a frozen LFIRL reward.

This is an explicit preference-learning extension. It uses no goal pose,
assembly threshold, native reward value, timer, or skill schedule as an input.
"""

import torch
from torch import nn


class RelativeRewardResidual(nn.Module):
    def __init__(self, base, normalizer_state):
        super().__init__()
        from env.gym_utils.furniture_normalizer import LinearNormalizer

        self.base = base
        self.base.requires_grad_(False)
        self.normalizer = LinearNormalizer()
        self.normalizer.load_state_dict(normalizer_state)
        if self.normalizer.stats["observations"]["min"].numel() != 58:
            raise ValueError("The one-leg relative reward requires original 58D observations")
        self.linear = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.linear.weight)
        self.register_buffer("feature_center", torch.zeros(12))
        self.register_buffer("residual_scale", torch.ones(()))

    def set_obs_normalizer(self, **kwargs):
        self.base.set_obs_normalizer(**kwargs)

    def features(self, states):
        from furniture_bench.controllers.control_utils import pose_from_vector

        if states.ndim != 3 or states.shape[1:] != (1, 58):
            raise ValueError("Expected original normalized one-step states [B, 1, 58]")
        # The policy min/max inverse is separate from LFIRL's optional internal
        # normalization. Apply it exactly once, before deriving physical poses.
        with torch.no_grad():
            physical = self.normalizer(states[:, 0], "observations", forward=False).float()
            top = pose_from_vector(physical[:, 16:23])
            leg = pose_from_vector(physical[:, 44:51])
            relative = torch.linalg.solve(top, leg)
            position = relative[:, :3, 3] * 10.0
            rotation = relative[:, :2, :3].reshape(-1, 6)
            return torch.cat((position, position.square(), rotation), dim=-1)

    def forward(self, states, actions):
        correction = self.linear(self.features(states) - self.feature_center).flatten()
        return self.base(states, actions) + self.residual_scale * correction
