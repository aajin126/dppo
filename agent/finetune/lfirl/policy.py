"""Native DPPO BC as one 80-dimensional macro action for original LFIRL."""

from types import SimpleNamespace
from functools import partial

import torch
from torch import nn

from agent.finetune.lfirl import official


class MacroDenoiser(nn.Module):
    def __init__(self, bc):
        super().__init__()
        self.bc = bc

    def forward(self, action, time, local_cond=None, global_cond=None):
        batch = len(action)
        chunks = action.reshape(batch, self.bc.horizon_steps, self.bc.action_dim)
        condition = global_cond.reshape(batch, -1, self.bc.obs_dim)
        return self.bc.actor(chunks, time, cond={"state": condition}).reshape_as(action)


class NativeScheduler:
    def __init__(self, bc):
        self.bc = bc
        self.config = SimpleNamespace(num_train_timesteps=bc.denoising_steps)

    @property
    def alphas_cumprod(self):
        return self.bc.alphas_cumprod

    def add_noise(self, actions, noise, time):
        batch = len(actions)
        shape = (batch, self.bc.horizon_steps, self.bc.action_dim)
        return self.bc.q_sample(actions.reshape(shape), time, noise.reshape(shape)).reshape_as(actions)


class MacroDiffusionPolicy(nn.Module):
    """The original score routine operates on [B,1,80], not a tiled first action.

    Inputs already use the original DPPO normalization. Identity normalization
    here avoids a second normalization and gives scores in normalized units.
    """

    score_first_action = official.score_first_action

    def __init__(self, bc, observation_steps=1, score_noise_samples=1):
        super().__init__()
        if not isinstance(score_noise_samples, int) or score_noise_samples < 1:
            raise ValueError("Score noise sample count must be a positive integer")
        if not bc.predict_epsilon:
            raise ValueError("LFIRL adapter requires an epsilon-prediction BC checkpoint")
        self.model = MacroDenoiser(bc)
        self.noise_scheduler = NativeScheduler(bc)
        self.obs_as_global_cond = True
        self.obs_as_local_cond = False
        self.pred_action_steps_only = True
        self.obs_dim = bc.obs_dim
        self.n_obs_steps = observation_steps
        self.n_action_steps = 1
        self.action_dim = bc.horizon_steps * bc.action_dim
        self.score_noise_samples = score_noise_samples
        if score_noise_samples > 1:
            # Reuse the upstream routine's existing Monte Carlo option. This
            # only reduces target noise during Q fitting; sampling is unchanged.
            self.score_first_action = partial(
                self.score_first_action, n_noise=score_noise_samples,
            )
        self.eval()
        self.requires_grad_(False)

    def flatten_obs(self, obs):
        return obs

    def _normalize_obs(self, obs):
        return obs

    def _normalize_action(self, action):
        return action

    def _get_action_scale_for_score(self, device, dtype, action_dim):
        return torch.ones((1, 1, action_dim), device=device, dtype=dtype)


class NativeActionPolicy(MacroDiffusionPolicy):
    """Use LFIRL's upstream first-action score approximation unchanged.

    The upstream score helper tiles a single action across the prediction
    horizon. This is its small-noise approximation, not an exact marginal
    action density. The native BC network and sampler are unchanged.
    """

    def __init__(self, bc, observation_steps=1, score_noise_samples=1):
        super().__init__(bc, observation_steps, score_noise_samples)
        self.n_action_steps = bc.horizon_steps
        self.action_dim = bc.action_dim
