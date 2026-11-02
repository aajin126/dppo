"""Adapt native DPPO batches to the original imitation AIRL implementation."""

import numpy as np
import torch

from agent.finetune.airl.sources import enable_sources

enable_sources()

import gymnasium as gym
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util import logger as imitation_logger


class DPPOAIRLReward(AIRL):
    """Reuse upstream networks, logits, BCE, optimizer step and discriminator stats.

    DPPO already owns its simulator and policy optimizer, so this adapter does
    not initialize the SB3-specific generator/environment setup in AIRL.__init__.
    The inherited train_disc and logits_expert_is_high methods remain unchanged.
    """

    def __init__(self, observation_dim, action_shape, discount, device, logdir, batch_size):
        self._reward_net = BasicShapedRewardNet(
            gym.spaces.Box(-np.inf, np.inf, (observation_dim,), np.float32),
            gym.spaces.Box(-np.inf, np.inf, action_shape, np.float32),
            reward_hid_sizes=(256, 256),
            potential_hid_sizes=(256, 256),
            use_action=False,
            discount_factor=discount,
        ).to(device)
        self._disc_opt = torch.optim.Adam(self._reward_net.parameters(), lr=3e-4)
        self._logger = imitation_logger.configure(str(logdir), ["csv", "tensorboard"])
        self.demo_batch_size = batch_size
        self.demo_minibatch_size = batch_size
        self._disc_step = 0
        self._global_step = 0
        self._init_tensorboard = False

    def _make_disc_train_batches(self, *, expert_samples=None, gen_samples=None):
        if expert_samples is None or gen_samples is None:
            raise ValueError("Provide explicit expert and current-policy DPPO batches")
        if len(expert_samples["obs"]) != self.demo_batch_size:
            raise ValueError("Expert batch size differs from AIRL configuration")
        if len(gen_samples["obs"]) != self.demo_batch_size:
            raise ValueError("Policy batch size differs from AIRL configuration")
        joined = {
            name: torch.cat([expert_samples[name], gen_samples[name]])
            for name in ("obs", "acts", "next_obs", "dones", "logp")
        }
        if not torch.isfinite(joined["logp"]).all():
            raise ValueError("AIRL requires finite validated final-action log probabilities")
        yield dict(
            state=joined["obs"],
            action=joined["acts"],
            next_state=joined["next_obs"],
            done=joined["dones"],
            log_policy_act_prob=joined["logp"].detach(),
            labels_expert_is_one=torch.cat([
                torch.ones(self.demo_batch_size, device=joined["obs"].device),
                torch.zeros(self.demo_batch_size, device=joined["obs"].device),
            ]),
        )

    def close(self):
        self._logger.close()
