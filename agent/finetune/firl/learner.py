"""Reuse original f-IRL networks and its non-importance-sampled covariance loss."""

from pathlib import Path
import sys

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[3] / "external/firl"
if not (SOURCE / "firl/divs/f_div_disc.py").is_file():
    raise FileNotFoundError(f"Required official f-IRL source: {SOURCE}")
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from firl.divs.f_div_disc import f_div_disc_loss
from firl.models.discrim import SMMIRLDisc
from firl.models.reward import MLPReward


class FIRLRewardLearner:
    """Explicit reward learning; state-density ratios never become policy log-probs."""

    def __init__(self, input_dim, device, settings):
        self.input_dim = int(input_dim)
        self.device = torch.device(device)
        self.settings = settings
        self.reward = MLPReward(
            input_dim,
            hidden_sizes=tuple(settings.reward.hidden_sizes),
            hid_act="tanh",
            use_bn=False,
            clamp_magnitude=float(settings.reward.clamp_magnitude),
            device=self.device,
        ).to(self.device)
        self.disc = SMMIRLDisc(
            input_dim,
            num_layer_blocks=3,
            hid_dim=int(settings.discriminator.hidden_size),
            hid_act="tanh",
            use_bn=False,
            batch_size=int(settings.discriminator.batch_size),
            lr=float(settings.discriminator.learning_rate),
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(
            self.reward.parameters(), lr=float(settings.reward.learning_rate),
            weight_decay=float(settings.reward.weight_decay), betas=(.9, .999),
        )
        self.reward_updates = 0
        self.discriminator_updates = 0

    def update(self, expert_states, policy_trajectories):
        expert = np.asarray(expert_states, dtype=np.float32)
        trajectories = np.asarray(policy_trajectories, dtype=np.float32)
        if trajectories.ndim != 3 or trajectories.shape[-1] != self.input_dim:
            raise ValueError("Expected policy trajectories [episodes, time, features]")
        if len(trajectories) < 2:
            raise ValueError("f-IRL covariance needs at least two policy trajectories")
        if expert.ndim != 2 or expert.shape[-1] != self.input_dim:
            raise ValueError("Expert and policy reward features differ")
        if not np.isfinite(expert).all() or not np.isfinite(trajectories).all():
            raise ValueError("Nonfinite f-IRL input states")
        disc_steps = int(self.settings.discriminator.updates)
        losses = self.disc.learn(
            expert, trajectories.reshape(-1, self.input_dim), iter=disc_steps
        )
        self.discriminator_updates += disc_steps
        reward_losses = []
        gradient_norms = []
        for _ in range(int(self.settings.reward.updates)):
            # IS=False is the official on-policy f-IRL path. No action
            # probabilities are supplied, estimated, or replaced by path scores.
            loss, _ = f_div_disc_loss(
                "js", False, (trajectories, None, None),
                self.disc, self.reward, self.device,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite upstream f-IRL reward loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            squared = sum(
                parameter.grad.square().sum()
                for parameter in self.reward.parameters()
                if parameter.grad is not None
            )
            gradient_norm = squared.sqrt()
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Nonfinite f-IRL reward gradient")
            self.optimizer.step()
            self.reward_updates += 1
            reward_losses.append(float(loss.detach()))
            gradient_norms.append(float(gradient_norm.detach()))
        return dict(
            discriminator_updates=self.discriminator_updates,
            reward_updates=self.reward_updates,
            discriminator_loss=float(np.mean(losses)),
            reward_loss=float(np.mean(reward_losses)),
            reward_gradient_norm=float(np.mean(gradient_norms)),
            policy_trajectories=len(trajectories),
            states_per_trajectory=trajectories.shape[1],
            expert_states=len(expert),
        )

    def predict(self, states):
        return self.reward.get_scalar_reward(np.asarray(states, dtype=np.float32))

    def state_dict(self):
        return dict(
            reward=self.reward.state_dict(),
            discriminator=self.disc.state_dict(),
            reward_optimizer=self.optimizer.state_dict(),
            discriminator_optimizer=self.disc.optimizer.state_dict(),
            reward_updates=self.reward_updates,
            discriminator_updates=self.discriminator_updates,
        )

    def load_state_dict(self, saved):
        self.reward.load_state_dict(saved["reward"], strict=True)
        self.disc.load_state_dict(saved["discriminator"], strict=True)
        self.optimizer.load_state_dict(saved["reward_optimizer"])
        self.disc.optimizer.load_state_dict(saved["discriminator_optimizer"])
        self.reward_updates = int(saved["reward_updates"])
        self.discriminator_updates = int(saved["discriminator_updates"])
