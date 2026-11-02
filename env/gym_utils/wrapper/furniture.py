"""
Environment wrapper for Furniture-Bench environments.

"""

import gym
import numpy as np
import torch
from collections import deque

from furniture_bench.isaaclab.control_utils_lab import proprioceptive_quat_to_6d_rotation
from ..furniture_normalizer import LinearNormalizer
from .multi_step import repeated_space

import logging

log = logging.getLogger(__name__)


def stack_last_n_obs_dict(all_obs, n_steps):
    """Apply padding"""
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    result = {
        key: torch.zeros(
            list(all_obs[-1][key].shape)[0:1]
            + [n_steps]
            + list(all_obs[-1][key].shape)[1:],
            dtype=all_obs[-1][key].dtype,
        ).to(
            all_obs[-1][key].device
        )  # add step dimension
        for key in all_obs[-1]
    }
    available = min(n_steps, len(all_obs))
    start_idx = n_steps - available
    for key in all_obs[-1]:
        result[key][:, start_idx:] = torch.concatenate(
            [obs[key][:, None] for obs in all_obs[-available:]], dim=1
        )  # add step dimension
        if n_steps > len(all_obs):
            # pad
            result[key][:, :start_idx] = result[key][:, start_idx:start_idx+1]
    return result


class FurnitureRLSimEnvMultiStepWrapper(gym.Wrapper):
    env: gym.Env

    def __init__(
        self,
        env: gym.Env,
        n_obs_steps=1,
        n_action_steps=1,
        max_episode_steps=None,
        sparse_reward=False,
        reset_within_step=False,
        pass_full_observations=False,
        normalization_path=None,
        prev_action=False,
    ):
        assert (
            not reset_within_step
        ), "reset_within_step must be False for furniture envs"
        assert (
            not pass_full_observations
        ), "pass_full_observations is not implemented yet"
        assert not prev_action, "prev_action is not implemented yet"

        super().__init__(env)
        self._single_action_space = env.action_space
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.record_action_observations = False
        if n_obs_steps < 1 or n_action_steps < 1:
            raise ValueError("Observation/action step counts must be positive.")
        self.pass_full_observations = pass_full_observations
        if hasattr(env, "policy_observation_dim"):
            self.action_space = gym.spaces.Box(
                -1., 1., (env.num_envs, n_action_steps, env.action_space.shape[-1]), dtype=np.float32)
            self.observation_space = gym.spaces.Dict({"state": gym.spaces.Box(
                -5., 5., (env.num_envs, n_obs_steps, env.policy_observation_dim), dtype=np.float32)})

        # Use the original reward function where the robot does not receive new reward after completing one part
        self.sparse_reward = sparse_reward

        # set up normalization
        self.normalize = normalization_path is not None
        self.normalizer = LinearNormalizer()
        if self.normalize:
            self.normalizer.load_state_dict(
                torch.load(normalization_path, map_location=self.device, weights_only=True)
            )
            log.info(f"Loaded normalization from {normalization_path}")
            if hasattr(env, "policy_observation_dim"):
                expected = (env.policy_observation_dim, env.action_space.shape[-1])
                actual = (self.normalizer.stats["observations"]["min"].numel(),
                          self.normalizer.stats["actions"]["min"].numel())
                if actual != expected:
                    raise ValueError(f"Normalizer dimensions {actual} do not match environment {expected}.")

    def reset(
        self,
        **kwargs,
    ):
        """Resets the environment."""
        obs = self.env.reset()
        self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        obs = stack_last_n_obs_dict(self.obs, self.n_obs_steps)
        nobs = self.process_obs(obs)
        self.best_reward = torch.zeros(self.env.num_envs).to(self.device)
        self.done = list()
        return {"state": nobs}

    def reset_arg(self, options_list=None):
        return self.reset()

    def reset_one_arg(self, env_ind=None, options=None):
        if env_ind is not None:
            env_ind = torch.tensor([env_ind], device=self.device)
        return self.reset()

    def step(self, action: np.ndarray):
        """
        Takes in a chunk of actions of length n_action_steps
        and steps the environment n_action_steps times
        and returns an aggregated observation, reward, and done signal
        """
        # action: (n_envs, n_action_steps, action_dim)
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        expected = (self.env.num_envs, self.n_action_steps, self.env.action_space.shape[-1])
        if tuple(action.shape) != expected or not torch.isfinite(action).all():
            raise ValueError(f"Expected finite action chunk {expected}, got {tuple(action.shape)}.")

        # Denormalize the action
        if self.normalize:
            action = self.normalizer(action, "actions", forward=False)

        # Step the environment n_action_steps times
        obs, sparse_reward, dense_reward, info = self._inner_step(action)
        if self.sparse_reward:
            reward = sparse_reward.clone().cpu().numpy()
        else:
            reward = dense_reward.clone().cpu().numpy()

        # Only mark the environment as done if it times out, ignore done from inner steps
        limit = self.max_episode_steps if self.max_episode_steps is not None else self.env.max_env_steps
        truncated = self.env.env_steps >= limit

        nobs: np.ndarray = self.process_obs(obs)
        if self.record_action_observations:
            info = dict(info)
            info["action_observations"] = self.process_obs(
                stack_last_n_obs_dict(self.obs, self.n_action_steps)
            )
        truncated: np.ndarray = truncated.reshape(-1).cpu().numpy()
        # terminated: np.ndarray = np.zeros_like(truncated, dtype=bool)

        # since we only assign reward at the timestep where one stage is finished, and reward does not accumulate, we consider the final step of the episode as terminal
        return {"state": nobs}, reward, truncated, truncated, info

    def _inner_step(self, action_chunk: torch.Tensor):
        dense_reward = torch.zeros(action_chunk.shape[0], device=action_chunk.device)
        sparse_reward = torch.zeros(action_chunk.shape[0], device=action_chunk.device)
        for i in range(self.n_action_steps):
            # The dimensions of the action_chunk are (num_envs, chunk_size, action_dim)
            obs, reward, done, info = self.env.step(action_chunk[:, i, :])
            self.obs.append(obs)

            # track raw reward
            sparse_reward += reward.squeeze()

            # track best reward --- reward nonzero only one part is assembled
            self.best_reward += reward.squeeze()

            # assign "permanent" rewards
            dense_reward += self.best_reward

        obs = stack_last_n_obs_dict(self.obs, self.n_obs_steps)
        return obs, sparse_reward, dense_reward, info

    def process_obs(self, obs: torch.Tensor) -> np.ndarray:
        # Convert the robot state to have 6D pose
        robot_state = obs["robot_state"]
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)

        parts_poses = obs["parts_poses"]
        obs = torch.cat([robot_state, parts_poses], dim=-1)

        nobs = self.normalizer(obs, "observations", forward=True) if self.normalize else obs
        nobs = torch.clamp(nobs, -5, 5)
        return nobs.cpu().numpy()  # (n_envs, n_obs_steps, obs_dim)
