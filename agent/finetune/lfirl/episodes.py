"""Optional absorbing assembly boundaries for fixed-horizon Furniture batches."""

import numpy as np


def finite_episode_returns(rewards, valid, gamma):
    """Discount observed rewards, with zero value beyond the episode budget."""
    rewards, valid = np.asarray(rewards), np.asarray(valid)
    if (rewards.ndim != 2 or rewards.shape != valid.shape or valid.dtype != bool
            or not np.isfinite(rewards).all() or not 0 <= gamma <= 1):
        raise ValueError("Expected aligned finite rewards, boolean validity, and a discount in [0, 1]")
    result = np.zeros_like(rewards, dtype=np.float64)
    following = np.zeros(rewards.shape[1], dtype=np.float64)
    for time in reversed(range(len(rewards))):
        following = np.where(valid[time], rewards[time] + gamma * following, 0.)
        result[time] = following
    return result


def assembly_boundaries(environment_rewards, threshold):
    """Keep the completion chunk; discard subsequent chunks without bonuses."""
    rewards = np.asarray(environment_rewards)
    if rewards.ndim != 2 or not np.isfinite(rewards).all() or (rewards < 0).any():
        raise ValueError("Expected finite nonnegative native assembly events (time, environments)")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Assembly success threshold must be positive")
    cumulative = rewards.cumsum(axis=0)
    completed = cumulative >= threshold
    before = np.concatenate((np.zeros_like(completed[:1]), completed[:-1]))
    return ~before, completed & ~before
