"""Expert state occupancy from the existing DPPO NPZ and sequence loader."""

import numpy as np

from agent.dataset.sequence import StitchedSequenceDataset


class FIRLExpertStates:
    def __init__(self, path, observation_dim, max_trajectories=0, seed=42):
        data = StitchedSequenceDataset(
            str(path), horizon_steps=1, cond_steps=1, device="cpu"
        )
        if data.states.shape[-1] != observation_dim:
            raise ValueError("Expert and environment observation dimensions differ")
        with np.load(path, allow_pickle=False) as source:
            lengths = source["traj_lengths"].astype(np.int64)
        valid = ~(
            data.states.eq(0).all(-1) & data.actions.eq(0).all(-1)
        ).numpy()
        boundaries = np.r_[0, lengths.cumsum()]
        episodes = [
            np.flatnonzero(valid[start:end]) + start
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        episode_ids = [index for index, rows in enumerate(episodes) if len(rows)]
        if not episode_ids:
            raise ValueError("No nonpadding expert states in the DPPO dataset")
        if max_trajectories > 0 and len(episode_ids) > max_trajectories:
            episode_ids = np.random.default_rng(seed).choice(
                episode_ids, max_trajectories, replace=False
            ).tolist()
        rows = np.concatenate([episodes[index] for index in episode_ids])
        self.states = data.states[rows].numpy().astype(np.float32, copy=True)
        self.episode_ids = episode_ids
        self.row_indices = rows
        if not np.isfinite(self.states).all():
            raise ValueError("Nonfinite expert observations")
