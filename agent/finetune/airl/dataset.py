"""AIRL transitions built from the original DPPO sequence loader."""

import numpy as np
import torch

from agent.dataset.sequence import StitchedSequenceDataset


class AIRLExpertSequences:
    def __init__(self, path, horizon, observation_dim):
        self.data = StitchedSequenceDataset(
            str(path), horizon_steps=horizon, cond_steps=1, device="cpu"
        )
        if self.data.states.shape[-1] != observation_dim:
            raise ValueError("Expert and policy observation dimensions differ")
        with np.load(path, allow_pickle=False) as source:
            lengths = source["traj_lengths"]
        end_for_row = np.repeat(np.cumsum(lengths), lengths)
        valid = ~(
            self.data.states.eq(0).all(-1) & self.data.actions.eq(0).all(-1)
        ).numpy()
        invalid_prefix = np.r_[0, np.cumsum(~valid)]
        self.horizon = horizon
        self.indices = [
            (start, history)
            for start, history in self.data.indices
            if start + horizon < end_for_row[start]
            and invalid_prefix[start + horizon + 1] == invalid_prefix[start]
        ]
        if not self.indices:
            raise ValueError("No complete nonpadding expert state/action/next-state sequences")

    def sample(self, size, device):
        ids = torch.randint(len(self.indices), (size,)).tolist()
        starts = torch.tensor([self.indices[index][0] for index in ids])
        offsets = torch.arange(self.horizon)
        return dict(
            obs=self.data.states[starts].to(device),
            acts=self.data.actions[starts[:, None] + offsets].to(device),
            next_obs=self.data.states[starts + self.horizon].to(device),
            # The NPZ contains no terminal next-state metadata. Windows ending
            # outside the recorded episode are excluded rather than fabricated.
            dones=torch.zeros(size, device=device),
        )
