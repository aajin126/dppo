"""Bounded successful-episode memory for auxiliary diffusion BC, never PPO."""

import json
from pathlib import Path

import numpy as np
import torch


class SuccessfulRolloutBuffer:
    def __init__(self, directory, capacity, observation_shape, action_shape, task, normalizer_hash):
        if capacity < 1:
            raise ValueError("Success replay capacity must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        self.schema = dict(
            observation_shape=list(observation_shape), action_shape=list(action_shape),
            task=str(task), normalization_sha256=normalizer_hash,
        )
        self.episodes = []
        self.next_id = 0

    def __len__(self):
        return len(self.episodes)

    def add(self, observations, actions, metadata):
        observations = np.asarray(observations, dtype=np.float32).copy()
        actions = np.asarray(actions, dtype=np.float32).copy()
        self._validate(observations, actions)
        episode = dict(
            observations=observations, actions=actions,
            metadata=dict(metadata, episode_id=self.next_id),
        )
        self.next_id += 1
        self.episodes.append(episode)
        self.episodes = self.episodes[-self.capacity:]
        self._write_episode(episode)
        self._write_index()

    def _validate(self, observations, actions):
        if not len(observations) or len(observations) != len(actions):
            raise ValueError("Success replay requires aligned nonempty observations/actions")
        if list(observations.shape[1:]) != self.schema["observation_shape"]:
            raise ValueError("Success replay observation dimensions differ")
        if list(actions.shape[1:]) != self.schema["action_shape"]:
            raise ValueError("Success replay action dimensions differ")
        if not np.isfinite(observations).all() or not np.isfinite(actions).all():
            raise ValueError("Nonfinite success replay data")

    def sample(self, batch_size, device):
        if not self.episodes:
            raise ValueError("Success replay is empty")
        # Sample episodes uniformly so longer successes do not dominate.
        ids = np.random.randint(len(self.episodes), size=batch_size)
        rows = [
            (self.episodes[index], np.random.randint(len(self.episodes[index]["actions"])))
            for index in ids
        ]
        observations = np.stack([episode["observations"][step] for episode, step in rows])
        actions = np.stack([episode["actions"][step] for episode, step in rows])
        return (
            torch.as_tensor(observations, device=device),
            torch.as_tensor(actions, device=device),
        )

    def _write_episode(self, episode):
        path = self.directory / f"episode_{episode['metadata']['episode_id']:06d}.npz"
        np.savez_compressed(
            path, observations=episode["observations"], actions=episode["actions"],
            metadata_json=json.dumps(episode["metadata"]),
        )

    def _write_index(self):
        index = dict(
            schema=self.schema, capacity=self.capacity, next_id=self.next_id,
            active_episodes=[episode["metadata"] for episode in self.episodes],
        )
        (self.directory / "index.json").write_text(json.dumps(index, indent=2) + "\n")

    def state_dict(self):
        return dict(
            schema=self.schema, next_id=self.next_id,
            episodes=[dict(
                observations=torch.from_numpy(episode["observations"]),
                actions=torch.from_numpy(episode["actions"]),
                metadata=episode["metadata"],
            ) for episode in self.episodes],
        )

    def load_state_dict(self, saved):
        if saved["schema"] != self.schema:
            raise ValueError("Success replay task, dimensions or normalization differ")
        restored = []
        for episode in saved["episodes"][-self.capacity:]:
            observations = episode["observations"].detach().cpu().numpy().copy()
            actions = episode["actions"].detach().cpu().numpy().copy()
            self._validate(observations, actions)
            restored.append(dict(observations=observations, actions=actions, metadata=episode["metadata"]))
        self.episodes = restored
        self.next_id = int(saved["next_id"])
        for episode in self.episodes:
            self._write_episode(episode)
        self._write_index()
