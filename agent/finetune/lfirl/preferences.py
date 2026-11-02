"""Native outcome preferences for the existing LFIRL reward network.

This is an explicit extension of LFIRL, using imitation's original preference
model, cross-entropy loss and reward trainer. No policy network is introduced.
"""

import json
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from imitation.algorithms.preference_comparisons import CrossEntropyRewardLoss, LossAndMetrics
from imitation.data.types import Trajectory
from imitation.rewards.reward_nets import RewardNet


def select_trainable_reward_layers(reward, count=1, relative_only=False):
    """Fine-tune existing trailing Linear layers; keep all other features fixed."""
    if relative_only:
        if count != 1 or not hasattr(reward, "base") or not isinstance(reward.linear, torch.nn.Linear):
            raise ValueError("Relative-only calibration requires the existing relative linear head")
        reward.requires_grad_(False)
        reward.linear.requires_grad_(True)
        return ("linear.",)
    target = reward.base if hasattr(reward, "base") else reward
    prefix = "base." if target is not reward else ""
    layers = [(name, layer) for name, layer in target.mlp.named_children()
              if isinstance(layer, torch.nn.Linear)]
    if not 1 <= count <= len(layers):
        raise ValueError("Trainable reward layer count is outside the existing MLP")
    reward.requires_grad_(False)
    selected = layers[-count:]
    for _, layer in selected:
        layer.requires_grad_(True)
    return tuple(f"{prefix}mlp.{name}." for name, _ in selected)


def load_curriculum_pairs(manifest):
    """Read real same-donor branches with source-episode-separated splits.

    Outcome labels select trajectory pairs; they are never reward inputs.
    Different simulator clones introduce float32 origin translation error, so
    physical and normalized-observation discrepancies are bounded and recorded.
    """
    from agent.finetune.lfirl.reset_bank import relocate_snapshots
    from agent.finetune.lfirl.reward import sha256

    specification = json.loads(Path(manifest).read_text())
    bank = Path(specification["bank"])
    report = json.loads((bank / "report.json").read_text())
    if (not report.get("completed") or report["mode"] != "near_completion"
            or report["bank_sha256"] != sha256(bank / "states.npz")):
        raise ValueError("Require a verified real pre-completion bank")
    donors = [row for row in report["rows"] if row["accepted"]]
    archives, pairs, groups = {}, [], {}
    seen = set()
    for entry in specification["pairs"]:
        path = entry["trajectory"]
        if path not in archives:
            if entry.get("trajectory_sha256") and sha256(path) != entry["trajectory_sha256"]:
                raise ValueError("Curriculum trajectory changed after pair selection")
            with np.load(path, allow_pickle=False) as raw:
                archives[path] = {key: raw[key].copy() for key in (
                    "observations", "actions", "assembly_rewards", "initial_physics", "donor_ids",
                )}
        data = archives[path]
        good, bad = int(entry["successful_environment"]), int(entry["failed_environment"])
        identity = (path, good, bad)
        if identity in seen:
            raise ValueError("Duplicate curriculum preference pair")
        seen.add(identity)
        donor = int(data["donor_ids"][good])
        if donor < 0 or int(data["donor_ids"][bad]) != donor:
            raise ValueError("Preference branches must share the same real reset donor")
        source = donors[donor]
        group = f"{source['seed']}:{source['env_id']}"
        split = entry["split"]
        if split not in ("training", "validation") or groups.setdefault(group, split) != split:
            raise ValueError("Source episode leaked across preference splits")
        if data["assembly_rewards"][:, good].sum() < 1 or data["assembly_rewards"][:, bad].sum() >= 1:
            raise ValueError("Preference label disagrees with the native assembly event")
        end = int(np.flatnonzero(data["assembly_rewards"][:, good] > 0)[0]) + 1
        if end >= len(data["observations"]):
            raise ValueError("Preference windows require a known actual next observation")
        joint_count, object_count = report["joint_count"], len(report["root_objects"])
        physical = relocate_snapshots(data["initial_physics"][[good, bad]], np.zeros((2, 3)),
                                      joint_count, object_count)
        physical_error = float(np.abs(physical[0] - physical[1]).max())
        observation_error = float(np.abs(data["observations"][0, good] - data["observations"][0, bad]).max())
        if physical_error > 2e-5 or observation_error > 1e-3:
            raise ValueError("Same-donor branches have different physical states or policy inputs")
        fragments = tuple(Trajectory(
            obs=data["observations"][:end + 1, index, 0].astype(np.float32),
            acts=data["actions"][:end, index].reshape(end, -1).astype(np.float32),
            infos=None, terminal=successful,
        ) for index, successful in ((good, True), (bad, False)))
        metadata = dict(entry, source_episode=group, end_chunk_exclusive=end,
                        maximum_relative_physical_difference=physical_error,
                        maximum_initial_observation_difference=observation_error)
        pairs.append((metadata, fragments))
    return pairs


class PreferenceRewardAdapter(RewardNet):
    def __init__(self, reward, observation_dim, action_dim, temperature):
        super().__init__(spaces.Box(-np.inf, np.inf, (observation_dim,), dtype=np.float32),
                         spaces.Box(-1., 1., (action_dim,), dtype=np.float32), normalize_images=False)
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Preference temperature must be finite and positive")
        self.reward = reward
        self.temperature = float(temperature)

    def forward(self, state, action, next_state, done):
        return self.reward(state[:, None], action) / self.temperature


class AnchoredPreferenceLoss(CrossEntropyRewardLoss):
    def __init__(self, states, actions, targets, scale, coefficient=.1):
        super().__init__()
        self.states, self.actions, self.targets = states, actions, targets
        self.scale, self.coefficient = float(scale), float(coefficient)

    def forward(self, fragment_pairs, preferences, preference_model):
        result = super().forward(fragment_pairs, preferences, preference_model)
        adapter = preference_model.model
        prediction = adapter.reward(self.states, self.actions)
        anchor = ((prediction - self.targets) / self.scale).square().mean()
        return LossAndMetrics(result.loss + self.coefficient * anchor.to(result.loss.device),
                              {**result.metrics, "anchor_mse": anchor.detach().cpu()})


def load_outcome_pairs(manifest):
    """Keep equal-duration windows, exact reset matches, and whole-seed splits."""
    rows = json.loads(Path(manifest).read_text())["pairs"]
    archives, pairs = {}, []
    seen = set()
    for row in rows:
        identity = (int(row["seed"]), int(row["environment"]))
        if identity in seen:
            raise ValueError("Duplicate initial scene in preference data")
        seen.add(identity)
        fragments, starts = [], []
        for key, successful in (("success_trajectory", True), ("failure_trajectory", False)):
            path = row[key]
            if path not in archives:
                with np.load(path, allow_pickle=False) as raw:
                    archives[path] = {name: raw[name].copy() for name in
                                      ("pre_states", "actions", "assembly_rewards", "initial_physics", "initial_states")}
            data = archives[path]
            index, start, end = row["environment"], row["start_chunk"], row["end_chunk_exclusive"]
            events = data["assembly_rewards"][index]
            if bool(events.sum() >= 1) != successful:
                raise ValueError("Preference label disagrees with the native assembly event")
            if not 0 <= start < end < data["pre_states"].shape[1]:
                raise ValueError("Preference windows require a known actual next observation")
            if successful and int(np.flatnonzero(events.cumsum() >= 1)[0]) != end - 1:
                raise ValueError("Successful window must end at first assembly")
            fragments.append(Trajectory(
                obs=data["pre_states"][index, start:end + 1].astype(np.float32),
                acts=data["actions"][index, start:end].reshape(end - start, -1).astype(np.float32),
                infos=None, terminal=successful,
            ))
            starts.append((data["initial_physics"][index], data["initial_states"][index]))
        for left, right in zip(starts[0], starts[1]):
            np.testing.assert_array_equal(left, right)
        pairs.append((row, tuple(fragments)))
    return pairs
