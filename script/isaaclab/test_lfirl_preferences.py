"""Check native labels and the reused preference-training interface."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.finetune.lfirl.preferences import (
    AnchoredPreferenceLoss, PreferenceRewardAdapter, load_outcome_pairs,
    load_curriculum_pairs, select_trainable_reward_layers,
)
from imitation.algorithms.preference_comparisons import BasicRewardTrainer, PreferenceDataset, PreferenceModel
from imitation.data.types import Trajectory
from imitation.util import logger


class TinyReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(5, 8)
        self.mlp = nn.Sequential(nn.Linear(8, 1))

    def forward(self, states, actions):
        return self.mlp(torch.tanh(self.body(torch.cat((states[:, 0], actions), -1)))).flatten()


class PreferenceTests(unittest.TestCase):
    def test_relative_parent_head_selection_freezes_correction(self):
        reward = nn.Module()
        reward.base = TinyReward()
        reward.linear = nn.Linear(12, 1)
        self.assertEqual(select_trainable_reward_layers(reward), ("base.mlp.0.",))
        self.assertFalse(any(parameter.requires_grad for parameter in reward.linear.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in reward.base.body.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in reward.base.mlp.parameters()))
        self.assertEqual(select_trainable_reward_layers(reward, relative_only=True), ("linear.",))
        self.assertTrue(all(parameter.requires_grad for parameter in reward.linear.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in reward.base.parameters()))
        with self.assertRaises(ValueError):
            select_trainable_reward_layers(reward, 2, relative_only=True)

    def test_curriculum_pairs_require_real_outcomes_matches_and_episode_splits(self):
        from agent.finetune.lfirl.reward import sha256

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "states.npz", physical=np.zeros((2, 35), np.float32))
            donor = dict(accepted=True, seed=7, env_id=3)
            (root / "report.json").write_text(json.dumps(dict(
                completed=True, mode="near_completion", bank_sha256=sha256(root / "states.npz"),
                rows=[donor, donor], joint_count=2, root_objects=["robot", "part"],
            )))
            events = np.zeros((5, 4), np.float32)
            events[2, [0, 2]] = 1.
            physical = np.zeros((4, 35), np.float32)

            def write_archive():
                np.savez(root / "rollout.npz", observations=np.zeros((5, 4, 1, 58), np.float32),
                         actions=np.zeros((5, 4, 8, 10), np.float32), assembly_rewards=events,
                         initial_physics=physical, donor_ids=np.array([0, 0, 1, 1]))

            write_archive()
            entry = dict(trajectory=str(root / "rollout.npz"), successful_environment=0,
                         failed_environment=1, split="training")
            specification = dict(bank=str(root), pairs=[entry])
            manifest = root / "pairs.json"

            def write_manifest():
                manifest.write_text(json.dumps(specification))

            write_manifest()
            loaded = load_curriculum_pairs(manifest)
            self.assertEqual(loaded[0][0]["source_episode"], "7:3")
            self.assertEqual(len(loaded[0][1][0]), 3)
            self.assertTrue(loaded[0][1][0].terminal)
            self.assertFalse(loaded[0][1][1].terminal)
            specification["pairs"].append(dict(entry, successful_environment=2,
                                                failed_environment=3, split="validation"))
            write_manifest()
            with self.assertRaisesRegex(ValueError, "leaked"):
                load_curriculum_pairs(manifest)
            specification["pairs"] = [entry]
            write_manifest()
            physical[1, 0] = .01
            write_archive()
            with self.assertRaisesRegex(ValueError, "different physical"):
                load_curriculum_pairs(manifest)
            physical[:] = 0.
            events[2, 1] = 1.
            write_archive()
            with self.assertRaisesRegex(ValueError, "label disagrees"):
                load_curriculum_pairs(manifest)

    def test_layer_selection_preserves_frozen_features_and_validates_scope(self):
        reward = TinyReward()
        reward.mlp = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1))
        prefixes = select_trainable_reward_layers(reward)
        self.assertEqual(prefixes, ("mlp.2.",))
        self.assertFalse(any(parameter.requires_grad for parameter in reward.body.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in reward.mlp[0].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in reward.mlp[2].parameters()))
        self.assertEqual(select_trainable_reward_layers(reward, 2), ("mlp.0.", "mlp.2."))
        for count in (0, 3):
            with self.assertRaises(ValueError):
                select_trainable_reward_layers(reward, count)

    def test_adapter_preserves_raw_reward_and_never_uses_labels_as_inputs(self):
        reward = TinyReward()
        states, actions = torch.randn(7, 3), torch.randn(7, 2)
        adapter = PreferenceRewardAdapter(reward, 3, 2, 5.)
        expected = reward(states[:, None], actions) / 5
        torch.testing.assert_close(adapter(states, actions, states * 9, torch.ones(7)), expected)
        torch.testing.assert_close(adapter(states, actions, states * -9, torch.zeros(7)), expected)

    def test_original_trainer_increases_correct_preference_with_frozen_features(self):
        torch.manual_seed(3)
        reward = TinyReward()
        with torch.no_grad():
            reward.mlp[-1].weight.zero_()
            reward.mlp[-1].bias.zero_()
        reward.requires_grad_(False)
        reward.mlp[-1].requires_grad_(True)
        body = {key: value.clone() for key, value in reward.body.state_dict().items()}
        first = Trajectory(np.ones((4, 3), np.float32), np.zeros((3, 2), np.float32), None, True)
        second = Trajectory(-np.ones((4, 3), np.float32), np.zeros((3, 2), np.float32), None, False)
        pairs = [(first, second)]
        model = PreferenceModel(PreferenceRewardAdapter(reward, 3, 2, 1.))
        states = torch.tensor(np.concatenate((first.obs[:-1], second.obs[:-1])))[:, None]
        actions = torch.zeros(6, 2)
        loss = AnchoredPreferenceLoss(states, actions, torch.zeros(6), 1.)
        before = float(model(pairs)[0].detach())
        dataset = PreferenceDataset()
        dataset.push(pairs, np.ones(1, np.float32))
        with tempfile.TemporaryDirectory() as directory:
            trainer = BasicRewardTrainer(model, loss, np.random.default_rng(3), batch_size=1,
                                         epochs=10, lr=.03, custom_logger=logger.configure(directory, []))
            trainer.train(dataset)
        self.assertGreater(float(model(pairs)[0].detach()), before + .01)
        for key, value in reward.body.state_dict().items():
            torch.testing.assert_close(value, body[key], rtol=0, atol=0)

    def test_pair_loader_rejects_bad_outcomes_unmatched_resets_and_unknown_next_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            states = np.arange(15, dtype=np.float32).reshape(1, 5, 3)
            actions = np.zeros((1, 5, 2, 2), dtype=np.float32)
            physical = np.zeros((1, 8), dtype=np.float32)
            inputs = np.zeros((1, 3), dtype=np.float32)

            def archive(name, events, offset=0.):
                np.savez(root / name, pre_states=states, actions=actions, assembly_rewards=events,
                         initial_physics=physical + offset, initial_states=inputs)

            archive("good.npz", [[0., 0., 1., 0., 0.]])
            archive("bad.npz", np.zeros((1, 5)))
            row = dict(seed=1, environment=0, start_chunk=0, end_chunk_exclusive=3,
                       success_trajectory=str(root / "good.npz"), failure_trajectory=str(root / "bad.npz"))
            manifest = root / "pairs.json"
            manifest.write_text(json.dumps(dict(pairs=[row])))
            pairs = load_outcome_pairs(manifest)
            self.assertEqual(len(pairs[0][1][0]), 3)
            np.testing.assert_array_equal(pairs[0][1][0].obs[-1], states[0, 3])
            archive("bad.npz", [[0., 0., 0., 1., 0.]])
            with self.assertRaisesRegex(ValueError, "label disagrees"):
                load_outcome_pairs(manifest)
            archive("bad.npz", np.zeros((1, 5)), offset=.001)
            with self.assertRaises(AssertionError):
                load_outcome_pairs(manifest)
            archive("bad.npz", np.zeros((1, 5)))
            row["end_chunk_exclusive"] = 5
            manifest.write_text(json.dumps(dict(pairs=[row])))
            with self.assertRaisesRegex(ValueError, "known actual next"):
                load_outcome_pairs(manifest)


if __name__ == "__main__":
    unittest.main()
