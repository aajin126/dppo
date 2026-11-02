"""Check relative reward frames, sign invariance, frozen base and serialization."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "external/furniture-bench")]

from agent.finetune.lfirl.relative_reward import RelativeRewardResidual
from agent.finetune.lfirl.reward import LFIRLReward
from furniture_bench.controllers.control_utils import mat2pose_batched, pose_from_vector


class BaseReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.3))

    def forward(self, states, actions):
        return self.weight * (states[:, 0].sum(-1) + actions.mean(-1))

    def set_obs_normalizer(self, **kwargs):
        self.normalization = kwargs


def fixture():
    normalizer = {
        "stats.observations.min": -torch.ones(58),
        "stats.observations.max": torch.ones(58),
        "stats.actions.min": -torch.ones(10),
        "stats.actions.max": torch.ones(10),
    }
    reward = RelativeRewardResidual(BaseReward(), normalizer)
    states = torch.zeros(2, 1, 58)
    states[:, 0, 16:23] = torch.tensor([.2, -.1, .3, 0., 0., 0., 1.])
    states[:, 0, 44:51] = torch.tensor([.25, -.08, .29, 0., 0., 2 ** -.5, 2 ** -.5])
    return reward, states, normalizer


class RelativeRewardTests(unittest.TestCase):
    def test_frozen_calibration_reset_rejects_before_mutating_any_network(self):
        reward = LFIRLReward.__new__(LFIRLReward)
        reward.settings = {"relative_residual": {"enabled": True}}
        reward.b, reward.v = object(), object()
        original_b, original_v = reward.b, reward.v
        with self.assertRaises(ValueError):
            reward.reset_calibration_heads()
        self.assertIs(reward.b, original_b)
        self.assertIs(reward.v, original_v)

    def test_zero_residual_preserves_base_and_uses_original_pose_units_once(self):
        reward, states, _ = fixture()
        actions = torch.randn(2, 80)
        torch.testing.assert_close(reward(states, actions), reward.base(states, actions), rtol=0, atol=0)
        features = reward.features(states)
        torch.testing.assert_close(features[:, :3], torch.tensor([[.5, .2, -.1]]).expand(2, -1), atol=1e-6, rtol=0)
        torch.testing.assert_close(features[:, 3:6], features[:, :3].square())
        torch.testing.assert_close(features[:, 6:], torch.tensor([[0., -1., 0., 1., 0., 0.]]).expand(2, -1), atol=1e-6, rtol=0)
        self.assertFalse(any(parameter.requires_grad for parameter in reward.base.parameters()))

    def test_features_ignore_quaternion_sign_and_common_world_transform(self):
        reward, states, _ = fixture()
        expected = reward.features(states)
        flipped = states.clone()
        flipped[:, 0, 19:23] *= -1
        flipped[:, 0, 47:51] *= -1
        torch.testing.assert_close(reward.features(flipped), expected, atol=1e-6, rtol=0)
        transform = pose_from_vector(torch.tensor([[.1, -.2, .05, 0., .2, 0., .96]]))
        changed = states.clone()
        for start in (16, 44):
            matrix = transform @ pose_from_vector(states[:, 0, start:start + 7])
            position, quaternion = mat2pose_batched(matrix)
            changed[:, 0, start:start + 7] = torch.cat((position, quaternion), dim=-1)
        torch.testing.assert_close(reward.features(changed), expected, atol=2e-5, rtol=0)

    def test_roundtrip_preserves_nonzero_residual_and_rejects_wrong_state_contract(self):
        reward, states, normalizer = fixture()
        with torch.no_grad():
            reward.linear.weight.copy_(torch.arange(12)[None] * .01)
            reward.feature_center.fill_(.2)
            reward.residual_scale.fill_(17.)
        restored = RelativeRewardResidual(BaseReward(), normalizer)
        restored.load_state_dict(reward.state_dict(), strict=True)
        actions = torch.ones(2, 80)
        torch.testing.assert_close(restored(states, actions), reward(states, actions), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            reward.features(states[:, 0])


if __name__ == "__main__":
    unittest.main()
