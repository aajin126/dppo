"""Check that training reset relocation preserves every non-position field."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent.finetune.lfirl.reset_bank import mix_reset_snapshots, relocate_snapshots


class ResetBankTests(unittest.TestCase):
    def test_mixed_starts_preserve_unselected_resets_and_copy_donor_controls(self):
        rng = np.random.default_rng(7)
        current = rng.normal(size=(8, 154)).astype(np.float32)
        bank = rng.normal(size=(4, 154)).astype(np.float32)
        before = current.copy()
        mixed = mix_reset_snapshots(current, bank, np.array([2, 6]), np.array([3, 1]), 9, 9)
        np.testing.assert_array_equal(current, before)
        np.testing.assert_array_equal(mixed[[0, 1, 3, 4, 5, 7]], current[[0, 1, 3, 4, 5, 7]])
        np.testing.assert_array_equal(mixed[[2, 6], :18], bank[[3, 1], :18])
        np.testing.assert_array_equal(mixed[[2, 6], 135:], bank[[3, 1], 135:])
        np.testing.assert_allclose(mixed[[2, 6], 18:21], current[[2, 6], 18:21], atol=1e-6)
        for selected, donor in ((2, 3), (6, 1)):
            for obj in range(9):
                start = 18 + 13 * obj
                np.testing.assert_allclose(mixed[selected, start:start + 3] - mixed[selected, 18:21],
                                           bank[donor, start:start + 3] - bank[donor, 18:21], atol=1e-6)
        with self.assertRaises(ValueError):
            mix_reset_snapshots(current, bank, np.array([2, 2]), np.array([0, 1]), 9, 9)

    def test_relocation_preserves_rigid_geometry_and_control_state(self):
        rng = np.random.default_rng(42)
        joint_count, objects = 9, 9
        original = rng.normal(size=(5, 154)).astype(np.float32)
        saved = original.copy()
        destinations = rng.normal(size=(5, 3)).astype(np.float32) * 20
        shifted = relocate_snapshots(original, destinations, joint_count, objects)
        np.testing.assert_array_equal(original, saved)
        np.testing.assert_allclose(shifted[:, 18:21], destinations, atol=2e-6)
        unchanged = np.ones(154, dtype=bool)
        for index in range(objects):
            offset = 18 + 13 * index
            unchanged[offset:offset + 3] = False
            np.testing.assert_allclose(shifted[:, offset:offset + 3] - shifted[:, 18:21],
                                       original[:, offset:offset + 3] - original[:, 18:21], atol=4e-6)
        np.testing.assert_array_equal(shifted[:, unchanged], original[:, unchanged])

    def test_incompatible_snapshot_and_nonfinite_inputs_fail(self):
        for values, positions in ((np.zeros((2, 153)), np.zeros((2, 3))),
                                  (np.zeros((2, 154)), np.zeros((3, 3))),
                                  (np.full((2, 154), np.nan), np.zeros((2, 3)))):
            with self.assertRaises(ValueError):
                relocate_snapshots(values, positions, 9, 9)


if __name__ == "__main__":
    unittest.main()
