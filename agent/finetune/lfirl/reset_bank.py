"""Validate and relocate real physical snapshots used only for training resets."""

import json
from pathlib import Path

import numpy as np


def relocate_snapshots(physical, robot_positions, joint_count, object_count):
    """Translate all roots together; preserve joint state, velocities and controls."""
    physical = np.asarray(physical, dtype=np.float32)
    robot_positions = np.asarray(robot_positions, dtype=np.float32)
    expected_width = 4 * joint_count + 13 * object_count + 1
    if (physical.ndim != 2 or physical.shape[1] != expected_width
            or robot_positions.shape != (len(physical), 3)
            or not np.isfinite(physical).all() or not np.isfinite(robot_positions).all()):
        raise ValueError("Reset snapshot schema or destination positions differ")
    result = physical.copy()
    start = 2 * joint_count
    translation = robot_positions - physical[:, start:start + 3]
    for index in range(object_count):
        offset = start + 13 * index
        result[:, offset:offset + 3] += translation
    return result


def mix_reset_snapshots(current, bank, environment_ids, donor_ids, joint_count, object_count):
    """Replace only selected starts; keep ordinary reset states byte-for-byte."""
    current, bank = np.asarray(current), np.asarray(bank)
    environment_ids, donor_ids = np.asarray(environment_ids), np.asarray(donor_ids)
    if (environment_ids.ndim != 1 or donor_ids.shape != environment_ids.shape
            or environment_ids.dtype.kind not in "iu" or donor_ids.dtype.kind not in "iu"
            or len(np.unique(environment_ids)) != len(environment_ids)
            or (environment_ids < 0).any() or (environment_ids >= len(current)).any()
            or (donor_ids < 0).any() or (donor_ids >= len(bank)).any()):
        raise ValueError("Invalid curriculum environment/donor selection")
    result = current.copy()
    start = 2 * joint_count
    result[environment_ids] = relocate_snapshots(
        bank[donor_ids], current[environment_ids, start:start + 3], joint_count, object_count,
    )
    return result


def load_reset_bank(directory, normalizer_hash, source_bc_hash, root_objects, joint_count):
    from agent.finetune.lfirl.reward import sha256

    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    if (not report.get("completed") or report.get("accepted", 0) < 16
            or report.get("normalizer_sha256") != normalizer_hash
            or report.get("source_bc_sha256") != source_bc_hash
            or report.get("bank_sha256") != sha256(directory / "states.npz")
            or report.get("root_objects") != root_objects
            or report.get("joint_count") != joint_count):
        raise ValueError("Reset bank provenance or physical scene schema differs")
    with np.load(directory / "states.npz", allow_pickle=False) as raw:
        physical, observations = raw["physical"].copy(), raw["observations"].copy()
    if len(physical) != report["accepted"] or observations.shape != (len(physical), 58):
        raise ValueError("Reset bank observation/count contract differs")
    # Also check schema, finite values, and the first root coordinate layout.
    relocate_snapshots(physical, physical[:, 2 * joint_count:2 * joint_count + 3],
                       joint_count, len(root_objects))
    return physical, report
