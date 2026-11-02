"""Locate physical failure stages without changing rewards or policy actions."""

import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "external/furniture-bench")]


def assembly_targets():
    import numpy as np
    from furniture_bench.utils.pose import get_mat

    source = ROOT / "external/furniture-bench/furniture_bench/furniture/square_table.py"
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List):
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                and target.value.attr == "assembled_rel_poses" and ast.literal_eval(target.slice) == (0, 1)):
            return np.stack([get_mat(*(ast.literal_eval(arg) for arg in call.args))
                             for call in node.value.elts]).astype(np.float32)
    raise ValueError("Original SquareTable assembly target definition not found")


def main():
    import numpy as np
    import torch
    import furniture_bench.controllers.control_utils as C
    from env.gym_utils.furniture_normalizer import LinearNormalizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", default=["bc", "lfirl_control", "final"])
    args = parser.parse_args()
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(torch.load(ROOT / "data/furniture/one_leg_low/normalization.pth",
                                          map_location="cpu", weights_only=True))
    targets = torch.from_numpy(assembly_targets())
    results, candidates = {}, []
    for label in args.policies:
        report = json.loads((args.rollouts / label / "report.json").read_text())
        if not report["completed"]:
            raise ValueError("Phase analysis requires complete physical rollouts")
        totals = dict(episodes=0, successes=0, sustained_held=0, held_near_80mm=0,
                      held_near_20mm=0, held_near_20mm_aligned=0, released_away_from_goal=0,
                      known_pose_steps=0, total_pose_steps=0)
        errors = []
        for rollout in report["rollouts"]:
            with np.load(rollout["trajectory"]) as raw:
                observations = raw["pre_states"].copy()
                events, commands = raw["assembly_rewards"].copy(), raw["actions"].copy()
                height, distance = raw["leg_height_gain_m"], raw["tcp_leg_distance_m"]
                width, relative = raw["gripper_width_m"], raw["tcp_leg_relative_position_m"]
                stable = np.ones_like(height, dtype=bool)
                stable[:, 1:] = np.linalg.norm(np.diff(relative, axis=1), axis=-1) < .01
                held_post = (height > .06) & (distance < .04) & (width > .015) & (width < .045) & stable
            shape = observations.shape[:2]
            physical = normalizer(torch.from_numpy(observations), "observations", forward=False)
            top = C.pose_from_vector(physical[..., 16:23].reshape(-1, 7))
            leg = C.pose_from_vector(physical[..., 44:51].reshape(-1, 7))
            relative_pose = torch.linalg.inv(top) @ leg
            error = (relative_pose[:, None, :3, 3] - targets[None, :, :3, 3]).abs().amax(-1).amin(-1).reshape(shape).numpy()
            aligned = C.is_similar_rot(relative_pose[:, :3, :3], torch.eye(3)[None], .94).reshape(shape).numpy()
            known = (np.abs(observations[..., np.r_[16:23, 44:51]]) < 4.999).all(-1)
            totals["known_pose_steps"] += int(known.sum())
            totals["total_pose_steps"] += int(known.size)
            before = events.cumsum(1) - events < 1
            held_pre = np.zeros(shape, dtype=bool)
            consecutive = np.zeros(shape[0], dtype=int)
            for time in range(shape[1] - 1):
                consecutive = np.where(held_post[:, time], consecutive + 1, 0)
                held_pre[:, time + 1] = consecutive >= 3
            held = held_pre & before & known
            success = events.sum(1) >= 1
            totals["episodes"] += len(success)
            totals["successes"] += int(success.sum())
            totals["sustained_held"] += int(held.any(1).sum())
            for name, threshold in (("held_near_80mm", .08), ("held_near_20mm", .02)):
                totals[name] += int((held & (error < threshold)).any(1).sum())
            totals["held_near_20mm_aligned"] += int((held & (error < .02) & aligned).any(1).sum())
            grip = normalizer(torch.from_numpy(commands), "actions", forward=False)[..., 9].numpy()
            memory, opened = np.full(len(success), -1.), np.zeros(shape, dtype=bool)
            for time in range(shape[1]):
                for command in grip[:, time].T:
                    change = (np.sign(command) != np.sign(memory)) & (np.abs(command) > .019)
                    memory[change] = command[change]
                opened[:, time] = memory < 0
            totals["released_away_from_goal"] += int((held & opened & (error > .02) & (events < 1)).any(1).sum())
            minimum = np.where(held, error, np.inf).min(1)
            errors.extend((minimum[np.isfinite(minimum)] * 1000).tolist())
            if label == "bc":
                eligible = held & (error > .006) & (error < .08)
                eligible[:, :5] = False
                eligible[:, -8:] = False
                for index in np.flatnonzero(eligible.any(1)):
                    times = np.flatnonzero(eligible[index])
                    time = int(times[np.argmin(error[index, times])])
                    candidates.append(dict(trajectory=rollout["trajectory"], seed=rollout["seed"],
                                           env_id=int(index), step=time,
                                           position_error_m=float(error[index, time]),
                                           orientation_aligned=bool(aligned[index, time]),
                                           source_episode_success=bool(success[index])))
        totals["median_best_held_position_error_mm"] = float(np.median(errors)) if errors else None
        results[label] = totals
    rng = np.random.default_rng(2026110301)
    positive = [row for row in candidates if row["source_episode_success"]]
    negative = [row for row in candidates if not row["source_episode_success"]]
    selected = positive[:8]
    selected += [negative[index] for index in rng.choice(len(negative), min(32 - len(selected), len(negative)), replace=False)]
    result = dict(policies=results, potential_curriculum_states=len(candidates), proposed_curriculum_rows=selected,
                  assembly_targets=targets.tolist(), policy_updated=False, rewards_changed=False,
                  caveat="Sampled geometry proxies at macro boundaries, not contact labels or a new success metric. Clipped part poses are excluded. Proposed reset states require physical prefix replay and revalidation before use.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(policies=results, potential_curriculum_states=len(candidates), selected=len(selected)), indent=2))


if __name__ == "__main__":
    main()
