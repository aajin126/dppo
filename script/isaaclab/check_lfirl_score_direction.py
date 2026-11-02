"""Check recovered Q against frozen BC scores on recorded policy states.

No policy or reward is trained. The score integral is a small-noise numerical
reference, not an exact DDIM action log probability.
"""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import hydra
    import numpy as np
    import torch
    from agent.finetune.lfirl.policy import MacroDiffusionPolicy
    from agent.finetune.lfirl.reward import LFIRLReward, sha256
    from env.gym_utils.furniture_normalizer import LinearNormalizer
    from script.isaaclab.run_lab import compose_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.environ.setdefault("DPPO_DATA_DIR", str(ROOT / "data"))
    os.environ.setdefault("DPPO_LOG_DIR", str(ROOT / "log"))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(torch.load(saved["normalization_path"], map_location="cpu", weights_only=True))
    columns = {}
    for path in sorted(args.rollouts.glob("bc_seed_*.npz")):
        with np.load(path, allow_pickle=False) as raw:
            for key in ("pre_states", "actions", "leg_height_gain_m", "tcp_leg_distance_m",
                        "gripper_width_m", "tcp_leg_relative_position_m"):
                columns.setdefault(key, []).append(raw[key].copy())
    raw = {key: np.concatenate(value) for key, value in columns.items()}
    states, commands = raw["pre_states"][:, 1:-1], raw["actions"][:, 1:-1]
    shape = states.shape[:2]
    obs = torch.from_numpy(states.reshape(-1, 1, 58)).float()
    actions = torch.from_numpy(commands.reshape(-1, 8, 10)).float()
    physical = normalizer(obs[:, 0], "observations", forward=False)
    physical_actions = normalizer(actions, "actions", forward=False)
    held = physical_actions.clone()
    held[:, :, :9] = physical[:, None, :9]
    hold = normalizer(held, "actions", forward=True)
    feasible = (hold.abs() <= 1 + 1e-5).all(dim=(1, 2)).numpy().reshape(shape)
    moving = ((physical_actions[:, :, :3] - physical[:, None, :3]).norm(dim=-1).amax(1) > .003).numpy().reshape(shape)
    before_distance, after_distance = raw["tcp_leg_distance_m"][:, :-2], raw["tcp_leg_distance_m"][:, 1:-1]
    before_height, after_height = raw["leg_height_gain_m"][:, :-2], raw["leg_height_gain_m"][:, 1:-1]
    width = raw["gripper_width_m"][:, 1:-1]
    slip = np.linalg.norm(np.diff(raw["tcp_leg_relative_position_m"], axis=1)[:, :-1], axis=-1)
    groups = {
        "approach_progress": moving & (before_distance > .05) & (before_distance - after_distance > .01),
        "approach_stalled": moving & (before_distance > .05) & (np.abs(before_distance - after_distance) < .002),
        "coupled_upward_motion": moving & (after_height - before_height > .02) & (after_distance < .04)
            & (width > .015) & (width < .045) & (slip < .01),
        "leg_drop": (before_height > .06) & (before_height - after_height > .02),
    }
    rng = np.random.default_rng(2026110121)
    indices, names = [], []
    for name, mask in groups.items():
        # At most one state per episode within a category avoids letting a
        # single long oscillation dominate a 32-state diagnostic.
        episodes = np.flatnonzero((mask & feasible).any(1))
        episodes = rng.choice(episodes, min(32, len(episodes)), replace=False)
        for episode in episodes:
            time = rng.choice(np.flatnonzero((mask & feasible)[episode]))
            indices.append(int(episode * shape[1] + time))
            names.append(name)
    indices = torch.tensor(indices)
    obs = obs[indices].to(args.device)
    actions = actions[indices].reshape(-1, 80).to(args.device)
    hold = hold[indices].reshape(-1, 80).clamp(-1, 1).to(args.device)
    cfg = compose_config("cfg/furniture/finetune/one_leg_low", "ft_ppo_diffusion_mlp", [])
    cfg.model.device = args.device
    cfg.model.network_path = saved["source_bc_checkpoint"]
    assert sha256(cfg.model.network_path) == saved["source_bc_sha256"]
    native = hydra.utils.instantiate(cfg.model).eval()
    teacher = MacroDiffusionPolicy(native, score_noise_samples=16)
    model = LFIRLReward(saved["learner"]["settings"], 58, 80, args.device)
    model.load_state_dict(saved["learner"])
    references = []
    for seed in (2026110122, 2026110123):
        derivatives = []
        for fraction in np.linspace(0, 1, 9):
            torch.manual_seed(seed)
            query = hold + float(fraction) * (actions - hold)
            score = teacher.score_first_action(obs, query[:, None], t_eval=0)
            derivatives.append((score * (actions - hold)).sum(-1).cpu().numpy())
        references.append(np.trapz(np.stack(derivatives), dx=1 / 8, axis=0))
    reference = np.mean(references, axis=0)
    with torch.no_grad():
        q_gap = (model.q(obs, actions) - model.q(obs, hold)).reshape(-1).cpu().numpy()
    results = {}
    for name in groups:
        selected = np.asarray(names) == name
        results[name] = dict(
            states=int(selected.sum()),
            teacher_actual_over_hold_fraction=float((reference[selected] > 0).mean()),
            q_actual_over_hold_fraction=float((q_gap[selected] > 0).mean()),
            teacher_q_direction_agreement=float(((reference[selected] > 0) == (q_gap[selected] > 0)).mean()),
            teacher_direction_repeatability=float(((references[0][selected] > 0) == (references[1][selected] > 0)).mean()),
        )
    result = dict(
        results=results, source_bc_sha256=saved["source_bc_sha256"],
        source_indices=indices.tolist(), policy_updated=False, reward_updated=False,
        reference="Trapezoidal integral of the unchanged upstream small-noise BC score: nine nodes, two independent 16-noise-draw repeats.",
        caveat="Not exact DDIM likelihood or a causal dynamics comparison. A recorded drop may occur despite an intended recovery action.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "source_indices"}, indent=2))


if __name__ == "__main__":
    main()
