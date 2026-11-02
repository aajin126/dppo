"""NPZ transitions at the same eight-action time scale as native DPPO."""

import numpy as np
import torch


class MacroTransitions:
    def __init__(self, path, horizon=8, terminal_mode="truncate", max_episodes=0, window_stride=None):
        if terminal_mode not in ("truncate", "demonstration_end"):
            raise ValueError("terminal_mode must be truncate or demonstration_end")
        window_stride = horizon if window_stride is None else window_stride
        if horizon < 1 or window_stride < 1:
            raise ValueError("Action horizon and window stride must be positive")
        with np.load(path, allow_pickle=False) as source:
            states = source["states"].astype(np.float32)
            actions = source["actions"].astype(np.float32)
            lengths = source["traj_lengths"].astype(np.int64)
        if lengths.sum() != len(states) or len(actions) != len(states):
            raise ValueError("NPZ lengths do not align")
        valid = ~((states == 0).all(-1) & (actions == 0).all(-1))
        boundaries = np.r_[0, lengths.cumsum()]
        observations, chunks, next_observations, terminals, episode_ids = [], [], [], [], []
        used = 0
        for episode, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            rows = np.flatnonzero(valid[start:end]) + start
            if len(rows) <= horizon:
                continue
            # Split at padding gaps; never splice separated source time steps.
            segments = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
            before = len(chunks)
            for segment in segments:
                starts = list(range(0, len(segment) - horizon, window_stride))
                final_offset = len(segment) - horizon - 1
                if terminal_mode == "demonstration_end" and segment is segments[-1] and final_offset >= 0:
                    starts = sorted(set(starts + [final_offset]))
                for offset in starts:
                    t = segment[offset]
                    next_t = segment[offset + horizon]
                    observations.append(states[t:t + 1])
                    chunks.append(actions[t:t + horizon].reshape(-1))
                    next_observations.append(states[next_t:next_t + 1])
                    # NPZ has no terminal flags. The conservative default keeps
                    # only known interior transitions, without inventing done.
                    terminals.append(terminal_mode == "demonstration_end" and next_t == rows[-1])
                    episode_ids.append(episode)
            if len(chunks) > before:
                used += 1
            if max_episodes and used >= max_episodes:
                break
        if not chunks:
            raise ValueError("No complete contiguous macro transitions in NPZ")
        self.observations = torch.from_numpy(np.stack(observations))
        self.actions = torch.from_numpy(np.stack(chunks))
        self.next_observations = torch.from_numpy(np.stack(next_observations))
        self.terminals = torch.tensor(terminals, dtype=torch.float32)
        self.episode_ids = np.asarray(episode_ids)
        self.summary = dict(
            transitions=len(chunks), expert_episodes=used, horizon=horizon,
            observation_dim=states.shape[1], action_dim=actions.shape[1] * horizon,
            terminal_mode=terminal_mode, terminal_flags_available=False,
            terminal_transitions=int(self.terminals.sum()),
            source_time_step_seconds=.1, macro_time_step_seconds=.1 * horizon,
            expert_timestamps_available=False,
            time_step_assumption="Nominal action period from the current fork; NPZ has no timestamps",
            window_stride=window_stride,
        )

    def batches(self, batch_size, device, shuffle=False, generator=None):
        order = torch.randperm(len(self.actions), generator=generator) if shuffle else None
        for start in range(0, len(self.actions), batch_size):
            selection = (order[start:start + batch_size] if shuffle
                         else slice(start, start + batch_size))
            yield tuple(array[selection].to(device) for array in (
                self.observations, self.actions, self.next_observations, self.terminals
            ))


class ReferenceRewardTransitions(MacroTransitions):
    """Add unlabeled frozen-BC transitions for calibration, never Q imitation.

    Complete reference episodes are reserved before constructing any windows.
    Assembly outcomes are ignored by default. An explicit option uses native
    completion only as the episode boundary, without adding reward targets.
    """

    def __init__(self, expert, directory, normalizer_hash, bc_hash, training_fraction=.75,
                 assembly_terminal=False, success_threshold=None):
        import json
        from pathlib import Path
        import yaml
        from agent.finetune.lfirl.reward import sha256

        directory = Path(directory)
        report = json.loads((directory / "report.json").read_text())
        cfg = yaml.safe_load((directory / "evaluation_config.yaml").read_text())
        if not report.get("completed") or not report.get("bc_matches_original_checkpoint"):
            raise ValueError("Reference collection must be complete and use the original frozen BC")
        if (sha256(cfg["normalization_path"]) != normalizer_hash
                or sha256(cfg["base_policy_path"]) != bc_hash):
            raise ValueError("Reference BC or observation normalization differs")
        if not 0 < training_fraction < 1:
            raise ValueError("Reserve complete reference episodes for validation")
        if assembly_terminal and (success_threshold is None or not np.isfinite(success_threshold)
                                  or success_threshold <= 0):
            raise ValueError("Native assembly terminals require an explicit positive success threshold")
        horizon = expert.summary["horizon"]
        action_dim = expert.summary["action_dim"] // horizon
        observation_dim = expert.summary["observation_dim"]
        training, validation = [[] for _ in range(4)], [[] for _ in range(4)]
        archives = []
        training_episodes = validation_episodes = 0
        for path in sorted(directory.glob("bc_seed_*.npz")):
            with np.load(path, allow_pickle=False) as raw:
                states = raw["pre_states"].astype(np.float32)
                actions = raw["actions"].astype(np.float32)
                events = raw["assembly_rewards"].copy() if assembly_terminal else None
            if (states.ndim != 3 or states.shape[1] < 2 or states.shape[-1] != observation_dim
                    or actions.shape != (*states.shape[:2], horizon, action_dim)
                    or not np.isfinite(states).all() or not np.isfinite(actions).all()):
                raise ValueError(f"Invalid aligned reference state/action arrays: {path}")
            valid = np.ones(states.shape[:2], dtype=bool)[:, :-1]
            terminals = np.zeros(valid.shape, dtype=np.float32)
            if assembly_terminal:
                from agent.finetune.lfirl.episodes import assembly_boundaries

                if events.shape != states.shape[:2]:
                    raise ValueError("Native assembly events must align with reference episodes and chunks")
                all_valid, all_terminals = assembly_boundaries(events.T, success_threshold)
                valid = all_valid.T[:, :-1]
                terminals = all_terminals.T[:, :-1].astype(np.float32)
            split = int(len(states) * training_fraction)
            if split < 1 or split >= len(states):
                raise ValueError("Reference collection needs training and held-out episodes")
            for first, last, columns in ((0, split, training), (split, len(states), validation)):
                selected = valid[first:last]
                pre = states[first:last, :-1][selected][:, None]
                post = states[first:last, 1:][selected][:, None]
                commands = actions[first:last, :-1][selected].reshape(len(pre), -1)
                # The last next-state is unavailable. Only known interiors
                # enter regression; no absorbing state or success is invented.
                for column, array in zip(columns, (pre, commands, post, terminals[first:last][selected])):
                    column.append(torch.from_numpy(array.copy()))
            training_episodes += split
            validation_episodes += len(states) - split
            archives.append(dict(path=str(path.resolve()), sha256=sha256(path),
                                 training_env_ids=[0, split], validation_env_ids=[split, len(states)]))
        if not archives:
            raise ValueError("No frozen-BC reference archives found")
        reference = tuple(torch.cat(column) for column in training)
        self.reference_validation = tuple(torch.cat(column) for column in validation)
        for name, original, added in zip(
            ("observations", "actions", "next_observations", "terminals"),
            (expert.observations, expert.actions, expert.next_observations, expert.terminals), reference,
        ):
            setattr(self, name, torch.cat((original, added)))
        self.episode_ids = np.r_[expert.episode_ids, np.full(len(reference[0]), -1)]
        self.summary = dict(expert.summary)
        self.summary.update(
            transitions=len(self.actions), expert_transitions=len(expert.actions),
            reference_transitions=len(reference[0]), reference_training_episodes=training_episodes,
            reference_validation_episodes=validation_episodes,
            reference_validation_transitions=len(self.reference_validation[0]),
            reference_archives=archives, reference_training_fraction=training_fraction,
            reference_usage="Frozen-Q calibration or R regression",
            reference_policy="frozen original BC evaluation sampler",
            q_score_reference_distribution_changed=False,
        )
        self.summary.update(
            reference_assembly_labels_used=bool(assembly_terminal),
            reference_assembly_terminal=bool(assembly_terminal),
            reference_success_threshold=success_threshold if assembly_terminal else None,
            reference_terminal_usage="Native episode boundary only; no assembly reward target or bonus",
            reference_training_terminals=int(reference[3].sum()),
            reference_validation_terminals=int(self.reference_validation[3].sum()),
        )

    def validation_batches(self, batch_size, device):
        for start in range(0, len(self.reference_validation[0]), batch_size):
            yield tuple(array[start:start + batch_size].to(device)
                        for array in self.reference_validation)
