"""Compare reward candidates on matched resets with reset-cluster uncertainty."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def preference_samples(scores, events, gamma):
    """One row per reset; policy pairs sharing a reset stay in the same row."""
    pairs = list(itertools.combinations(scores, 2))
    if not pairs:
        raise ValueError("At least two policies are required")
    count, horizon = next(iter(scores.values())).shape
    full = np.full((count, len(pairs)), np.nan)
    completion = full.copy()
    cumulative = {name: (values * gamma ** np.arange(horizon)).cumsum(1)
                  for name, values in scores.items()}
    successes = {name: values.sum(1) >= 1 for name, values in events.items()}
    for column, (first, second) in enumerate(pairs):
        different = successes[first] != successes[second]
        sign = np.where(successes[first], 1, -1)
        margins = (cumulative[first][:, -1] - cumulative[second][:, -1]) * sign
        full[different, column] = (margins[different] > 0) + .5 * (margins[different] == 0)
        for index in np.flatnonzero(different):
            successful = first if successes[first][index] else second
            end = int(np.flatnonzero(events[successful][index] >= 1)[0])
            if end < horizon:
                margin = (cumulative[first][index, end] - cumulative[second][index, end]) * sign[index]
                completion[index, column] = float((margin > 0) + .5 * (margin == 0))
    return {"full_horizon": full, "completion_prefix": completion}


def cluster_summary(values, draws=10000):
    """Bootstrap whole initial resets, never the correlated policy pairs."""
    values = values[np.isfinite(values).any(1)]
    if not len(values):
        return {"informative_resets": 0, "comparisons": 0, "mean": None, "ci95": None}
    totals = np.nansum(values, axis=1)
    counts = np.isfinite(values).sum(1)
    selected = np.random.default_rng(20261101).integers(len(values), size=(draws, len(values)))
    bootstrap = totals[selected].sum(1) / counts[selected].sum(1)
    return dict(informative_resets=len(values), comparisons=int(counts.sum()),
                mean=float(totals.sum() / counts.sum()),
                ci95=np.quantile(bootstrap, [.025, .975]).tolist())


def load_scores(path):
    report = json.loads(path.with_suffix(".json").read_text())
    if not report.get("initial_state_offsets_controlled"):
        raise ValueError("Run the physical-reset equality audit before comparing reward ranks")
    with np.load(path, allow_pickle=False) as raw:
        scores, events = raw["rewards"].copy(), raw["assembly"].copy()
        sources = json.loads(str(raw["sources_json"]))
        gamma = float(raw["gamma"])
        targets = raw["bellman_targets"].copy() if "bellman_targets" in raw else None
    for source in sources:
        source["archive"] = str(Path(source["archive"]).resolve())
    grouped_scores, grouped_events, grouped_targets = {}, {}, {}
    start = 0
    for source in sources:
        name, count = source["policy"], source["episodes"]
        grouped_scores.setdefault(name, []).append(scores[start:start + count])
        grouped_events.setdefault(name, []).append(events[start:start + count])
        if targets is not None:
            grouped_targets.setdefault(name, []).append(targets[start:start + count])
        start += count
    if start != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid aligned reward scores")
    concatenate = lambda groups: {name: np.concatenate(parts) for name, parts in groups.items()}
    return (concatenate(grouped_scores), concatenate(grouped_events), gamma,
            sources, concatenate(grouped_targets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=PATH.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results, baseline, reference = {}, None, None
    for argument in args.candidate:
        name, path = argument.split("=", 1)
        scores, events, gamma, sources, targets = load_scores(Path(path))
        if reference is None:
            reference = (events, gamma, sources)
        else:
            if gamma != reference[1] or sources != reference[2]:
                raise ValueError("Candidates must use identical rollout archives and discount")
            for policy in events:
                np.testing.assert_array_equal(events[policy], reference[0][policy])
        samples = preference_samples(scores, events, gamma)
        if baseline is None:
            baseline = samples
        result = {key: cluster_summary(value) for key, value in samples.items()}
        result["difference_from_first_candidate"] = {
            key: cluster_summary(value - baseline[key]) for key, value in samples.items()
        }
        if targets:
            result["frozen_bellman_targets"] = {
                key: cluster_summary(value)
                for key, value in preference_samples(targets, events, gamma).items()
            }
        results[name] = result
    report = dict(
        results=results,
        scope="Recorded-rollout validation. Reserved seeds and candidate selection are documented by the run's validation plan; this script alone does not establish a blind test. Assembly labels are evaluation-only.",
        uncertainty="Percentile bootstrap over initial reset IDs; paired trajectories are correlated.",
        limitation="Reward ranks do not establish a causal improvement in policy assembly success.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
