"""Measure density estimation error on an analytic mixed conditional law.

Run in the same Python environment as AIRL. This checks implementation error
against known log probabilities; it does not establish diffusion likelihoods.
"""

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.finetune.airl.density import MixedActionDensity


def exact_samples(count, bound, atoms):
    context = torch.rand(count, 1) * 2 - 1
    mean = .35 * torch.tanh(2 * context[:, 0])
    std = .10 + .05 * torch.sigmoid(context[:, 0])
    low = torch.special.ndtr((-bound - mean) / std)
    high = torch.special.ndtr((bound - mean) / std)
    uniform = low + torch.rand(count) * (high - low)
    values = mean + std * torch.special.ndtri(uniform.clamp(1e-7, 1 - 1e-7))
    mode = torch.multinomial(
        torch.tensor([.005, .005, .005, .005, .98]), count, replacement=True
    )
    discrete = mode < 4
    values[discrete] = atoms[mode[discrete]]
    logp = -.5 * ((values - mean) / std).square() - std.log()
    logp -= .5 * math.log(2 * math.pi)
    logp -= (high - low).log()
    logp += math.log(.98)
    logp[discrete] = math.log(.005)
    return context, values[:, None], logp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = MixedActionDensity(1, 1, hidden_features=64, mixtures=4)
    training = exact_samples(20000, model.bound, model.atoms)
    validation = exact_samples(8000, model.bound, model.atoms)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2000):
        ids = torch.randint(len(training[0]), (256,))
        loss = -model.log_prob(training[1][ids], training[0][ids]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        difference = model.log_prob(validation[1], validation[0]) - validation[2]
    report = dict(
        reference="analytic conditional truncated Gaussian plus four point masses",
        held_out_samples=len(validation[0]),
        logp_rmse_nats=float(difference.square().mean().sqrt()),
        logp_mae_nats=float(difference.abs().mean()),
        logp_p95_abs_error_nats=float(torch.quantile(difference.abs(), .95)),
        rmse_threshold_nats=.2,
        limitation="This reference is analytic, not the unknown diffusion marginal.",
    )
    report["passed"] = report["logp_rmse_nats"] <= report["rmse_threshold_nats"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Synthetic likelihood qualification failed")


if __name__ == "__main__":
    main()
