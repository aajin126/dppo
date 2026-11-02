"""Qualify final-action density on saved queries without updating policy/reward."""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from script.isaaclab.run_lab import compose_config, prepare_cuda_libraries


def main():
    prepare_cuda_libraries()
    import hydra
    import numpy as np
    import torch
    from agent.finetune.airl.validation import ValidatedActionDensity, policy_fingerprint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--config-dir", default="cfg/furniture/finetune/one_leg_low")
    parser.add_argument("--config-name", default="ft_airl_diffusion_mlp")
    args, overrides = parser.parse_known_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = compose_config(args.config_dir, args.config_name, overrides)
    cfg.model.device = args.device
    cfg.model.network_path = str(args.checkpoint.resolve())
    policy = hydra.utils.instantiate(cfg.model)
    queries = torch.load(args.queries, map_location="cpu", weights_only=True)
    policy.train(bool(queries.get("policy_training_mode", True)))
    fingerprint = policy_fingerprint(policy)
    expected = queries.get("policy_sha256")
    if expected is not None and fingerprint != expected:
        raise ValueError("Query snapshot belongs to a different policy/sampler")
    states = queries["states"].to(args.device)
    actions = queries["query_actions"].to(args.device)
    before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "qualification_source.json").write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint.resolve()),
        queries=str(args.queries.resolve()),
        sampling_seed=args.seed,
        simulator_advanced=False,
        policy_updated=False,
        reward_updated=False,
    ), indent=2))
    density = ValidatedActionDensity(cfg.airl.density, args.output)
    report = density.fit_and_validate(
        policy, states, actions, query_groups=queries.get("groups")
    )
    for name, tensor in policy.state_dict().items():
        if not torch.equal(before[name], tensor.detach().cpu()):
            raise RuntimeError(f"Policy tensor changed during qualification: {name}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
