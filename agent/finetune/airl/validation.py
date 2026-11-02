"""Sample the unchanged policy, fit a separate density, and fail closed on validation."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from agent.finetune.airl.density import (
    CalibratedActionDensity,
    MixedActionDensity,
    sampler_specification,
)


class DensityValidationError(RuntimeError):
    """The approximate final-action likelihood is not qualified for AIRL."""


def policy_fingerprint(policy):
    digest = hashlib.sha256()
    for module_name in ("actor", "actor_ft", "eta"):
        module = getattr(policy, module_name, None)
        if module is None:
            continue
        for name, tensor in module.state_dict().items():
            digest.update(f"{module_name}.{name}".encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    settings = {
        name: getattr(policy, name)
        for name in (
            "use_ddim", "ddim_steps", "denoising_steps", "ft_denoising_steps",
            "randn_clip_value", "denoised_clip_value", "final_action_clip_value",
            "eps_clip_value",
        )
    }
    settings["min_sampling_std"] = float(policy.get_min_sampling_denoising_std())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


@torch.no_grad()
def sample_policy(policy, observations, draws, batch_size=512, progress=None):
    """Use the original forward method, including its clipping and noise schedule."""
    contexts = observations.repeat_interleave(draws, dim=0)
    actions = []
    for start in range(0, len(contexts), batch_size):
        if progress is not None:
            progress()
        chunk = policy(
            {"state": contexts[start:start + batch_size, None]},
            deterministic=False,
            # The existing VPG sampler's full-finetuning branch expects a
            # chain container. Discard it here; do not change the sampler.
            return_chain=True,
        ).trajectories
        actions.append(chunk)
    return contexts, torch.cat(actions)


def ks_uniform(values):
    sorted_values = values.sort(dim=0).values
    n = len(values)
    lower = torch.arange(n, device=values.device, dtype=values.dtype)[:, None] / n
    return torch.maximum(
        (sorted_values - lower).amax(0),
        (lower + 1 / n - sorted_values).amax(0),
    )


@torch.no_grad()
def validation_metrics(model, train_data, validation_data, state_count, draws):
    train_state, train_action = train_data
    state, action = validation_data
    model.eval()
    train_logp = torch.cat([
        model.log_prob(train_action[start:start + 512], train_state[start:start + 512])
        for start in range(0, len(train_action), 512)
    ])
    logp = []
    pits = []
    for start in range(0, len(action), 512):
        observations = state[start:start + 512]
        actions = action[start:start + 512]
        logp.append(model.log_prob(actions, observations))
        pits.append(model.randomized_pit(actions, observations))
    logp = torch.cat(logp)
    pits = torch.cat(pits)
    per_state = pits.reshape(state_count, draws, -1)
    conditional_ks = torch.stack([ks_uniform(values) for values in per_state])
    centered = pits - pits.mean(0)
    scale = centered.square().mean(0).sqrt().clamp_min(1e-8)
    standardized = centered / scale
    correlations = standardized.T @ standardized / len(pits)
    correlations.fill_diagonal_(0)
    return dict(
        train_nll=float(-train_logp.mean()),
        validation_nll=float(-logp.mean()),
        overfit_gap_nats=float((train_logp.mean() - logp.mean()).clamp_min(0)),
        pooled_pit_ks_max=float(ks_uniform(pits).max()),
        conditional_pit_ks_max=float(conditional_ks.max()),
        pit_correlation_max=float(correlations.abs().max()),
        finite_log_prob=bool(torch.isfinite(logp).all()),
    )


class ValidatedActionDensity:
    """A policy-version-specific estimator that cannot be used before qualification."""

    def __init__(self, settings, output_dir):
        self.settings = settings
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = []
        self.validated = False
        self.report = None
        self.registered_states = None
        self.fingerprint = None

    def fit_and_validate(
        self, policy, states, query_actions, progress=None, query_groups=None
    ):
        self.validated = False
        self.fingerprint = policy_fingerprint(policy)
        self.registered_states = states.detach().clone()
        specification = sampler_specification(policy)
        if query_groups is None:
            query_groups = ["unspecified"] * len(states)
        if len(query_groups) != len(states) or len(query_actions) != len(states):
            raise ValueError("Each density query requires one state, action and group")
        # Save the actual discriminator queries before sampling or fitting.
        # A failed iteration can then be reproduced without advancing physics
        # or reconstructing lost rollout actions from random seeds.
        torch.save(
            dict(
                states=states.detach().cpu(),
                query_actions=query_actions.detach().cpu(),
                groups=list(query_groups),
                policy_sha256=self.fingerprint,
                policy_training_mode=policy.training,
                sampler_specification=specification,
                settings=dict(self.settings),
                rng_state=torch.random.get_rng_state(),
                cuda_rng_state=(
                    torch.cuda.get_rng_state(states.device) if states.is_cuda else None
                ),
            ),
            self.output_dir / "queries.pt",
        )
        draws = int(self.settings.get("training_draws", 256))
        validation_draws = int(self.settings.get("validation_draws", 256))
        development_draws = int(self.settings.get("development_draws", 4096))
        steps = int(self.settings.get("fit_steps", 5000))
        if min(draws, development_draws, validation_draws, steps) < 1:
            raise ValueError("Positive density sampling and fitting counts required")
        print(
            f"[AIRL density] frozen policy; contexts={len(states)}, "
            f"training draws/context={draws}, validation draws/context={validation_draws}",
            flush=True,
        )
        train = sample_policy(policy, states, draws, progress=progress)
        development = sample_policy(policy, states, development_draws, progress=progress)
        validation = sample_policy(policy, states, validation_draws, progress=progress)
        self.models = []
        model_reports = []
        for member in range(2):
            model = MixedActionDensity(
                **specification,
                hidden_features=int(self.settings.get("hidden_features", 512)),
                mixtures=int(self.settings.get("mixtures", 16)),
                reverse=bool(member),
            ).to(states.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=steps, eta_min=1e-5
            )
            best_development_nll = float("inf")
            best_weights = None
            for step in range(1, steps + 1):
                ids = torch.randint(len(train[0]), (512,), device=states.device)
                chunk_nll = -model.log_prob(train[1][ids], train[0][ids]).mean()
                loss = chunk_nll / model.features
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                scheduler.step()
                if step % 200 == 0 or step == steps:
                    with torch.no_grad():
                        development_logp = torch.cat([
                            model.log_prob(
                                development[1][start:start + 512],
                                development[0][start:start + 512],
                            )
                            for start in range(0, len(development[0]), 512)
                        ])
                        development_nll = float(-development_logp.mean())
                    if development_nll < best_development_nll:
                        best_development_nll = development_nll
                        best_weights = copy.deepcopy(model.state_dict())
                    print(
                        f"[AIRL density] member={member + 1}/2 step={step}/{steps} "
                        f"NLL={float(chunk_nll.detach()):.3f} nats/chunk",
                        flush=True,
                    )
                    if progress is not None:
                        progress()
            model.load_state_dict(best_weights, strict=True)
            model = CalibratedActionDensity(
                model, states, development, development_draws
            )
            self.models.append(model.eval())
            model_reports.append(validation_metrics(
                model, train, validation, len(states), validation_draws
            ))
        with torch.no_grad():
            predictions = [model.log_prob(query_actions, states) for model in self.models]
            difference = (predictions[0] - predictions[1]).abs()
        thresholds = dict(
            pooled_pit_ks_max=float(self.settings.get("max_pooled_pit_ks", 0.04)),
            conditional_pit_ks_max=float(self.settings.get("max_conditional_pit_ks", 0.20)),
            pit_correlation_max=float(self.settings.get("max_pit_correlation", 0.10)),
            overfit_gap_nats=float(self.settings.get("max_overfit_gap_nats", 2.0)),
            query_logp_rmse_nats=float(self.settings.get("max_query_logp_rmse_nats", 1.0)),
            query_logp_p95_nats=float(self.settings.get("max_query_logp_p95_nats", 2.0)),
        )
        failures = []
        if not all(torch.isfinite(prediction).all() for prediction in predictions):
            failures.append("nonfinite expert/policy query likelihood")
        for member, metrics in enumerate(model_reports):
            if not metrics["finite_log_prob"]:
                failures.append(f"member {member}: nonfinite likelihood")
            for name in (
                "pooled_pit_ks_max", "conditional_pit_ks_max",
                "pit_correlation_max", "overfit_gap_nats",
            ):
                if metrics[name] > thresholds[name]:
                    failures.append(f"member {member}: {name}={metrics[name]:.4f}")
        query_rmse = float(difference.square().mean().sqrt())
        query_p95 = float(torch.quantile(difference, 0.95))
        if query_rmse > thresholds["query_logp_rmse_nats"]:
            failures.append(f"query log-probability ensemble RMSE={query_rmse:.4f} nats")
        if query_p95 > thresholds["query_logp_p95_nats"]:
            failures.append(f"query log-probability ensemble p95={query_p95:.4f} nats")
        if policy_fingerprint(policy) != self.fingerprint:
            failures.append("policy changed during density fitting")
        self.report = dict(
            passed=not failures,
            method="conditional_mixed_autoregressive_density",
            policy_sha256=self.fingerprint,
            sampler_modified=False,
            log_probability_units="joint nats per action chunk; mixed reference measure",
            action_features=specification["action_features"],
            training_samples=len(train[0]),
            development_samples=len(development[0]),
            validation_samples=len(validation[0]),
            registered_contexts=len(states),
            member_metrics=model_reports,
            query_logp_rmse_nats=query_rmse,
            query_logp_p95_nats=query_p95,
            query_logp=[
                dict(
                    index=index,
                    group=query_groups[index],
                    member_logp_nats=[float(value[index]) for value in predictions],
                    disagreement_nats=float(difference[index]),
                )
                for index in range(len(states))
            ],
            density_settings=dict(self.settings),
            thresholds=thresholds,
            failures=failures,
            limitation=(
                "Held-out mixed PIT calibration and independent autoregressive orders "
                "qualify a numerical surrogate; true pointwise diffusion likelihood "
                "is unavailable and finite-sample validation is not an exact-error proof."
            ),
        )
        (self.output_dir / "validation.json").write_text(json.dumps(self.report, indent=2))
        with (self.output_dir / "validation_history.jsonl").open("a") as stream:
            stream.write(json.dumps(self.report) + "\n")
        torch.save(
            dict(models=[model.state_dict() for model in self.models], report=self.report),
            self.output_dir / "density.pt",
        )
        if failures:
            raise DensityValidationError(
                "Final-action density validation failed; AIRL/DPPO updates blocked. "
                + "; ".join(failures)
            )
        self.validated = True
        return self.report

    @torch.no_grad()
    def log_prob(self, actions, states):
        if not self.validated:
            raise DensityValidationError("Validate the current policy density before AIRL")
        distances = (
            states.flatten(1)[:, None] - self.registered_states.flatten(1)[None]
        ).abs().amax(-1)
        if (distances.amin(1) > 1e-6).any():
            raise DensityValidationError("AIRL queried an unvalidated observation context")
        logp = torch.stack([model.log_prob(actions, states) for model in self.models])
        if not torch.isfinite(logp).all():
            raise DensityValidationError("Nonfinite final-action likelihood")
        return (torch.logsumexp(logp, dim=0) - np.log(len(self.models))).float()

    def invalidate(self):
        self.validated = False
