"""Measure one isolated native LFIRL/DPPO iteration without changing its losses."""

from collections import defaultdict
from functools import wraps
import json
from pathlib import Path
import time

import torch

from agent.finetune.train_lfirl_diffusion_agent import TrainLFIRLDiffusionAgent


def shapes(value):
    if isinstance(value, dict):
        return {key: shapes(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [shapes(item) for item in value]
    if hasattr(value, "shape"):
        return list(value.shape)
    return type(value).__name__


class ProfileLFIRLDiffusionAgent(TrainLFIRLDiffusionAgent):
    def run(self):
        self._measurements = defaultdict(list)
        self._interfaces = {}
        self._gradient_probe = None
        bindings = []
        use_cuda = torch.device(self.device).type == "cuda"

        def instrument(owner, method, label):
            original = getattr(owner, method)

            @wraps(original)
            def measured(*args, **kwargs):
                category = label
                if label == "policy_sampling":
                    category = "teacher_sampling" if kwargs.get("use_base_policy") else "rollout_sampling"
                start_event = torch.cuda.Event(enable_timing=True) if use_cuda else None
                end_event = torch.cuda.Event(enable_timing=True) if use_cuda else None
                if use_cuda:
                    start_event.record()
                started = time.perf_counter()
                result = original(*args, **kwargs)
                elapsed = time.perf_counter() - started
                if use_cuda:
                    end_event.record()
                self._measurements[category].append((elapsed, start_event, end_event))
                if category not in self._interfaces:
                    self._interfaces[category] = dict(
                        inputs=[shapes(item) for item in args],
                        keyword_inputs={key: shapes(value) for key, value in kwargs.items()},
                        output=shapes(result),
                    )
                if label == "ppo_loss" and self._gradient_probe is None:
                    self._probe_gradients(result)
                return result

            setattr(owner, method, measured)
            bindings.append((owner, method, original))

        for owner, method, label in (
            (self, "reset_env_all", "reset"),
            (self.venv, "step", "environment_step"),
            (self.model, "forward", "policy_sampling"),
            (self.model, "loss", "ppo_loss"),
            (self.model, "get_logprobs", "denoising_logprobs"),
            (self.model.critic, "forward", "critic_forward"),
            (self.reward, "predict", "reward_prediction"),
            (self, "get_training_rewards", "reward_bridge"),
            (self.actor_optimizer, "step", "actor_optimizer_step"),
            (self.critic_optimizer, "step", "critic_optimizer_step"),
            (self, "save_model", "checkpoint_save"),
        ):
            instrument(owner, method, label)
        if use_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        completed = False
        try:
            super().run()
            completed = True
        finally:
            if use_cuda:
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            for owner, method, original in reversed(bindings):
                setattr(owner, method, original)
            timings = {}
            for label, records in self._measurements.items():
                timings[label] = dict(
                    calls=len(records), host_wall_seconds=sum(row[0] for row in records),
                    gpu_seconds=(sum(row[1].elapsed_time(row[2]) for row in records) / 1000
                                 if use_cuda else None),
                )
            nonfinite = [name for name, parameter in self.model.named_parameters()
                         if not bool(torch.isfinite(parameter).all())]
            recognized = []
            if (not self.model.learn_eta and type(self.model.eta).__name__ == "EtaFixed"
                    and not self.model.eta.eta_logit.requires_grad
                    and bool(torch.isinf(self.model.eta.eta_logit).all())):
                eta = self.model.eta({"state": torch.zeros(1, 1, self.obs_dim, device=self.device)})
                if bool(torch.isfinite(eta).all()):
                    recognized.append("eta.eta_logit")
            report = dict(
                completed=completed, measured_loop_seconds=elapsed,
                environments=self.n_envs, rollout_chunks=self.n_steps,
                action_horizon=self.act_steps, denoising_steps=self.model.ddim_steps,
                ppo_batch_size=self.batch_size, update_epochs=self.update_epochs,
                timings=timings, interfaces=self._interfaces,
                first_minibatch_actor_gradients=self._gradient_probe,
                nonfinite_parameter_names=nonfinite,
                recognized_frozen_eta_boundary_parameters=recognized,
                all_trainable_parameters_finite=all(
                    bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()
                    if parameter.requires_grad
                ),
                no_unexpected_nonfinite_parameters=all(name in recognized for name in nonfinite),
                peak_cuda_allocated_gib=(torch.cuda.max_memory_allocated(self.device) / 2**30
                                         if use_cuda else None),
                notes=[
                    "Timings are inclusive and nested; do not add all categories.",
                    "CUDA events measure GPU stream time; host timings may include synchronization.",
                    "Scene construction is excluded; checkpoint saving is included.",
                    "One minibatch has two diagnostic gradient passes; their overhead is reported separately.",
                    "This isolated diagnostic policy is never promoted to the training or deployment policy.",
                    "Upstream fixed eta=1 uses atanh(1)=infinity internally; its tanh output is finite.",
                ],
            )
            path = Path(self.logdir) / "profile.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"[LFIRL profile] completed={completed} seconds={elapsed:.3f} report={path}", flush=True)

    def _probe_gradients(self, losses):
        started = time.perf_counter()
        parameters = [parameter for parameter in self.model.actor_ft.parameters() if parameter.requires_grad]
        pg = torch.autograd.grad(losses[0], parameters, retain_graph=True, allow_unused=True)
        bc = torch.autograd.grad(
            self.bc_loss_coeff * losses[6], parameters, retain_graph=True, allow_unused=True,
        ) if self.use_bc_loss else [None] * len(parameters)
        pg_squared = sum(gradient.square().sum() for gradient in pg if gradient is not None)
        bc_squared = sum(gradient.square().sum() for gradient in bc if gradient is not None)
        dot = sum((first * second).sum() for first, second in zip(pg, bc)
                  if first is not None and second is not None)
        pg_norm = float(torch.as_tensor(pg_squared).sqrt())
        bc_norm = float(torch.as_tensor(bc_squared).sqrt())
        self._gradient_probe = dict(
            policy_gradient_norm=pg_norm, weighted_bc_gradient_norm=bc_norm,
            bc_to_policy_gradient_norm_ratio=bc_norm / max(pg_norm, 1e-12),
            gradient_cosine=float(torch.as_tensor(dot)) / max(pg_norm * bc_norm, 1e-12),
            diagnostic_overhead_seconds=time.perf_counter() - started,
            scope="First real PPO minibatch only; not an iteration-wide attribution or a performance ablation.",
        )
