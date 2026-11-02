# IRL method selection for the existing Furniture diffusion checkpoint

The immediate objective is to retain the existing BC policy and action sampler,
learn an explicit reward, and use native DPPO for policy improvement. The later
objective is common skill reward transfer with few-shot target demonstrations.
This selection concerns compatibility with those constraints, not a measured
claim of superior Furniture assembly performance.

| Method | Useful property | Compatibility issue |
| --- | --- | --- |
| AIRL plus a separately fitted action density | Reuses the original AIRL reward decomposition | The true marginal density of the actual clipped DDIM action sampler is not available. Agreement between two learned densities does not establish its accuracy. The earlier 1-nat gate has no AIRL-paper basis. |
| DPAIL, NeurIPS 2025 | Direct adversarial training of a diffusion policy using a tractable ASAF-objective lower bound | This is not an absolute action-log-density routine for the AIRL denominator. It does not train a separate transferable AIRL-style reward and then optimize it with DPPO. Its trajectory model and Gaussian DDPM assumptions differ from the current conditional action-chunk sampler. |
| LFIRL, September 2026 preprint | Recovers an explicit reward from a pretrained diffusion-policy score through sequential Q/V recovery | The published code's action-score interface repeats a single action across its horizon and extracts the first-action score. Adapting it to this policy's joint 80-dimensional macro action, reference-policy sampling and Bellman time step requires further validation. It is not an interchangeable AIRL module. |
| EnergyFlow, 2026 | Energy-based denoising yields a reward/preference signal | It parameterizes a scalar energy and denoises through its gradient, unlike the existing unconstrained noise-prediction MLP. Direct checkpoint reuse is not established. |
| f-IRL, CoRL 2020 | Learns an explicit stationary reward through state marginal matching; the original non-IS reward update needs no action likelihood | The original inner solver is maximum-entropy SAC. Native DPPO is a practical approximate replacement, so the original inner-optimality/convergence assumptions are not established. Cross-task skill transfer additionally needs shared features and labels. |

The current source-based implementation is **offline LFIRL reward recovery,
followed by native DPPO under a fixed recovered reward**. The official source
is pinned at `bce7509b789f1ac361d37618a85ded256af3d5d4` under
`external/lfirl`, with hashes in `external/LFIRL_SOURCES.json`.

It loads the original Q/V/offset/reward networks and complete original stage
functions, including Q-score matching, value fitting and Bellman
reward regression. Definitions are selected from the unchanged upstream script
to avoid importing unrelated benchmark environments. The native Furniture BC
epsilon model, its noise schedule and DPPO normalization are exposed through
the original LFIRL score routine. The legacy adapter treats a flattened 8x10
chunk as one 80-dimensional macro action. The additional `--reward-horizon 1`
path instead uses upstream LFIRL's single-action tiling and first-action score
approximation unchanged. Neither path supplies an exact DDIM action density.

For single-action reward recovery, the simulator records each action's actual
pre-state inside the unchanged eight-action policy chunk. PPO receives
`sum(gamma_micro**k * r(s[k], a[k]))`, with
`gamma_ppo = gamma_micro**8`. The policy's denoising gradient horizon remains
eight. The reward model's action horizon does not replace that PPO setting.

The source-supported `--v-loss expectile` option is being evaluated because
the Gumbel setting produced large value calibration errors on these data.
`--reward-observation-normalization policy` keeps the existing DPPO-normalized
inputs instead of applying a second expert-derived standardization. These
are experimental settings, not established assembly improvements. Original
loss function definitions and network implementations remain unchanged.

The original script explicitly freezes the finalized Q/B/V stage outputs
before reward regression. The adapter follows that behavior. An intermediate
experiment retained EMA targets instead, after an incomplete reading of the
source; that was a variant, not an upstream correction, and has been reverted.
New checkpoints label the target mode `final_frozen` and store those targets
so held-out regression checks can use the same values. Intermediate checkpoints
with saved targets but without that mode label belong to the EMA experiment.

NPZ terminal flags are absent. The original missing-done endpoint assumption
is available explicitly as `demonstration_end`; conservative `truncate` mode
does not fabricate terminals. This convention, the shared macro discount and
all completed stage updates are saved in the checkpoint. These are substantive
Furniture adapters, so this is not a claim of a tested Furniture reproduction
of the paper. Shared skill transfer is also not established.

Local source-default training completed, but model quality checks exposed
underfitting of the 80-dimensional Q-score target. Increasing only the Q
optimizer learning rate and number of passes improved fitting on excluded
demonstrations. This moved the bottleneck to value/reward calibration; extending
the original reward-regression stage eventually overfit those demonstrations.
Consequently the adapter is an experimentally checked integration, not a
validated Furniture reward model or evidence of better assembly. Successful
wiring and lower training loss must not be presented as reward recovery.

The October 31 diagnostics additionally compare a moving expert command with
holding the current TCP pose, on the same excluded states. Matching an average
score error was insufficient: a recovered Q could still prefer holding. Using
the native eight-action joint context avoids tiling one action across the
entire BC horizon, but fitting only expert-action queries did not resolve the
local Q error. A subsequent experiment queries the unchanged diffusion score
at expert, nearby noisy, hold, and midpoint actions. It reuses the original
Q training function, disables its expert-ranking term for nonexpert queries,
and averages 64 noise draws using the source routine's `n_noise` option.
This query distribution and target caching are adapter extensions, not an
unchanged upstream training recipe. BC weights and DDIM sampling are retained.

On the same 128 excluded moving states, Q agreement with the approximate
teacher directional integral increased from 37.5% to 94.5%. Mean queried-score
gradient MSE decreased from 2383.6 to 251.7, while expert-point gradient MSE
increased from 93.3 to 126.1. These are model-selection diagnostics, not exact
action likelihoods, an untouched test set, or assembly performance. The
experimental Q-only snapshots require refitting all V/b/R stages and are
explicitly rejected by the policy training entry point. After renewed GPU
authorization, all V/b/R stages completed. The source Gumbel configuration
scored length-matched expert trajectories above failed BC trajectories in only
6.5% of pairs, so it was rejected for policy training. The source-supported
expectile configuration improved that descriptive fraction to 78.3%. These
fractions share trajectories and are not independent trials. Its reward on
actual successful versus failed BC rollouts remains weak (only two successful
rollouts were available), and synthetic object-slip probes expose further
limitations. Both 20-iteration, 128-environment DPPO trials and separately
seeded BC/best/final evaluations completed without established improvement.
The short macro-discount experiment obtained 4/256 final-policy successes
against 3/256 BC successes, but used asymmetric snapshot restoration and did
not support a causal improvement claim. With the original DPPO macro discount
0.999 and the same restoration procedure for both policies, the final policy
obtained 2/256 successes against 3/256 BC successes. The selected intermediate
model obtained 0/256. These negative results must not be replaced by favorable
Q/reward diagnostics. The original BC checkpoint remains preserved.

An environment-reward DPPO control completed 20 training iterations and its
paired evaluation. On the same 256 initial conditions, BC achieved 3 successes
and the final control policy achieved 1 (paired exact p=0.625). The selected
checkpoint was the unchanged iteration-zero BC, yielding 3 successes again.
This control did not establish improvement either. It does not prove reward
learning is irrelevant: it is a small sparse-reward experiment, and does not
isolate insufficient optimization, exploration, controller or dynamics effects.
That control is explicitly not an IRL policy, and its copied LFIRL checkpoint
is used only for post-hoc scoring.
A further LFIRL run restores the original DPPO scale of 1000 environments,
batch size 17600 and macro discount 0.999, with 20 iterations. It retains the
same candidate reward, original BC/DDIM, and 0.1 BC anchor. No successful-replay
BC updates are enabled. This tests data scale, not a new IRL loss. The original
DPPO configuration specifies 1000 iterations, so these short trials do not
establish convergence or rule out learning with a larger training budget.
The initial 1000-environment selection evaluation obtained 2 successes before
any policy update. The first learned-reward PPO update completed with finite
losses. New policy performance remains pending. The paired benchmark seeds
are reused across trials; an apparent gain needs fresh-seed confirmation before
promotion. A finite monitor updates `improve_2026-10-31/progress.json` while the
authorized training/evaluation pipelines run. No checkpoint is promoted
automatically.

The 1000-environment capacity probe completed one eight-action chunk, not a
full task evaluation. A reset audit found identical recorded physics but
approximately 0.000173 differences in derived quaternion components near
zero. The unchanged upstream matrix conversion takes square roots of trace
expressions, amplifying float32 cancellation to order sqrt(epsilon). The
evaluator now budgets that numerical effect only on derived quaternion
features, while retaining the independent 1e-6 physical-state check and
10-micrometer position check. Both comparison policies receive the same
initial measurement. The observation converter and simulation dynamics were
not changed. Seventeen CPU tests cover the current interface and safeguards.

Assembly improvement remains unestablished.
The matched reports and exact experiment scripts are preserved under
`log/furniture-lfirl/calibrated_2026-10-31/`.

The earlier **f-IRL JS with DPPO inner solver** remains an archived experimental
path. Keeping its covariance loss did not establish the MaxEnt assumptions
required after replacing SAC. Its checks only validated wiring, not reward
recovery or assembly improvement. It is not the recommended implementation.

The original non-importance-sampled path receives `(state_trajectories, None,
None)` for samples. No expert-policy likelihood, generator likelihood, sum of
denoising log probabilities or learned action-density surrogate is substituted.
The native DPPO optimizer still uses its own original denoising probabilities
for its policy-gradient loss; those are not supplied to the reward learner.

Sources inspected:

- [f-IRL paper](https://arxiv.org/abs/2011.04709) and
  [official code](https://github.com/twni2016/f-IRL).
- [DPAIL paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/0939f13ffce3ff487509d902ddba4571-Paper-Conference.pdf),
  section 4 and equations 8-10, and `imitation/algo/dpail.py` in its official
  NeurIPS supplementary ZIP. The inspected source uses denoising losses; it is
  not an implementation of exact executable-action log probability.
- [LFIRL paper](https://arxiv.org/html/2609.38955v1) and
  [official code](https://github.com/NothingThrough/LFIRL), including
  `dp_action_score_from_policy` in `train_lfirl.py`.
- [EnergyFlow paper](https://arxiv.org/html/2605.00623v2), especially its
  conservative energy parameterization.

The current task-level reward does not yet solve the common-skill objective.
The old labels were deleted. Reconstructing reviewed semantic annotations and
agreeing on a shared representation of objects/goals is necessary before
reporting Push/Grasp/Insert/Rotate/Release transfer between different tasks.
