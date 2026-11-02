# DPPO Furniture on IsaacLab

Entry point: `run_lab.py`. Models, PPO agents, Furniture configs, normalizer,
and `env/gym_utils/wrapper/furniture.py` are reused.

On Linux, `run_lab.py` finds the existing Isaac Sim bundled cuSPARSELt/CUPTI
libraries and adds missing library directories before loading Torch or Kit.
It restarts Python once so the dynamic linker sees the paths, while preserving
existing `LD_LIBRARY_PATH` entries and all command-line arguments. Set
`FURNITURE_ML_BUNDLE` to use a different existing bundle location.

## BC training with IsaacLab evaluation

Select the existing pretraining config with the same launcher:

```bash
./isaaclab.sh -p ../dppo/script/isaaclab/run_lab.py \
  --config-dir cfg/furniture/pretrain/lamp_low \
  --config-name pre_diffusion_mlp \
  --eval-envs 4 \
  wandb=null train.save_model_freq=100
```

The existing `TrainDiffusionAgent` trains from the existing normalized position
demonstrations. Its loss and optimizer loop are reused. At each checkpoint,
the EMA policy executes a full episode in the real IsaacLab environment through
the existing Furniture wrapper. Evaluation uses the Furniture config's DDIM
sampling; this does not change the diffusion training loss. Training and
evaluation run in the same process. The viewer remains responsive during BC
and shows robot motion during evaluation.

The default training length is 8000 epochs. `train.save_model_freq` also sets
the simulation evaluation interval; the final epoch is always saved/evaluated.
`isaaclab_eval.jsonl` stores actual rollout results, and `checkpoint/best.pt`
selects by success rate, then reward, then latest epoch on ties. Checkpoints
retain the original BC `model` / `ema` format. The paired `normalization.pth`
comes from the same directory as `train.npz`. No new demonstrations are created.

Use `one_leg_low` or `round_table_low` for the other tasks. Add `--headless`
for training/evaluation without the viewer, or `--eval-envs 1` for one robot.

## Fine-tuning

All Lab backend code lives in
`external/furniture-bench/furniture_bench/isaaclab/`. Environment class names
stay `FurnitureSimEnv` and `FurnitureRLSimEnv`; filenames add `_lab`.
The local original `envs/furniture_sim_env.py`, `sim_config.py` and
`controllers/control_utils.py` are restored to their original contents.

The RL backend follows the DPPO fork's absolute pose / row-6D actions,
least-squares DiffIK (one reference per action), arm gains 1000/200, effort
gripper, nominal task poses and force-based reset. Source reference:
[ankile/furniture-bench, 52e7c686](https://github.com/ankile/furniture-bench/tree/52e7c686a16b10615539ec480694ef95f69dfe43).
Collision meshes use SDF resolution 512. Primitive colliders keep their analytic
shape. These overrides are applied on the running stage after URDF conversion.

```bash
conda activate dppo-furniture
export DPPO_DATA_DIR="$HOME/develop/dppo/data"
export DPPO_LOG_DIR="$HOME/develop/dppo/log"
export FURNITURE_ML_BUNDLE="$HOME/develop/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle"
export LD_LIBRARY_PATH="$FURNITURE_ML_BUNDLE/cusparselt/lib:$FURNITURE_ML_BUNDLE/nvidia/cuda_cupti/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd ~/develop/IsaacLab
./isaaclab.sh -p ../dppo/script/isaaclab/run_lab.py \
  --config-dir cfg/furniture/finetune/lamp_low \
  --config-name ft_ppo_diffusion_mlp \
  wandb=null env.n_envs=4 train.batch_size=256 train.n_train_itr=20
```

Select `one_leg_low` or `round_table_low` using the same config name. If the
checkpoint is elsewhere, add `base_policy_path=/absolute/path/to/state_N.pt`.
No files are downloaded automatically. Add `--headless` to hide the viewer.

For a BC-only rollout using the existing training agent, set
`train.n_train_itr=1`: iteration zero is evaluation and performs no PPO update.
Numerical equivalence of the two physics engines and pretrained-policy grasp
success have not been established; successful startup is not a grasp test.

## View every recorded NPZ demonstration

From `~/develop/IsaacLab`, run:

```bash
./isaaclab.sh -p ../dppo/script/isaaclab/view_npz.py \
  --data data/furniture/one_leg_low/train.npz \
  --task one_leg --mode states --rendering_mode quality
```

This plays every nonempty demonstration in order, with Previous, Next,
Pause/Resume and Restart controls. The current file has 50 trajectory slots,
49 nonempty demonstrations and 21,987 valid frames. The all-zero slot is
reported and skipped. Default playback is 10 recorded frames per second;
`--speed 2` doubles playback speed, and `--loop` repeats the dataset.

State playback sets recorded object and front-obstacle poses directly and
reconstructs the robot arm using IsaacLab Differential IK. The NPZ stores TCP
pose and gripper width, not arm joint angles, so its exact elbow posture is
unavailable. A red sphere shows the recorded TCP, and the panel displays the
reconstruction error. This visualization is not a physical assembly test.

Use `--mode actions` to execute the NPZ commands through the existing physical
environment, without any BC policy or IRL updates. This initializes objects
from the recorded first state and reconstructs the initial arm by IK, then
lets physics evolve without resetting objects each frame. Since original
joint angles and object velocities are missing, it is an approximate action
replay, not an exact simulator-state restoration. Its red sphere represents
the commanded TCP target. `--report /absolute/path/report.json` records replay
errors and, in action mode, environment assembly returns.

## Diffusion-policy IRL: original LFIRL recovery, then native DPPO

Use `ft_lfirl_diffusion_mlp` for the experimental, source-based diffusion-policy
IRL path. Its wiring has been checked; Furniture reward quality and assembly
improvement have not been established.
The official [LFIRL implementation](https://github.com/NothingThrough/LFIRL)
is pinned under `external/lfirl`. Its Q-gradient matching, initial Gumbel value
fit, state-dependent offset calibration, calibrated value fit, reward regression
and score routine are loaded unchanged. `external/LFIRL_SOURCES.json` records
the exact source files and adaptations. Selective loading only excludes the
original script's unrelated PushT/Minari/MuJoCo imports and environment launcher.

The existing Furniture BC denoiser and its original noise schedule provide the
frozen diffusion score. Eight commands form **one 80-dimensional macro action**;
the adapter never repeats a 10-dimensional first action and calls that the
whole chunk's score. NPZ `(state_t, actions_t:t+8, state_t+8)` transitions stay
within demonstrations and padding-free segments. Their 0.8-second time step and
discount match the online DPPO step. Reward inputs are the pre-action state and
the entire commanded action chunk, in the original DPPO normalization.

From `~/develop/IsaacLab`, recover the reward offline:

```bash
./isaaclab.sh -p ../dppo/script/isaaclab/train_lfirl.py \
  --task one_leg --device cuda:0 \
  --terminal-mode demonstration_end \
  --output log/furniture-lfirl/offline_one_leg
```

The NPZ has no terminal flags. `demonstration_end` explicitly adopts the
upstream missing-done fallback: a demonstration endpoint is treated as terminal.
The last complete action window is aligned to its final observed state.
This is an assumption, not a recorded termination label. The alternative
`--terminal-mode truncate` uses only known interior transitions without terminal
anchors. Neither choice proves that the reconstructed MDP exactly matches the
original collector. The selected convention is stored in `report.json` and
`reward.pt`. `--max-batches-per-stage 5` is a limited integration check only.

The following is the online fine-tuning command once a reward candidate has
been assessed. It is not evidence that the default candidate improves assembly:

```bash
./isaaclab.sh -p ../dppo/script/isaaclab/run_lab.py \
  --config-dir cfg/furniture/finetune/one_leg_low \
  --config-name ft_lfirl_diffusion_mlp \
  --rendering_mode quality \
  wandb=null env.n_envs=32 train.batch_size=2048 \
  +train.logprob_batch_size=8192 train.n_train_itr=100
```

Here the recovered reward stays fixed. Native DPPO does not replace a MaxEnt
solver inside the reward-gradient derivation: the offline LFIRL stages have
already finished. This downstream optimizer differs from the reference code's
SAC evaluation, so Furniture performance must still be evaluated. The original
BC/DDIM policy, successful-episode auxiliary BC, frozen-BC anchoring and
evaluation-based best-model selection are retained. Logs are under
`log/furniture-lfirl`; paired best files are `best.pt` and `lfirl_best.pt`.
The old running f-IRL process is not automatically stopped or converted.

## Controller calibration and BC diagnostics (2026-10-30)

`evaluate_lfirl.py` runs existing checkpoints without optimizer updates. It
records applied actuator properties, physical initial states and actual assembly
events. `--restore-reference` restores a saved physical/control snapshot before
a candidate rollout; `--override` accepts an existing Hydra override. Reports
are under `log/furniture-lfirl/performance_2026-10-30/`.

The factory now forwards requested stiffness/damping instead of replacing them
with 1000/200. Its defaults are retained. IsaacLab-only calibration options are
`env.specific.gripper_joint_friction`, `env.specific.gripper_velocity_limit` and
`env.specific.legacy_link_armature`. They are unset by default. The last option
adds the value to link inertia diagonals and removes the previous joint-space
armature. Gym's `AssetOptions.armature=0.01` meant link inertia, as explained by
[NVIDIA](https://forums.developer.nvidia.com/t/examples-fail-setting-armature-values/179346).
These options are experimental comparisons, not automatically selected training
settings. Contact friction and SDF resolution are unaffected.

The unchanged BC predicted actions on 490 recorded expert states with median
first-action position error 3.67 mm, rotation error 0.43 degrees and gripper sign
accuracy 99.39%. These are training-data checks, not held-out task success.
In a separate empty-gripper test, the original joint-friction setting closed
only about 1.3 mm in 100 ms. The median first closing displacement in 49 expert
events was about 11.4 mm; expert object-contact conditions differ from that test.
Neither result establishes the sole cause of failed policy rollouts.

Arm-gain changes and gripper-only changes did not establish an assembly
improvement in the tested resets. Link-inertia correction alone yielded 0/64
successes versus the unchanged baseline's 1/64; this small sample does not
establish degradation either. No candidate controller defaults were promoted.
The original LFIRL state-only reward option
was also tested with its original regression loss and frozen Q/V/b networks;
held-out Bellman-target fit was worse than the existing state/action reward.
It was not promoted. Loss/fit, lift proxies and learned return must not be
reported as assembly improvement. Initial BC and prior training checkpoints
remain preserved.

`check_lfirl_reward.py` scores expert commands, random alternative commands and
saved physical rollouts. Its outputs are descriptive checks, not invented
acceptance thresholds. Unit/integration success is not an assembly-success
claim. Common skill reward transfer is still unimplemented and requires a
shared state/context representation and reviewed skill annotations.

Local checks on 2026-10-29 passed 15 tests and a complete 700-command-horizon
evaluation/training integration run (88 eight-action chunks, so 704 commands).
The initial policy matched the original BC exactly, trainable policy weights
changed, and frozen BC/reward weights did not. Initial evaluation succeeded in
0/4 environments; this check did not establish policy improvement.

Further fitting checks found that the Kitchen defaults underfit the Furniture
macro-action Q score. `--qhat-passes 400 --q-lr 0.0003` improved Q-score matching
on four excluded demonstrations, but did not automatically yield a well-fitted
reward head. Additional reward regression with the original loss improved
validation Bellman-target R-squared from 0.25 to 0.51 at 1,000 extra updates,
then overfit (below zero at 5,000). These targets come from the recovered
Q/value models, not true environmental rewards. The original candidate and
separate fitting candidates remain under `log/furniture-lfirl`; none has been
promoted based on assembly performance. The validation demonstrations were
used for model selection, not an independent final test.
The reward-regression validation winner also ranked the one available physical
success above only 47% of 88 failed prefixes. This single-success probe has
different resets and cannot estimate performance, but it provides no consistent
evidence of a better success signal. No long online training run has been
started with that candidate. The combined findings are recorded in
`log/furniture-lfirl/verification_2026-10-29.json`.

## Archived experimental path: f-IRL JS with DPPO

Use `ft_firl_diffusion_mlp`. It keeps the original One-leg BC checkpoint,
DDIM sampler, normalizer, simulator wrapper and DPPO optimizer. This path does
not distill or replace the policy, estimate final-action likelihoods, or use
the experimental two-density-model stopping threshold.

The reward learner reuses the original MIT-licensed
[f-IRL code](https://github.com/twni2016/f-IRL) in `external/firl`:
`MLPReward`, `SMMIRLDisc`, and `f_div_disc_loss("js", IS=False, ...)` are
unchanged. `external/FIRL_SOURCES.json` records their commit and file hashes.
The discriminator estimates the expert/policy **state occupancy density ratio**;
the separate reward MLP is updated by the original trajectory covariance loss.
Its output, rather than the discriminator score, trains the policy.

The reference f-IRL implementation uses maximum-entropy SAC. Replacing its inner
solver with native DPPO is an adaptation. In particular, native DPPO does not
establish the exact maximum-entropy inner optimum assumed by the f-IRL
derivation. This implementation makes no equivalent convergence or reward
recovery guarantee. The method selection and alternatives are documented in
`IRL_METHODS.md`.

From `~/develop/IsaacLab`, with the existing conda environment activated:

```bash
./isaaclab.sh -p ../dppo/script/isaaclab/run_lab.py \
  --config-dir cfg/furniture/finetune/one_leg_low \
  --config-name ft_firl_diffusion_mlp \
  --rendering_mode quality \
  wandb=null env.n_envs=4 train.n_train_itr=20
```

Iteration 0 evaluates the unchanged BC policy. Each training iteration collects
the existing fixed-horizon rollouts, trains the state density-ratio classifier,
updates the explicit reward MLP and runs native DPPO with that learned reward.
The original assembly reward is retained for evaluation. Initial actor warmup
continues to use the native `n_critic_warmup_itr=3` setting.

This config also enables the existing DPPO frozen-BC regularizer with
`train.use_bc_loss=true` and `train.bc_loss_coeff=0.1`. Successful training and
evaluation episodes are retained using the actual environment assembly return,
not the learned reward. Only the prefix through the first successful action
chunk is stored, with aligned pre-action observations and commanded action
chunks. Failed rollouts remain in the ordinary on-policy PPO batch.
The native anchor loss is a negative denoising log density and can be negative;
the separate success replay loss below is the original denoising MSE.

`firl.success_replay` controls the separate supervised diffusion BC term:
`capacity=100` active episodes, `batch_size=128`, `updates=4` minibatches per
training iteration, and `coefficient=0.1`. It reuses the original diffusion BC
loss at the trainable denoiser's sampling timesteps. It never treats stored
successes as fresh on-policy PPO samples. Memory is bounded; archived episode
NPZ files remain in `firl/success/`. Empty replay skips this loss until actual
successes occur. Old runs without saved successful trajectories cannot be
reconstructed from policy checkpoints alone.

Each nonempty evaluation may promote `checkpoint/best.pt`, paired
`checkpoint/firl_best.pt`, and `checkpoint/best.json`. Selection uses assembly
success rate, then mean environment assembly return. Equal scores keep the
earlier model, including the initial BC baseline if no model improves. These
are the best observed evaluation results, not a guarantee of monotonic true
success probability. Training rollouts do not promote the best model.

Success replay is included in each `firl_N.pt`. Loading a paired reward
checkpoint restores its replay automatically; alternatively use
`firl.success_replay.checkpoint=/absolute/path/to/firl_N.pt` to restore just
the successful episodes while learning a new reward. Task, normalization and
dimensions must match. `firl/retention.jsonl` and optional W&B `retention/*`
metrics record replay size, new successes, auxiliary BC updates and losses.
Environment assembly rewards still do not enter PPO returns, but now select
the supervised success replay and the best model. This is an additional
success-BC objective alongside f-IRL/DPPO, not the unmodified f-IRL algorithm.

All nonpadding expert states from the selected NPZ demonstrations and all
post-action policy states are eligible for classifier training. The reward
update keeps trajectories intact: for four environments it uses four complete
state sequences, not four isolated states. It needs at least two independent
policy trajectories, so use `env.n_envs>=2`. Rollouts may end together at their
fixed horizon; internal episode resets are rejected instead of splicing
unrelated episodes together. No action probabilities are passed to this loss.

Outputs are native policy `checkpoint/state_N.pt`, paired reward and optimizer
`checkpoint/firl_N.pt`, and `firl/updates.jsonl` with actual gradient norms,
update counts and trajectory sizes. Reward checkpoints can be reloaded with
`firl.reward_checkpoint=/absolute/path/to/firl_N.pt`. Use
`firl.freeze_reward=true` to train a policy under an already learned fixed
reward. `base_policy_path=.../state_N.pt` warm-starts the native policy; this
does not resume its entire optimizer/iteration state.

`firl.max_expert_trajectories=3` selects three entire nonempty demonstrations
with the configured seed, rather than selecting three individual states.
Together with a reward checkpoint this supports a small-demonstration
continuation experiment **within the same reward feature definition**.
Dimension and normalization checks reject incompatible checkpoints. One-leg to
Lamp transfer is not established merely by loading these task-level weights.

The previous skill annotations were deleted. Current `reward_scope=task`
therefore explicitly denotes full-task learning; requesting skill scope fails
instead of silently fabricating labels. Common Push/Grasp/Insert/Rotate/Release
reward transfer still needs reviewed labels and common cross-task state/context
features. A task-level run is not reported as a completed skill-transfer result.

`test_firl.py` checks upstream reuse, learning an expert preference on a known
example, covariance sample requirements, checkpoint restoration and whole-demo
few-shot selection. Actual simulator checks are stored under
`log/furniture-firl/integration_smoke` and
`log/furniture-firl/full_horizon_validation`.

## Archived experimental AIRL bridge

The older `ft_airl_diffusion_mlp` route below is retained for inspecting the
earlier experiment. It is not the recommended IRL command. Its ensemble
disagreement gates are engineering heuristics, not rules from the AIRL paper
and not a verified bound on true diffusion log-likelihood error.

`ft_airl_diffusion_mlp` inherits the One-leg DPPO config and the local BC
checkpoint. The BC network, checkpoint, DDIM forward sampler and denoising PPO
loss are unchanged. The only extension to the existing PPO agent records next
observations and supplies learned rewards before advantage calculation.

Sources are whole local checkouts, with commits, licenses and source hashes in
`external/AIRL_SOURCES.json`:

- [imitation AIRL](https://github.com/HumanCompatibleAI/imitation): the original
  `BasicShapedRewardNet`, `AIRL.logits_expert_is_high` and inherited
  `AdversarialTrainer.train_disc` (including its BCE loss and optimizer step).
  The adapter replaces SB3-specific batch/environment setup; it does not replace
  the AIRL objective. This is the imitation implementation, not the paper
  authors' original TensorFlow repository.
- [nflows](https://github.com/bayesiains/nflows): the original MADE network for
  the separate autoregressive conditional action density. A context-only output
  projection lets the first action coordinate depend on the observation too.

The extra local dependencies are pandas 2.2.3, python-dateutil 2.9.0.post0,
pytz 2025.2 and tzdata 2025.2 in `external/imitation_deps`. The existing IsaacLab
environment provides PyTorch, stable-baselines3 and gymnasium. No IsaacLab
PyTorch/NumPy version was replaced. GAIL is not an alternative training path.

### Why native `exact_likelihood` is not used

`model/diffusion/exact_likelihood.py` integrates an SDE-derived ODE. It does not
integrate the marginal distribution of the actual five-step, clipped DDIM
sampler. Clipping also creates probability masses that an ordinary continuous
ODE density cannot describe. Additional issues in the current implementation
include discrete time potentially reaching index N, different actor selection
for drift and divergence, a default `probability_flow=False`, and a bits/dim
return value instead of joint nats. Changing those details would still not
establish equality to the existing sampler. The file is left untouched.

AIRL needs `log pi(a[0:8] | state)`, the joint log density of the final 8 x 10
normalized action chunk. Denoising-path log probabilities remain native PPO
quantities and are never passed to AIRL. Training rollout density uses the
existing stochastic sampler (`deterministic=False`, minimum std 0.04);
deterministic evaluation is a separate policy mode.

### Qualification before discriminator or policy updates

At each policy version, sample final chunks from the unchanged policy at the
expert and generated states used for the discriminator. Fit two conditional
autoregressive estimators with opposite action orders. Their distribution has
four known clipping atoms per coordinate plus a bounded continuous Gaussian
mixture. Fit monotone CDF calibration on development data. All training,
development and final validation draws are separate.

The default pilot uses two expert states and two generated states, with 8192
training, 4096 development and 8192 validation draws per state. This is a small
integration batch, not a demonstrated setting for high assembly success.
Increasing discriminator state coverage requires fresh density qualification.

Fixed gates check held-out randomized PIT calibration (pooled KS <= 0.04,
conditional KS <= 0.20), PIT correlation <= 0.10, train/test NLL gap <= 2 nats,
finite probabilities, and disagreement between the two joint log densities on
actual discriminator queries (RMSE <= 1 nat, p95 <= 2 nats). The latter is
ensemble disagreement, not error against a known diffusion likelihood. These
checks qualify a numerical surrogate; they cannot prove the unknown true
80-dimensional pointwise likelihood. A separate analytic-distribution test
checks actual log-probability error on a distribution with known density.

On any failure, save `validation.json`, the candidate density and reviewable
checkpoints, then stop before AIRL/PPO updates. No fallback density or threshold
relaxation runs automatically. Validation is limited to registered states and
the current policy version; density qualification repeats after policy updates.

Run qualification without policy or reward updates (after the environment
exports above, from `~/develop/IsaacLab`):

```bash
./isaaclab.sh -p ../dppo/script/isaaclab/run_lab.py \
  --config-dir cfg/furniture/finetune/one_leg_low \
  --config-name ft_airl_diffusion_mlp \
  --rendering_mode quality \
  wandb=null env.n_envs=1 airl.validate_only=true
```

Run the pilot alternating loop, keeping the viewer visible:

```bash
./isaaclab.sh -p ../dppo/script/isaaclab/run_lab.py \
  --config-dir cfg/furniture/finetune/one_leg_low \
  --config-name ft_airl_diffusion_mlp \
  --rendering_mode quality \
  wandb=null env.n_envs=4 train.n_train_itr=20
```

Iteration zero is the unchanged BC evaluation. Training iterations collect
rollouts, qualify the current action density, run original AIRL discriminator
updates, then use its shaped reward `g(s) + gamma*h(s') - h(s)` for DPPO value
and policy updates. Native `n_critic_warmup_itr=3` preserves the initial actor
until iteration 3 (iteration 0 is evaluation). Environment assembly reward
remains an evaluation metric. The viewer does not move the robot while the
separate density model is fitting. This preserves DPPO's policy optimizer; it
does not claim equivalence to an exact maximum-entropy AIRL policy optimizer.

Outputs include native `checkpoint/state_N.pt`, paired
`checkpoint/airl_N.pt` (g/h, reward optimizer, validation metadata and the
unshaped g reward), `airl/updates.jsonl`, and per-iteration density reports.
`airl.reward_checkpoint=/absolute/path/to/airl_N.pt` restores the reward network
and its optimizer; `base_policy_path=.../state_N.pt` warm-starts the policy.
This does not restore the entire native DPPO optimizer/iteration state.

This first bridge is task-level One-leg AIRL. It does not add a skill label
input, invent labels for DPPO NPZ files, or establish cross-task reward transfer.
The exported g reward is available for subsequent transfer work.

### Recorded integration checks

`log/furniture-airl/integration_smoke/integration_verification.json` records a
real 16-action IsaacLab run: density qualification passed, upstream AIRL updated
once, the fine-tuned actor and critic changed, and the frozen BC actor did not.
`log/furniture-airl/integration_confirmation/integration_verification.json`
records two further iterations with seed 123, including qualification after
the policy changed. These are short integration checks, not assembly-success
or learned-reward-transfer results.

`script/isaaclab/test_airl.py` checks source-method reuse, discriminator updates,
mixed distribution normalization, autoregressive masking, state conditioning,
CDF calibration normalization and validation guards. The separate CPU script
`script/isaaclab/check_airl_density.py --output /absolute/path/report.json`
fits a known analytic mixed distribution and measures its actual log-probability
error on 8000 held-out examples. Run both in the same Python environment and
library paths as the AIRL launcher. Its report is also saved at
`log/furniture-airl/integration_smoke/analytic_density_validation.json`.
