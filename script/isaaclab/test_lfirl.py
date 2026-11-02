"""Check original LFIRL reuse and the native macro-action interfaces."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.finetune.lfirl import official
from agent.finetune.lfirl.dataset import MacroTransitions
from agent.finetune.lfirl.policy import MacroDiffusionPolicy, NativeActionPolicy
from agent.finetune.lfirl.reward import LFIRLReward, default_settings
from agent.finetune.train_lfirl_diffusion_agent import TrainLFIRLCurriculumAgent, TrainLFIRLDiffusionAgent


class GaussianNoise(nn.Module):
    def forward(self, action, time, cond):
        sigma = torch.tensor([.1, .4])[time].view(-1, 1, 1)
        return sigma * action


class GaussianBC(nn.Module):
    horizon_steps, action_dim, obs_dim, denoising_steps = 2, 2, 3, 2
    predict_epsilon = True

    def __init__(self):
        super().__init__()
        self.actor = GaussianNoise()
        self.register_buffer("alphas_cumprod", torch.tensor([.99, .84]))

    def q_sample(self, actions, time, noise):
        alpha = self.alphas_cumprod[time].view(-1, 1, 1)
        return alpha.sqrt() * actions + (1 - alpha).sqrt() * noise


class LFIRLTests(unittest.TestCase):
    def test_success_replay_rejects_old_gripper_semantics(self):
        from types import SimpleNamespace
        from omegaconf import OmegaConf
        from agent.finetune.firl.success import SuccessfulRolloutBuffer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buffer = SuccessfulRolloutBuffer(root / "source", 2, (1, 58), (8, 10), "one_leg", "normalizer")
            path = root / "replay.pt"
            payload = dict(success_replay=buffer.state_dict(), policy_gripper_command_memory="accepted_command")
            torch.save(payload, path)
            agent = TrainLFIRLDiffusionAgent.__new__(TrainLFIRLDiffusionAgent)
            agent.settings = OmegaConf.create(dict(success_replay=dict(
                enabled=True, capacity=2, batch_size=1, updates=1, coefficient=.1, checkpoint=str(path))))
            agent.retention_dir = root / "runtime"
            agent.horizon_steps = agent.act_steps = 8
            agent.n_cond_step, agent.obs_dim, agent.action_dim = 1, 58, 10
            agent.normalizer_hash = "normalizer"
            agent.reward_metadata = {"policy_gripper_command_memory": "upstream_raw_action"}
            with self.assertRaisesRegex(ValueError, "gripper command semantics"):
                agent.initialize_retention(SimpleNamespace(env_name="one_leg"))
            payload["policy_gripper_command_memory"] = "upstream_raw_action"
            torch.save(payload, path)
            agent.initialize_retention(SimpleNamespace(env_name="one_leg"))
            self.assertEqual(len(agent.success_replay), 0)

    def test_frozen_native_eta_probe_includes_exact_one_without_infinite_parameters(self):
        from script.isaaclab.evaluate_lfirl import fixed_eta_for_probe

        condition = {"state": torch.zeros(3, 1, 58)}
        for value in (.0001, .2, 1.):
            eta = fixed_eta_for_probe(value, "cpu")
            self.assertTrue(torch.isfinite(eta.eta_logit).all())
            torch.testing.assert_close(eta(condition), torch.full((3, 1), value), atol=1e-7, rtol=0)
        for value in (0., -1., 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fixed_eta_for_probe(value, "cpu")

    def test_native_reward_control_uses_original_environment_reward_only(self):
        from script.isaaclab.probe_lfirl_advantages import NativeRewardCurriculumControl

        agent = NativeRewardCurriculumControl.__new__(NativeRewardCurriculumControl)
        rewards = np.array([[0., 1.], [1., 0.]], np.float32)
        actual = agent.get_training_rewards(None, None, None, rewards, np.zeros_like(rewards))
        np.testing.assert_array_equal(actual, rewards)

    def test_curriculum_outcomes_do_not_evict_full_task_success_replay(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            agent = TrainLFIRLCurriculumAgent.__new__(TrainLFIRLCurriculumAgent)
            agent.curriculum_mask = np.array([False, True, True, False])
            agent.curriculum_success_replay = None
            agent.retention_dir = Path(directory)
            agent.itr = 0
            agent.best_reward_threshold_for_success = 1
            stored = []
            agent.success_replay = SimpleNamespace(add=lambda obs, actions, metadata: stored.append(metadata))
            observations = np.zeros((3, 4, 1, 58), np.float32)
            chains = np.zeros((3, 4, 6, 8, 10), np.float32)
            rewards = np.zeros((3, 4), np.float32)
            rewards[1] = 1.
            agent.on_rollout_collected(observations, chains, rewards,
                                       [(index, 0, 2) for index in range(4)], False)
            self.assertEqual([row["env_id"] for row in stored], [0, 3])
            self.assertEqual(agent._new_successes, 2)
            np.testing.assert_array_equal(agent.probe_native_rewards, rewards)
            local = []
            agent.curriculum_success_replay = SimpleNamespace(
                add=lambda obs, actions, metadata: local.append(metadata))
            agent.curriculum_donor_ids = np.array([-1, 4, 5, -1])
            agent.on_rollout_collected(observations, chains, rewards,
                                       [(index, 0, 2) for index in range(4)], False)
            self.assertEqual([row["env_id"] for row in stored], [0, 3, 0, 3])
            self.assertEqual([row["donor_id"] for row in local], [4, 5])
            self.assertTrue(all(row["source"] == "real_curriculum_training_success" for row in local))

    def test_curriculum_success_batch_mixes_separate_buffers_and_saves_both_roles(self):
        class Buffer:
            def __init__(self, value):
                self.value, self.requests = value, []

            def __len__(self):
                return 2

            def sample(self, batch, device):
                self.requests.append(batch)
                return (torch.full((batch, 1, 58), self.value),
                        torch.full((batch, 8, 10), self.value))

            def state_dict(self):
                return dict(marker=self.value)

        agent = TrainLFIRLCurriculumAgent.__new__(TrainLFIRLCurriculumAgent)
        agent.device = "cpu"
        agent.success_replay, agent.curriculum_success_replay = Buffer(1.), Buffer(2.)
        agent.curriculum_replay_fraction = .3
        observations, actions = agent.sample_success_batch(10)
        self.assertEqual(agent.success_replay.requests, [7])
        self.assertEqual(agent.curriculum_success_replay.requests, [3])
        self.assertEqual(tuple(observations.shape), (10, 1, 58))
        self.assertEqual(tuple(actions.shape), (10, 8, 10))
        self.assertTrue(torch.all(observations[:7] == 1.))
        self.assertTrue(torch.all(observations[7:] == 2.))
        self.assertEqual(agent.additional_retention_state(), dict(curriculum_success_replay=dict(marker=2.)))
        agent.curriculum_success_replay = None
        agent.sample_success_batch(10)
        self.assertEqual(agent.success_replay.requests, [7, 10])
        self.assertEqual(agent.additional_retention_state(), {})

    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)

    def test_reward_bridge_rejects_different_teacher_and_fitted_discount(self):
        from copy import deepcopy
        from agent.finetune.lfirl.reward import validate_reward_provenance

        saved = dict(
            normalization_sha256="normalizer", source_bc_sha256="teacher",
            gamma=.999, learner={"settings": {"irl": {"gamma": .999}}},
        )
        validate_reward_provenance(saved, "normalizer", "teacher")
        with self.assertRaisesRegex(ValueError, "different BC"):
            validate_reward_provenance(saved, "normalizer", "other_teacher")
        with self.assertRaisesRegex(ValueError, "normalization"):
            validate_reward_provenance(saved, "other_normalizer", "teacher")
        inconsistent = deepcopy(saved)
        inconsistent["learner"]["settings"]["irl"]["gamma"] = .99
        with self.assertRaisesRegex(ValueError, "discount"):
            validate_reward_provenance(inconsistent, "normalizer", "teacher")

    def test_vectorized_denoising_discount_preserves_native_policy_loss_and_gradient(self):
        from types import SimpleNamespace
        from model.diffusion.diffusion_ppo import PPODiffusion

        parameter = torch.tensor(0., requires_grad=True)
        indices = torch.tensor([0, 1, 3, 4])
        advantages = torch.tensor([-1., .5, 2., .8])
        reference = torch.tensor([.9 ** (5 - index - 1) for index in indices])
        expected = -(advantages * reference).mean()
        probe = SimpleNamespace(
            device="cpu", gamma_denoising=.9, ft_denoising_steps=5, norm_adv=False,
            clip_advantage_lower_quantile=0., clip_advantage_upper_quantile=1.,
            clip_ploss_coef_base=.001, clip_ploss_coef=.001, clip_ploss_coef_rate=3.,
            logprob_clip_min=-5., logprob_clip_max=2.,
            clip_vloss_coef=None, critic=lambda obs: torch.zeros(4),
            get_logprobs_subsample=lambda *args, **kwargs: (
                parameter.expand(4, 1, 1), torch.ones(4, 1),
            ),
        )
        probe.clip_logprobs = PPODiffusion.clip_logprobs.__get__(probe)
        losses = PPODiffusion.loss(
            probe, {}, None, None, indices, torch.zeros(4), torch.zeros(4),
            advantages.clone(), torch.zeros(4, 1, 1), reward_horizon=1,
        )
        gradient, = torch.autograd.grad(losses[0], parameter)
        torch.testing.assert_close(losses[0], expected, rtol=0, atol=1e-7)
        torch.testing.assert_close(gradient, expected, rtol=0, atol=1e-7)

    def test_reward_bridge_rejects_transposed_axes_wrong_action_size_and_nan(self):
        class Probe:
            def predict(self, states, actions):
                return np.zeros(len(states), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            agent = TrainLFIRLDiffusionAgent.__new__(TrainLFIRLDiffusionAgent)
            agent.obs_dim, agent.n_cond_step = 3, 1
            agent.horizon_steps, agent.action_dim = 2, 2
            agent.reward, agent.retention_dir = Probe(), Path(directory)
            agent.itr, agent.success_replay = 0, None
            states = np.zeros((3, 2, 1, 3), dtype=np.float32)
            chains = np.zeros((3, 2, 3, 2, 2), dtype=np.float32)
            rewards = np.zeros((3, 2), dtype=np.float32)
            actual = agent.get_training_rewards(states, states, chains, rewards, rewards)
            self.assertEqual(actual.shape, rewards.shape)
            with self.assertRaisesRegex(ValueError, "rollout states"):
                agent.get_training_rewards(states.transpose(1, 0, 2, 3), states, chains, rewards, rewards)
            with self.assertRaisesRegex(ValueError, "horizon/action"):
                agent.get_training_rewards(states, states, chains.reshape(3, 2, 3, 1, 4), rewards, rewards)
            chains[0, 0, -1, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                agent.get_training_rewards(states, states, chains, rewards, rewards)

    def test_legacy_resume_restores_policy_reward_and_iteration_without_replaying_warmup(self):
        from util.scheduler import CosineAnnealingWarmupRestarts

        with tempfile.TemporaryDirectory() as directory:
            agent = TrainLFIRLDiffusionAgent.__new__(TrainLFIRLDiffusionAgent)
            agent.device, agent.logdir = "cpu", directory
            agent.n_train_itr, agent.n_critic_warmup_itr = 50, 1
            agent.normalizer_hash, agent.reward_gamma = "normalizer", .999
            agent.source_bc_hash = "teacher"
            agent.model, agent.reward = nn.Linear(2, 2), nn.Linear(2, 1)
            agent.model.ft_denoising_steps_d = 0
            agent.model.ft_denoising_steps_t = 0
            agent.model.min_sampling_denoising_std = .04
            agent.success_replay, agent.learn_eta = None, False
            for name in ("actor", "critic"):
                optimizer = torch.optim.AdamW(agent.model.parameters(), lr=.001)
                scheduler = CosineAnnealingWarmupRestarts(
                    optimizer, first_cycle_steps=10000, max_lr=.001,
                    min_lr=.00001, warmup_steps=10,
                )
                setattr(agent, f"{name}_lr_scheduler", scheduler)
            policy = {key: value.clone() for key, value in agent.model.state_dict().items()}
            reward = {key: value.clone() for key, value in agent.reward.state_dict().items()}
            checkpoint = Path(directory) / "state_10.pt"
            torch.save(dict(itr=10, model=policy), checkpoint)
            companion = dict(
                iteration=10, normalization_sha256="normalizer", gamma=.999,
                source_bc_sha256="teacher",
                learner=reward, best_evaluation={"iteration": 10, "success_rate": .003},
            )
            torch.save(companion, Path(directory) / "lfirl_10.pt")
            with torch.no_grad():
                agent.model.weight.add_(1)
                agent.reward.weight.add_(1)
            agent.resume_training(checkpoint)
            self.assertEqual(agent.itr, 11)
            self.assertEqual(agent.actor_lr_scheduler.last_epoch, 10)
            self.assertEqual(agent.critic_lr_scheduler.last_epoch, 11)
            self.assertEqual(agent.best_evaluation, companion["best_evaluation"])
            for key, value in policy.items():
                torch.testing.assert_close(agent.model.state_dict()[key], value, rtol=0, atol=0)
            for key, value in reward.items():
                torch.testing.assert_close(agent.reward.state_dict()[key], value, rtol=0, atol=0)
            agent.n_train_itr = 11
            with self.assertRaises(ValueError):
                agent.resume_training(checkpoint)
            agent.n_train_itr = 50
            agent.reward_metadata = {"preference_calibration": {"parent_sha256": "other"}}
            with self.assertRaisesRegex(ValueError, "preference-calibrated"):
                agent.resume_training(checkpoint)
            agent.reward_metadata = {}
            agent.reward_metadata = {"policy_gripper_command_memory": "upstream_raw_action"}
            with self.assertRaisesRegex(ValueError, "gripper command semantics"):
                agent.resume_training(checkpoint)
            agent.reward_metadata = {}
            companion["training_reward_source"] = "native_environment_reward_control"
            torch.save(companion, Path(directory) / "lfirl_10.pt")
            with self.assertRaisesRegex(ValueError, "reward source"):
                agent.resume_training(checkpoint)
            companion.pop("training_reward_source")
            companion["normalization_sha256"] = "different"
            torch.save(companion, Path(directory) / "lfirl_10.pt")
            with self.assertRaises(ValueError):
                agent.resume_training(checkpoint)

    def test_optimizer_scheduler_and_reward_scaler_resume_preserves_next_update(self):
        from util.reward_scaling import RunningRewardScaler
        from util.scheduler import CosineAnnealingWarmupRestarts

        def build():
            agent = TrainLFIRLDiffusionAgent.__new__(TrainLFIRLDiffusionAgent)
            agent.itr, agent.n_envs, agent.learn_eta = 3, 2, False
            agent.reward_scale_running = True
            agent.running_reward_scaler = RunningRewardScaler(2)
            agent.model = nn.ModuleDict({"actor_ft": nn.Linear(2, 2), "critic": nn.Linear(2, 1)})
            agent.model.ft_denoising_steps_cnt = 4
            for name, network in (("actor", agent.model.actor_ft), ("critic", agent.model.critic)):
                optimizer = torch.optim.AdamW(network.parameters(), lr=.001)
                scheduler = CosineAnnealingWarmupRestarts(
                    optimizer, first_cycle_steps=100, max_lr=.001,
                    min_lr=.00001, warmup_steps=2,
                )
                setattr(agent, f"{name}_optimizer", optimizer)
                setattr(agent, f"{name}_lr_scheduler", scheduler)
            return agent

        def update(agent):
            inputs = torch.tensor([[.2, .3], [.4, .5]])
            agent.actor_optimizer.zero_grad()
            agent.critic_optimizer.zero_grad()
            loss = agent.model.actor_ft(inputs).square().mean() + agent.model.critic(inputs).square().mean()
            loss.backward()
            for name in ("actor", "critic"):
                getattr(agent, f"{name}_optimizer").step()
                getattr(agent, f"{name}_lr_scheduler").step()

        original = build()
        update(original)
        original.running_reward_scaler(np.array([[1., 2.], [3., 4.]]), np.array([[1., 0.], [1., 0.]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.pt"
            torch.save(original.capture_training_state(), path)
            state = torch.load(path, weights_only=True)
        restored = build()
        restored.model.load_state_dict(original.model.state_dict())
        restored.restore_training_state(state, completed=3)
        probe = np.array([[2., 1.], [4., 3.]])
        first = np.zeros_like(probe)
        np.testing.assert_array_equal(
            original.running_reward_scaler(probe, first), restored.running_reward_scaler(probe, first),
        )
        update(original)
        update(restored)
        for key, value in original.model.state_dict().items():
            torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            restored.restore_training_state(state, completed=2)

    def test_reward_probe_uses_actual_micro_states_and_rejects_missing_states(self):
        from script.isaaclab.check_lfirl_reward import rollout_transitions

        actions = np.arange(48).reshape(2, 3, 4, 2)
        states = np.arange(120).reshape(2, 3, 4, 5)
        raw = {"actions": actions, "micro_pre_states": states}
        observed, commands = rollout_transitions(raw, horizon=1)
        np.testing.assert_array_equal(observed[1, 5], states[1, 1, 1])
        np.testing.assert_array_equal(commands[1, 5], actions[1, 1, 1])
        with self.assertRaises(ValueError):
            rollout_transitions({"actions": actions, "pre_states": states[:, :, 0]}, horizon=1)

    def test_restored_comparison_shares_measurement_and_rejects_different_physics(self):
        from script.isaaclab.evaluate_lfirl import paired_initial_observation

        baseline = np.zeros((2, 58), dtype=np.float32)
        recomputed = baseline.copy()
        recomputed[0, 3] = .00035
        physics = np.zeros((2, 100), dtype=np.float32)
        actual, difference = paired_initial_observation(
            recomputed, physics, (baseline, physics), restored=True,
        )
        np.testing.assert_array_equal(actual, baseline)
        self.assertGreater(difference, .0003)
        self.assertEqual(recomputed[0, 3], np.float32(.00035))
        with self.assertRaises(RuntimeError):
            paired_initial_observation(recomputed, physics + .001, (baseline, physics), restored=True)
        with self.assertRaises(RuntimeError):
            paired_initial_observation(recomputed + .1, physics, (baseline, physics), restored=True)

    def test_lift_proxy_rejects_upright_rest_open_fingers_and_relative_slip(self):
        from script.isaaclab.evaluate_lfirl import grasp_lift_proxy

        height = np.full((1, 5), .08)
        distance = np.full((1, 5), .02)
        width = np.full((1, 5), .03)
        relative = np.zeros((1, 5, 3))
        self.assertEqual(grasp_lift_proxy(height, distance, width, relative), [True])
        self.assertEqual(grasp_lift_proxy(height * .5, distance, width, relative), [False])
        self.assertEqual(grasp_lift_proxy(height, distance, width * 2, relative), [False])
        self.assertEqual(grasp_lift_proxy(height, distance * 5, width, relative), [False])
        relative[0, :, 0] = np.arange(5) * .02
        self.assertEqual(grasp_lift_proxy(height, distance, width, relative), [False])

    def test_restoration_checks_physical_units_without_loosening_state_match(self):
        from script.isaaclab.evaluate_lfirl import paired_initial_observation

        baseline = np.zeros((2, 58), dtype=np.float32)
        recomputed = baseline.copy()
        recomputed[0, 20] = .00112
        physics = np.zeros((2, 100), dtype=np.float32)
        scale = np.ones(58, dtype=np.float32)
        scale[20] = .0001
        actual, _ = paired_initial_observation(
            recomputed, physics, (baseline, physics), restored=True, observation_scale=scale,
        )
        np.testing.assert_array_equal(actual, baseline)
        # Identical normalized error can mean millimeters on a wide-range
        # feature. That must fail, rather than raising the normalized threshold.
        with self.assertRaises(RuntimeError):
            paired_initial_observation(
                recomputed, physics, (baseline, physics), restored=True,
                observation_scale=np.ones(58),
            )
        with self.assertRaises(RuntimeError):
            paired_initial_observation(
                recomputed, physics + .00001, (baseline, physics), restored=True,
                observation_scale=scale,
            )

    def test_square_root_quaternion_roundoff_keeps_position_and_physics_checks_strict(self):
        from script.isaaclab.evaluate_lfirl import paired_initial_observation

        baseline = np.zeros((1, 58), dtype=np.float32)
        recomputed = baseline.copy()
        # sqrt(trace) with one float32 epsilon before the fork's 1/2 factor.
        numerical_error = .5 * np.sqrt(np.finfo(np.float32).eps)
        recomputed[0, 35] = numerical_error
        physics = np.zeros((1, 100), dtype=np.float32)
        actual, _ = paired_initial_observation(
            recomputed, physics, (baseline, physics), restored=True,
            observation_scale=np.ones(58),
        )
        np.testing.assert_array_equal(actual, baseline)
        # The quaternion budget must not permit equal-sized position errors.
        recomputed[0, 30] = numerical_error
        with self.assertRaises(RuntimeError):
            paired_initial_observation(
                recomputed, physics, (baseline, physics), restored=True,
                observation_scale=np.ones(58),
            )
        recomputed[0, 30] = 0
        recomputed[0, 35] = .002
        with self.assertRaises(RuntimeError):
            paired_initial_observation(
                recomputed, physics, (baseline, physics), restored=True,
                observation_scale=np.ones(58),
            )

    def test_upstream_functions_are_loaded_from_original_files(self):
        for name in ("train_qhat_LFIRL_step", "train_b_offset_step", "train_v_function_step", "train_reward_function_step"):
            self.assertEqual(Path(getattr(official, name).__code__.co_filename), official.SOURCE / "train_lfirl.py")
        self.assertEqual(Path(official.score_first_action.__wrapped__.__code__.co_filename), official.SOURCE / "models/diffusion_model.py")

    def test_macro_score_covers_all_action_coordinates(self):
        bc = GaussianBC()
        policy = MacroDiffusionPolicy(bc)
        actions = torch.tensor([[[.1, .2, .5, .9]], [[-.8, .4, .2, -.1]]])
        obs = torch.zeros(2, 1, 3)
        torch.manual_seed(13)
        score = policy.score_first_action(obs, actions, t_eval=0)
        torch.manual_seed(13)
        noisy = bc.q_sample(actions.reshape(2, 2, 2), torch.zeros(2, dtype=torch.long), torch.randn_like(actions).reshape(2, 2, 2))
        expected = -(.99 ** .5) * noisy.reshape(2, 4) * (.1 / ((1 - bc.alphas_cumprod[0]).sqrt() + 1e-8))
        torch.testing.assert_close(score, expected)
        self.assertEqual(tuple(score.shape), (2, 4))
        self.assertFalse(torch.equal(score[:, :2], score[:, 2:]))

    def test_score_noise_averaging_reuses_upstream_monte_carlo(self):
        policy = NativeActionPolicy(GaussianBC(), score_noise_samples=4)
        actions = torch.tensor([[[.1, .2], [.1, .2]]])
        obs = torch.zeros(1, 1, 3)
        torch.manual_seed(13)
        actual = policy.score_first_action(obs, actions, t_eval=0)
        torch.manual_seed(13)
        expected = official.score_first_action(policy, obs, actions, t_eval=0, n_noise=4)
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            NativeActionPolicy(GaussianBC(), score_noise_samples=0)

    def test_reward_batch_shuffle_preserves_complete_aligned_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            states = np.arange(54, dtype=np.float32).reshape(18, 3) + 1
            actions = np.arange(36, dtype=np.float32).reshape(18, 2) + 1
            np.savez(path, states=states, actions=actions, traj_lengths=np.array([9, 9]))
            data = MacroTransitions(path, horizon=2, window_stride=1)
            shuffled = list(data.batches(4, "cpu", shuffle=True,
                                        generator=torch.Generator().manual_seed(17)))
            columns = [torch.cat([batch[index] for batch in shuffled]) for index in range(4)]
            self.assertFalse(torch.equal(columns[0], data.observations))
            expected = {int(obs[0, 0]): index for index, obs in enumerate(data.observations)}
            mapped = torch.tensor([expected[int(obs[0, 0])] for obs in columns[0]])
            self.assertEqual(len(set(mapped.tolist())), len(data.actions))
            for actual, original in zip(columns, (data.observations, data.actions,
                                                  data.next_observations, data.terminals)):
                torch.testing.assert_close(actual, original[mapped], rtol=0, atol=0)

    def test_reference_regression_excludes_whole_episodes_and_ignores_success_labels(self):
        import json
        from agent.finetune.lfirl.dataset import ReferenceRewardTransitions
        from agent.finetune.lfirl.reward import sha256

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "train.npz"
            np.savez(path, states=np.ones((18, 3), dtype=np.float32),
                     actions=np.ones((18, 2), dtype=np.float32), traj_lengths=np.array([9, 9]))
            expert = MacroTransitions(path, horizon=2, window_stride=1)
            normalizer, teacher = root / "normalizer", root / "teacher"
            normalizer.write_bytes(b"normalizer")
            teacher.write_bytes(b"teacher")
            (root / "report.json").write_text(json.dumps(dict(completed=True, bc_matches_original_checkpoint=True)))
            (root / "evaluation_config.yaml").write_text(json.dumps(dict(
                normalization_path=str(normalizer), base_policy_path=str(teacher),
            )))
            states = np.arange(48, dtype=np.float32).reshape(4, 4, 3) + 100
            actions = np.arange(64, dtype=np.float32).reshape(4, 4, 2, 2)
            archive = root / "bc_seed_123.npz"
            np.savez(archive, pre_states=states, actions=actions, assembly_rewards=np.zeros((4, 4)))
            first = ReferenceRewardTransitions(expert, root, sha256(normalizer), sha256(teacher), .5)
            np.savez(archive, pre_states=states, actions=actions, assembly_rewards=np.ones((4, 4)))
            second = ReferenceRewardTransitions(expert, root, sha256(normalizer), sha256(teacher), .5)
            for name in ("observations", "actions", "next_observations", "terminals"):
                torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0, atol=0)
            self.assertEqual(first.summary["reference_transitions"], 6)
            self.assertEqual(first.summary["reference_validation_transitions"], 6)
            torch.testing.assert_close(first.observations[-6:, 0], torch.from_numpy(states[:2, :-1].reshape(-1, 3)))
            torch.testing.assert_close(first.reference_validation[0][:, 0], torch.from_numpy(states[2:, :-1].reshape(-1, 3)))
            self.assertEqual(float(first.terminals[-6:].sum()), 0.)
            with self.assertRaisesRegex(ValueError, "differs"):
                ReferenceRewardTransitions(expert, root, "different", sha256(teacher), .5)
            events = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
            np.savez(archive, pre_states=states, actions=actions, assembly_rewards=events)
            terminal = ReferenceRewardTransitions(
                expert, root, sha256(normalizer), sha256(teacher), .5,
                assembly_terminal=True, success_threshold=1,
            )
            self.assertEqual(terminal.summary["reference_transitions"], 5)
            self.assertEqual(terminal.summary["reference_validation_transitions"], 4)
            self.assertEqual(terminal.summary["reference_training_terminals"], 1)
            self.assertEqual(terminal.summary["reference_validation_terminals"], 1)
            torch.testing.assert_close(terminal.terminals[-5:], torch.tensor([0., 1., 0., 0., 0.]))
            torch.testing.assert_close(terminal.observations[-5:, 0],
                                      torch.from_numpy(np.concatenate((states[0, :2], states[1, :3]))))
            torch.testing.assert_close(terminal.reference_validation[3], torch.tensor([1., 0., 0., 0.]))
            with self.assertRaisesRegex(ValueError, "explicit positive"):
                ReferenceRewardTransitions(expert, root, sha256(normalizer), sha256(teacher), .5,
                                           assembly_terminal=True)

    def test_alignment_metrics_reject_high_return_failed_outliers_as_improvement(self):
        from script.isaaclab.check_lfirl_alignment import outcome_metrics

        values = np.array([[1., 2.], [0., 100.], [-1., -1.]])
        events = np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]])
        result = outcome_metrics(values, events, gamma=1.)
        self.assertEqual(result["successes"], 1)
        self.assertGreater(result["failed_mean_return"], result["successful_mean_return"])
        self.assertEqual(result["top_return_success_fraction"]["1"], 0.)
        self.assertEqual(result["discounted_return_success_auc"], .5)

    def test_paired_reward_comparison_ignores_shared_reset_offsets(self):
        from script.isaaclab.check_lfirl_alignment import paired_outcome_metrics

        scores = {
            "first": np.array([[2., 3., 4.], [-2., 1., 0.], [0., 0., 0.]]),
            "second": np.array([[0., 1., 0.], [1., 2., 3.], [2., 2., 2.]]),
        }
        assembly = {
            "first": np.array([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
            "second": np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]),
        }
        expected = paired_outcome_metrics(scores, assembly, .9)["first_vs_second"]
        shifted = {name: value + np.array([[10000.], [-10000.], [500.]])
                   for name, value in scores.items()}
        actual = paired_outcome_metrics(shifted, assembly, .9)["first_vs_second"]
        self.assertEqual(actual["informative_resets"], 2)
        self.assertEqual(actual["success_preference"], 1.)
        self.assertEqual(actual["completion_prefix_preference"], 1.)
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key], places=8)

    def test_reward_uncertainty_clusters_correlated_policy_pairs_by_reset(self):
        from script.isaaclab.compare_lfirl_alignment import cluster_summary, preference_samples
        from script.isaaclab.check_lfirl_alignment import paired_outcome_metrics

        scores = {
            "first": np.array([[3., 3.], [0., 0.], [0., 0.]]),
            "second": np.array([[0., 0.], [2., 2.], [0., 0.]]),
            "third": np.zeros((3, 2)),
        }
        events = {
            "first": np.array([[0., 1.], [0., 0.], [0., 0.]]),
            "second": np.array([[0., 0.], [0., 1.], [0., 0.]]),
            "third": np.zeros((3, 2)),
        }
        samples = preference_samples(scores, events, .9)
        result = cluster_summary(samples["completion_prefix"])
        self.assertEqual(result["informative_resets"], 2)
        self.assertEqual(result["comparisons"], 4)
        self.assertEqual(result["mean"], 1.)
        self.assertEqual(result["ci95"], [1., 1.])
        pairs = paired_outcome_metrics(scores, events, .9)
        self.assertEqual(result["comparisons"], sum(row["completion_prefix_resets"] for row in pairs.values()))
        for row in pairs.values():
            self.assertEqual(row["completion_prefix_preference"], result["mean"])

    def test_state_only_reward_reload_preserves_frozen_targets_and_ignores_action(self):
        from copy import deepcopy

        settings = default_settings()
        for name in ("q_net", "b_net", "v_net", "r_net"):
            settings[name].update(state_embed_dim=16, hidden=16, layers=2)
            settings[name]["obs_encoder"].update(out_dim=16, per_key_hidden=16)
        original = LFIRLReward(settings, 3, 4, "cpu")
        original.set_normalization(torch.ones(3), torch.full((3,), 2.))
        saved = deepcopy(original.state_dict())
        state_settings = deepcopy(settings)
        state_settings["r_net"]["include_action"] = False
        restored = LFIRLReward(state_settings, 3, 4, "cpu")
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(saved)
        restored.load_state_dict(saved, load_reward_head=False)
        restored.reset_reward_head()
        for name in ("q", "b", "v"):
            for key, value in saved["models"][name].items():
                torch.testing.assert_close(getattr(restored, name).state_dict()[key],
                                           value, rtol=0, atol=0)
        obs = torch.randn(7, 1, 3)
        np.testing.assert_array_equal(restored.predict(obs, torch.randn(7, 4)),
                                      restored.predict(obs, torch.randn(7, 4) * 100))
        torch.testing.assert_close(restored.mean, original.mean, rtol=0, atol=0)
        torch.testing.assert_close(restored.std, original.std, rtol=0, atol=0)

    def test_reference_calibration_cannot_update_q_or_input_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            np.savez(path, states=np.ones((18, 3), dtype=np.float32),
                     actions=np.ones((18, 2), dtype=np.float32), traj_lengths=np.array([9, 9]))
            data = MacroTransitions(path, horizon=2, window_stride=1)
            data.summary["reference_transitions"] = 1
            settings = default_settings()
            for name in ("q_net", "b_net", "v_net", "r_net"):
                settings[name].update(state_embed_dim=16, hidden=16, layers=2)
                settings[name]["obs_encoder"].update(out_dim=16, per_key_hidden=16)
            settings["irl"]["batch_size"] = 4
            model = LFIRLReward(settings, 3, 4, "cpu")
            with self.assertRaisesRegex(ValueError, "frozen Q"):
                model.fit(data, None, directory, start_stage="QHAT", update_normalization=False)
            with self.assertRaisesRegex(ValueError, "normalization"):
                model.fit(data, None, directory, start_stage="V_INIT", update_normalization=True)
            original_q = {key: value.clone() for key, value in model.q.state_dict().items()}
            original_mean, original_std = model.mean.clone(), model.std.clone()
            stages = model.fit(data, None, directory, start_stage="V_INIT",
                               update_normalization=False, max_batches_per_stage=1,
                               shuffle_calibration=True)
            self.assertEqual([row["stage"] for row in stages], ["V_INIT", "B", "V_CALIBRATED", "R"])
            for key, value in original_q.items():
                torch.testing.assert_close(model.q.state_dict()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(model.mean, original_mean, rtol=0, atol=0)
            torch.testing.assert_close(model.std, original_std, rtol=0, atol=0)

    def test_macro_transitions_do_not_cross_padding_or_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            states = np.arange(1, 46, dtype=np.float32).reshape(15, 3)
            actions = np.arange(1, 31, dtype=np.float32).reshape(15, 2)
            states[4], actions[4] = 0, 0
            np.savez(path, states=states, actions=actions, traj_lengths=np.array([9, 6]))
            data = MacroTransitions(path, horizon=2)
            self.assertFalse(data.terminals.any())
            for action in data.actions.numpy():
                self.assertFalse((action == 0).any())
                self.assertEqual(action[2] - action[0], 2)
            inferred = MacroTransitions(path, horizon=2, terminal_mode="demonstration_end")
            self.assertEqual(int(inferred.terminals.sum()), 2)

    def test_single_action_adapter_uses_original_score_and_native_prediction_horizon(self):
        bc = GaussianBC()
        policy = NativeActionPolicy(bc)
        actions = torch.tensor([[.1, .2], [-.8, .4]])
        obs = torch.zeros(2, 1, 3)
        torch.manual_seed(13)
        actual = official.dp_action_score_from_policy(
            dp_model=policy, obs_t=obs, action_t=actions, device=torch.device("cpu"),
            policy_obs_keys=None, t_eval=0,
        )
        torch.manual_seed(13)
        tiled = actions[:, None].repeat(1, bc.horizon_steps, 1)
        noisy = bc.q_sample(tiled, torch.zeros(2, dtype=torch.long), torch.randn_like(tiled))
        expected = -(.99 ** .5) * noisy[:, 0] * (.1 / ((1-bc.alphas_cumprod[0]).sqrt()+1e-8))
        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 2))
        self.assertEqual(policy.n_action_steps, bc.horizon_steps)

    def test_single_action_rewards_use_pre_action_states_and_preserve_ppo_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            class Probe:
                def predict(self, observations, actions):
                    return observations[:, 0, 0] + actions.sum(-1)
            agent = TrainLFIRLDiffusionAgent.__new__(TrainLFIRLDiffusionAgent)
            agent.single_action_reward = True
            agent.irl_action_horizon = 1
            agent.reward_horizon = agent.act_steps = agent.horizon_steps = 2
            agent.reward_gamma = .5
            agent.n_envs, agent.n_steps, agent.obs_dim, agent.action_dim = 1, 1, 3, 2
            agent.n_cond_step, agent.itr, agent.success_replay = 1, 1, None
            agent.reward, agent.retention_dir = Probe(), Path(directory)
            observations = {"state": np.array([[[1., 2., 3.]]])}
            actions = np.array([[[10., 20.], [30., 40.]]])
            info = {"action_observations": np.array([[[4., 5., 6.], [7., 8., 9.]]])}
            agent.on_transition_collected(0, observations, actions, None, info, False)
            np.testing.assert_array_equal(agent._action_reward_trajs, [[68.]])
            rewards = agent.get_training_rewards(
                np.zeros((1, 1, 1, 3)), np.zeros((1, 1, 1, 3)), np.zeros((1, 1, 1, 2, 2)),
                np.full((1, 1), 10000.), np.zeros((1, 1)),
            )
            np.testing.assert_array_equal(rewards, [[68.]])
            self.assertEqual(agent.reward_horizon, 2)
            agent.on_transition_collected(0, observations, actions, None, {}, True)

    def test_wrapper_records_each_post_action_observation(self):
        sys.path.insert(0, str(ROOT / "external/furniture-bench"))
        import gym
        from env.gym_utils.wrapper.furniture import FurnitureRLSimEnvMultiStepWrapper

        class Environment(gym.Env):
            num_envs, max_env_steps, policy_observation_dim = 1, 100, 23
            device = torch.device("cpu")
            action_space = gym.spaces.Box(-1., 1., (1, 10), dtype=np.float32)
            observation_space = gym.spaces.Dict({
                "robot_state": gym.spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32),
                "parts_poses": gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
            })

            def observation(self):
                state = torch.zeros(1, 14)
                state[:, 0] = self.env_steps.float() * .1
                state[:, 6] = 1.
                return dict(robot_state=state, parts_poses=torch.zeros(1, 7))

            def reset(self):
                self.env_steps = torch.zeros(1, dtype=torch.long)
                return self.observation()

            def step(self, action):
                self.env_steps += 1
                return self.observation(), torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.bool), {}

        wrapper = FurnitureRLSimEnvMultiStepWrapper(Environment(), n_action_steps=8)
        wrapper.reset()
        wrapper.record_action_observations = True
        observation, _, _, _, info = wrapper.step(np.zeros((1, 8, 10), dtype=np.float32))
        np.testing.assert_allclose(info["action_observations"][0, :, 0], np.arange(1, 9)*.1)
        self.assertAlmostEqual(observation["state"][0, -1, 0], .8)

    def test_all_offline_stages_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            np.savez(path, states=np.random.randn(28, 3).astype(np.float32),
                     actions=np.random.uniform(-1, 1, (28, 2)).astype(np.float32),
                     traj_lengths=np.array([14, 14]))
            data = MacroTransitions(path, horizon=2)
            settings = default_settings()
            for name in ("q_net", "b_net", "v_net", "r_net"):
                settings[name].update(state_embed_dim=16, hidden=16, layers=2)
                settings[name]["obs_encoder"].update(out_dim=16, per_key_hidden=16)
            settings["irl"]["batch_size"] = 4
            reward = LFIRLReward(settings, 3, 4, "cpu")
            policy = MacroDiffusionPolicy(GaussianBC())
            history = reward.fit(data, policy, directory, max_batches_per_stage=2)
            self.assertEqual([r["stage"] for r in history], ["QHAT", "V_INIT", "B", "V_CALIBRATED", "R"])
            self.assertTrue(all(r["updates"] == 2 and np.isfinite(r["mean_loss"]) for r in history))
            before = reward.predict(data.observations, data.actions)
            restored = LFIRLReward(settings, 3, 4, "cpu")
            checkpoint = Path(directory) / "reward.pt"
            torch.save(reward.state_dict(), checkpoint)
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            np.testing.assert_array_equal(before, restored.predict(data.observations, data.actions))
            self.assertEqual(restored.rms.mean, reward.rms.mean)
            self.assertEqual(restored.rms.var, reward.rms.var)
            self.assertEqual(restored.rms.count, reward.rms.count)
            self.assertEqual(set(restored.target_states), {"q", "b", "v"})
            self.assertEqual(restored.target_mode, "final_frozen")
            for name in ("q", "b", "v"):
                for key, value in reward.target_states[name].items():
                    torch.testing.assert_close(value, getattr(reward, name).state_dict()[key])
            restored.fit(data, None, directory, max_batches_per_stage=1, start_stage="R")
            for name in ("q", "b", "v"):
                for key, value in reward.target_states[name].items():
                    torch.testing.assert_close(restored.target_states[name][key], value)

    def test_sliding_windows_use_intermediate_states_without_changing_macro_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            states = np.arange(1, 49, dtype=np.float32).reshape(16, 3)
            actions = np.arange(1, 33, dtype=np.float32).reshape(16, 2)
            np.savez(path, states=states, actions=actions, traj_lengths=np.array([8, 8]))
            data = MacroTransitions(path, horizon=3, terminal_mode="demonstration_end", window_stride=1)
            self.assertEqual(len(data.actions), 10)
            self.assertEqual(int(data.terminals.sum()), 2)
            self.assertAlmostEqual(data.summary["macro_time_step_seconds"], .3)
            self.assertEqual(data.summary["window_stride"], 1)
            for obs, action, next_obs, _ in data.batches(4, "cpu"):
                self.assertTrue(torch.equal(next_obs - obs, torch.full_like(obs, 9.)))
                self.assertTrue(torch.equal(action[:, 2:4] - action[:, :2], torch.full_like(action[:, :2], 2.)))

    def test_frozen_reward_hook_receives_pre_states_and_entire_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            class Probe:
                def predict(self, observations, actions):
                    return observations[:, 0, 0] + actions.sum(-1)
            agent = TrainLFIRLDiffusionAgent.__new__(TrainLFIRLDiffusionAgent)
            agent.obs_dim, agent.n_cond_step = 3, 1
            agent.horizon_steps, agent.action_dim = 2, 2
            agent.itr, agent.success_replay = 1, None
            agent.reward, agent.retention_dir = Probe(), Path(directory)
            observations = np.arange(18, dtype=np.float32).reshape(3, 2, 1, 3)
            chains = np.ones((3, 2, 3, 2, 2), dtype=np.float32)
            actual = agent.get_training_rewards(observations, observations * 100, chains, np.zeros((3, 2)), np.zeros((3, 2)))
            np.testing.assert_array_equal(actual, observations[:, :, 0, 0] + 4)

    def test_resumed_calibration_preserves_q_and_its_input_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            np.savez(path, states=np.random.randn(28, 3).astype(np.float32),
                     actions=np.random.uniform(-1, 1, (28, 2)).astype(np.float32),
                     traj_lengths=np.array([14, 14]))
            data = MacroTransitions(path, horizon=2)
            settings = default_settings()
            for name in ("q_net", "b_net", "v_net", "r_net"):
                settings[name].update(state_embed_dim=16, hidden=16, layers=2)
                settings[name]["obs_encoder"].update(out_dim=16, per_key_hidden=16)
            settings["irl"]["batch_size"] = 4
            reward = LFIRLReward(settings, 3, 4, "cpu")
            reward.set_normalization(torch.ones(3), torch.full((3,), 2.))
            initial_q = {key: value.clone() for key, value in reward.q.state_dict().items()}
            history = reward.fit(data, None, directory, max_batches_per_stage=2,
                                 start_stage="V_INIT", checkpoint_interval=1)
            self.assertEqual([row["stage"] for row in history], ["V_INIT", "B", "V_CALIBRATED", "R"])
            for key, value in initial_q.items():
                torch.testing.assert_close(reward.q.state_dict()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(reward.mean, torch.ones(3), rtol=0, atol=0)
            torch.testing.assert_close(reward.std, torch.full((3,), 2.), rtol=0, atol=0)
            self.assertTrue((Path(directory) / "reward_step_2.pt").is_file())

    def test_fresh_calibration_discards_stale_heads_but_preserves_fitted_q(self):
        settings = default_settings()
        for name in ("q_net", "b_net", "v_net", "r_net"):
            settings[name].update(state_embed_dim=16, hidden=16, layers=2)
            settings[name]["obs_encoder"].update(out_dim=16, per_key_hidden=16)
        reward = LFIRLReward(settings, 3, 4, "cpu")
        reward.set_normalization(torch.ones(3), torch.full((3,), 2.))
        old = {
            name: {key: value.clone() for key, value in getattr(reward, name).state_dict().items()}
            for name in ("q", "b", "v", "r")
        }
        reward.target_states = {"stale": True}
        reward.rms.count = 1000.
        reward.reset_calibration_heads()
        for key, value in old["q"].items():
            torch.testing.assert_close(reward.q.state_dict()[key], value, rtol=0, atol=0)
        for name in ("b", "v", "r"):
            self.assertTrue(any(
                not torch.equal(value, getattr(reward, name).state_dict()[key])
                for key, value in old[name].items()
            ))
        self.assertIsNone(reward.target_states)
        self.assertEqual(reward.target_mode, "requires_calibration_refit")
        self.assertLess(reward.rms.count, 1.)
        torch.testing.assert_close(reward.mean, torch.ones(3), rtol=0, atol=0)
        torch.testing.assert_close(reward.std, torch.full((3,), 2.), rtol=0, atol=0)

    def test_policy_normalized_inputs_are_not_standardized_again(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            states = np.random.randn(28, 3).astype(np.float32)
            states[:, 2] *= 1e-5
            np.savez(
                path, states=states,
                actions=np.random.uniform(-1, 1, (28, 2)).astype(np.float32),
                traj_lengths=np.array([14, 14]),
            )
            data = MacroTransitions(path, horizon=2)
            settings = default_settings()
            for name in ("q_net", "b_net", "v_net", "r_net"):
                settings[name].update(state_embed_dim=16, hidden=16, layers=2)
                settings[name]["obs_encoder"].update(out_dim=16, per_key_hidden=16)
            settings["irl"]["batch_size"] = 4
            reward = LFIRLReward(settings, 3, 4, "cpu")
            reward.fit(
                data, MacroDiffusionPolicy(GaussianBC()), directory,
                max_batches_per_stage=1, update_normalization=False,
            )
            torch.testing.assert_close(reward.mean, torch.zeros(3))
            torch.testing.assert_close(reward.std, torch.ones(3))


if __name__ == "__main__":
    unittest.main()
