"""Check launcher configuration without starting Isaac Sim or W&B."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from script.isaaclab.run_lab import compose_config, preflight


class LauncherConfigTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            "DPPO_DATA_DIR": str(ROOT / "data"),
            "DPPO_LOG_DIR": str(ROOT / "log"),
        })
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("DPPO_WANDB_ENTITY", None)

    def test_logging_defaults_to_disabled_without_an_entity(self):
        cfg = compose_config(
            "cfg/furniture/finetune/one_leg_low", "ft_lfirl_diffusion_mlp", []
        )
        self.assertIsNone(cfg.wandb)
        self.assertEqual(cfg.env.n_envs, 4)

    def test_actual_lfirl_command_with_explicit_wandb_fields(self):
        reward = ROOT / "log/furniture-lfirl/offline_one_leg_qfit_rfit_1000/reward.pt"
        cfg = compose_config(
            "cfg/furniture/finetune/one_leg_low",
            "ft_lfirl_diffusion_mlp",
            [
                f"lfirl.reward_checkpoint={reward}",
                "env.n_envs=32",
                "train.batch_size=2048",
                "+train.logprob_batch_size=8192",
                "train.n_train_itr=20",
                "train.val_freq=5",
                "wandb.project=furniture-one-leg-lfirl",
                "wandb.entity=leehahj77-ewha-womans-university",
                "wandb.run=one_leg_lfirl_trial",
            ],
        )
        self.assertEqual(cfg.wandb.project, "furniture-one-leg-lfirl")
        self.assertEqual(cfg.wandb.entity, "leehahj77-ewha-womans-university")
        self.assertEqual(cfg.wandb.run, "one_leg_lfirl_trial")
        self.assertEqual(cfg.env.n_envs, 32)
        self.assertEqual(cfg.train.logprob_batch_size, 8192)
        self.assertEqual(cfg.env.specific.backend, "isaaclab")
        preflight(cfg)
        self.assertTrue(Path(cfg.lfirl.reward_checkpoint).is_file())

    def test_explicit_logging_uses_entity_from_environment(self):
        os.environ["DPPO_WANDB_ENTITY"] = "example-team"
        for config_dir, config_name in (
            ("cfg/furniture/finetune/one_leg_low", "ft_lfirl_diffusion_mlp"),
            ("cfg/furniture/pretrain/one_leg_low", "pre_diffusion_mlp"),
        ):
            with self.subTest(config=config_name):
                cfg = compose_config(config_dir, config_name, [
                    "wandb.project=example-project", "wandb.run=example-run"
                ])
                self.assertEqual(cfg.wandb.entity, "example-team")
                self.assertEqual(cfg.wandb.project, "example-project")

    def test_explicit_logging_disable_is_preserved(self):
        cfg = compose_config(
            "cfg/furniture/finetune/one_leg_low",
            "ft_lfirl_diffusion_mlp",
            ["wandb=null"],
        )
        self.assertIsNone(cfg.wandb)


if __name__ == "__main__":
    unittest.main()
