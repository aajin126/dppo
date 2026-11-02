"""Run Furniture BC with IsaacLab evaluation, or DPPO fine-tuning on IsaacLab.

Reuses the selected original DPPO model, dataset, training agent and wrapper.
No datasets, checkpoints, or external repositories are downloaded here.
"""
import argparse
import logging
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def prepare_cuda_libraries():
    """Expose Isaac Sim's bundled CUDA libraries before Torch or Kit imports."""
    if sys.platform != "linux":
        return
    bundles = []
    if os.environ.get("FURNITURE_ML_BUNDLE"):
        bundles.append(Path(os.environ["FURNITURE_ML_BUNDLE"]).expanduser())
    roots = [ROOT.parent / "IsaacLab/_isaac_sim", ROOT.parent / "isaacsim"]
    if os.environ.get("ISAAC_PATH"):
        roots.insert(0, Path(os.environ["ISAAC_PATH"]).expanduser())
    bundles.extend(root / "exts/omni.isaac.ml_archive/pip_prebundle" for root in roots)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    paths = current.split(os.pathsep) if current else []
    additions = []
    for relative, library in (
        ("cusparselt/lib", "libcusparseLt.so.0"),
        ("nvidia/cuda_cupti/lib", "libcupti.so.12"),
    ):
        for bundle in bundles:
            directory = (bundle / relative).resolve()
            if (directory / library).is_file():
                if str(directory) not in paths:
                    additions.append(str(directory))
                break
    if additions:
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(additions + paths)
        print("[INFO] Added Isaac Sim CUDA library paths; restarting Python.", flush=True)
        # The dynamic linker reads LD_LIBRARY_PATH when the process starts.
        # Restart before importing AppLauncher, Torch, or simulation extensions.
        os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)


def is_bc_config(cfg):
    return cfg._target_ == "agent.pretrain.train_diffusion_agent.TrainDiffusionAgent"


def enable_progress_logging():
    # SimulationContext replaces console handlers with WARNING-level handlers.
    # Keep its file logger and warning filter, and show actual training progress.
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.INFO)


def compose_config(config_dir, config_name, overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    os.environ.setdefault("DPPO_DATA_DIR", str(ROOT / "data"))
    os.environ.setdefault("DPPO_LOG_DIR", str(ROOT / "log"))
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
    directory = Path(config_dir).expanduser()
    if not directory.is_absolute():
        directory = ROOT / directory
    with initialize_config_dir(version_base=None, config_dir=str(directory.resolve())):
        # Operational defaults; all model/task/hyperparameters come from the
        # selected existing YAML. Explicit Hydra overrides take precedence.
        wandb_overridden = any(
            override.lstrip("+").startswith(("wandb.", "wandb="))
            for override in overrides
        )
        operational_overrides = [] if wandb_overridden else ["wandb=null"]
        cfg = compose(
            config_name=config_name.removesuffix(".yaml"),
            overrides=[*operational_overrides, *overrides],
        )
    bc = is_bc_config(cfg)
    if bc and cfg.get("task") not in ("one_leg", "lamp", "round_table"):
        raise ValueError("IsaacLab BC evaluation supports one_leg, lamp and round_table.")
    if not bc and cfg.env.get("env_type") != "furniture":
        raise ValueError("This launcher is for the existing Furniture DPPO configs.")
    if not bc:
        with open_dict(cfg):
            cfg.env.specific.backend = "isaaclab"
    OmegaConf.resolve(cfg)
    for name in ("normalization_path", "base_policy_path", "train_dataset_path"):
        value = cfg.get(name)
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            cfg[name] = str(path.resolve())
    if cfg.model.get("network_path"):
        path = Path(cfg.model.network_path).expanduser()
        cfg.model.network_path = str((path if path.is_absolute() else ROOT/path).resolve())
    if bc:
        cfg.train_dataset.dataset_path = cfg.train_dataset_path
    else:
        cfg.env.specific.normalization_path = cfg.normalization_path
    return cfg


def preflight(cfg):
    if is_bc_config(cfg):
        dataset = Path(cfg.train_dataset_path)
        if not dataset.is_file():
            raise FileNotFoundError(f"Local BC dataset not found: {dataset}.")
        normalization = dataset.with_name("normalization.pth")
        if not normalization.is_file():
            raise FileNotFoundError(f"Paired normalization file not found: {normalization}.")
        return
    normalization = Path(cfg.normalization_path)
    if not normalization.is_file():
        raise FileNotFoundError(f"Local normalization file not found: {normalization}. No automatic download is enabled.")
    path = cfg.model.get("network_path")
    if not path or not Path(path).is_file():
        raise FileNotFoundError(
            f"Local pretrained Furniture checkpoint not found: {path}. "
            "Set base_policy_path=/absolute/path/to/state_N.pt.")


def make_bc_agent(cfg, args, app):
    """Add simulator evaluation at the original BC agent's checkpoint hook."""
    import json
    import shutil
    from copy import deepcopy
    import numpy as np
    import torch
    from omegaconf import OmegaConf, open_dict
    from agent.pretrain.train_diffusion_agent import TrainDiffusionAgent
    from model.diffusion.diffusion_eval import DiffusionEval
    from env.gym_utils import make_async

    eval_cfg = compose_config(f"cfg/furniture/finetune/{cfg.task}_{cfg.randomness}",
                              "ft_ppo_diffusion_mlp", [])
    normalization = str(Path(cfg.train_dataset_path).with_name("normalization.pth"))
    with open_dict(eval_cfg):
        eval_cfg.env.specific.normalization_path = normalization
        eval_cfg.env.specific.headless = args.headless
        eval_cfg.env.specific.gpu_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    if (cfg.obs_dim, cfg.action_dim, cfg.horizon_steps, cfg.cond_steps) != (
            eval_cfg.obs_dim, eval_cfg.action_dim, eval_cfg.horizon_steps, eval_cfg.cond_steps):
        raise ValueError("BC model dimensions must match the selected Furniture evaluation config.")
    class TrainDiffusionAgentLab(TrainDiffusionAgent):
        def __init__(self):
            self.venv = make_async(
                eval_cfg.env.name, env_type="furniture", num_envs=args.eval_envs,
                max_episode_steps=eval_cfg.env.max_episode_steps,
                **OmegaConf.to_container(eval_cfg.env.specific, resolve=True))
            enable_progress_logging()
            self.best_score = (-1., -1., -1)
            self.eval_policy = None
            super().__init__(cfg)
            OmegaConf.save(cfg, str(Path(self.logdir)/"config.yaml"))
            logging.info("BC: existing position demonstrations, original supervised loss; no PPO updates.")
            logging.info("BC output directory: %s", self.logdir)
            logging.info("IsaacLab: %s, %d evaluation environments, rollout after each checkpoint (%d epochs).",
                         cfg.task, args.eval_envs, self.save_model_freq)

        def step_ema(self):
            super().step_ema()
            if not app.is_running():
                raise KeyboardInterrupt
            if not args.headless:
                self.venv.env.sim.render()

        def save_model(self):
            super().save_model()
            was_training = self.ema_model.training
            self.ema_model.eval()
            numpy_state = np.random.get_state()
            try:
                device = torch.device(cfg.device)
                with torch.random.fork_rng(devices=[device.index or 0]), torch.inference_mode():
                    torch.manual_seed(cfg.seed)
                    np.random.seed(cfg.seed)
                    if self.eval_policy is None:
                        parameters = OmegaConf.to_container(cfg.model, resolve=True)
                        for key in ("_target_", "network", "network_path"):
                            parameters.pop(key, None)
                        parameters.update(use_ddim=eval_cfg.model.use_ddim,
                                          ddim_steps=eval_cfg.model.ddim_steps,
                                          randn_clip_value=eval_cfg.model.randn_clip_value)
                        self.eval_policy = DiffusionEval(
                            network=deepcopy(self.ema_model.network),
                            network_path=str(Path(self.checkpoint_dir)/f"state_{self.epoch}.pt"),
                            ft_denoising_steps=0, **parameters).eval()
                    self.eval_policy.network.load_state_dict(self.ema_model.network.state_dict(), strict=True)
                    obs = self.venv.reset()
                    returns = np.zeros(args.eval_envs, dtype=np.float64)
                    steps = math.ceil(eval_cfg.env.max_episode_steps/eval_cfg.act_steps)
                    logging.info("BC epoch %d: starting IsaacLab policy rollout (%d chunks).", self.epoch, steps)
                    for step in range(steps):
                        if not app.is_running():
                            raise KeyboardInterrupt
                        while not self.venv.env.sim.is_playing():
                            if not app.is_running():
                                raise KeyboardInterrupt
                            self.venv.env.sim.render()
                        state = torch.as_tensor(obs["state"], device=device, dtype=torch.float32)
                        action = self.eval_policy(cond={"state": state}, deterministic=True).trajectories
                        if not torch.isfinite(action).all():
                            raise RuntimeError("BC policy produced nonfinite actions during IsaacLab evaluation.")
                        obs, reward, terminated, truncated, _ = self.venv.step(
                            action[:, :eval_cfg.act_steps].cpu().numpy())
                        returns += reward
                        if (terminated | truncated).all():
                            break
                    result = {"epoch": self.epoch, "success_rate": float(np.mean(
                        returns >= eval_cfg.env.best_reward_threshold_for_success)),
                        "mean_reward": float(returns.mean()), "episode_rewards": returns.tolist(),
                        "environment_steps": (step+1)*eval_cfg.act_steps}
                    with (Path(self.logdir)/"isaaclab_eval.jsonl").open("a") as stream:
                        stream.write(json.dumps(result)+"\n")
                    score = (result["success_rate"], result["mean_reward"], self.epoch)
                    if score > self.best_score:
                        self.best_score = score
                        shutil.copyfile(Path(self.checkpoint_dir)/f"state_{self.epoch}.pt",
                                        Path(self.checkpoint_dir)/"best.pt")
                    logging.info("BC epoch %d: IsaacLab success %.1f%%, mean reward %.3f",
                                 self.epoch, 100*result["success_rate"], result["mean_reward"])
            finally:
                self.ema_model.train(was_training)
                np.random.set_state(numpy_state)

    return TrainDiffusionAgentLab()


def main():
    prepare_cuda_libraries()
    from isaaclab.app import AppLauncher
    from omegaconf import OmegaConf, open_dict

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="cfg/furniture/finetune/one_leg_low")
    parser.add_argument("--config-name", default="ft_ppo_diffusion_mlp")
    parser.add_argument("--eval-envs", type=int, default=4,
                        help="Parallel IsaacLab evaluation environments for BC checkpoints.")
    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args()
    if args.eval_envs < 1:
        parser.error("--eval-envs must be positive")
    cfg = compose_config(args.config_dir, args.config_name, overrides)
    preflight(cfg)
    with open_dict(cfg):
        cfg.device = args.device
        cfg.model.device = args.device
        if is_bc_config(cfg):
            cfg.train_dataset.device = args.device
        else:
            cfg.env.specific.gpu_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
            cfg.env.specific.headless = args.headless
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "external/furniture-bench"))
    Path(cfg.logdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(Path(cfg.logdir) / "config.yaml"), resolve=True)
    launcher = AppLauncher(args)
    app, agent = launcher.app, None
    try:
        import hydra

        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
        logging.getLogger("agent").setLevel(logging.INFO)
        logging.getLogger("env.gym_utils").setLevel(logging.INFO)
        agent = make_bc_agent(cfg, args, app) if is_bc_config(cfg) else hydra.utils.get_class(cfg._target_)(cfg)
        enable_progress_logging()
        agent.run()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if agent is not None:
            agent.venv.close()
        else:
            from isaaclab.sim import SimulationContext
            context = SimulationContext.instance()
            if context is not None:
                context.clear_all_callbacks()
                context.clear_instance()
                context.stop()
        sys.stdout.flush()
        sys.stderr.flush()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
