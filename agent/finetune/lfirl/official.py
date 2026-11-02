"""Load unchanged LFIRL definitions without its unrelated environment imports.

The upstream training script imports PushT, Minari and its own DP trainer at
module scope. We compile only the explicitly listed original definitions. Their
bodies, decorators and source filenames remain unchanged and auditable.
"""

import ast
import importlib
import math
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parents[3] / "external/lfirl"
PACKAGE = "dppo_lfirl_models"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(SOURCE / "models")]
    sys.modules[PACKAGE] = package
QNetwork = importlib.import_module(f"{PACKAGE}.q_network").QNetwork
VNetwork = importlib.import_module(f"{PACKAGE}.v_network").VNetwork
RewardNetwork = importlib.import_module(f"{PACKAGE}.r_network").RewardNetwork

NAMES = {
    "soft_update", "RunningMeanStd", "_build_flat_obs_for_policy",
    "dp_action_score_from_policy", "_qhat_value", "_select_obs_by_mask",
    "_repeat_obs_for_samples", "_sample_uniform_actions", "train_qhat_LFIRL_step",
    "train_b_offset_step", "QHatOnlyWrapper", "QFullWrapper",
    "_gumbel_rescale_loss", "_expectile_loss", "_grad_gumbel", "_gumbel_log_loss",
    "_augment_obs_and_actions_for_v", "train_v_function_step", "train_reward_function_step",
}


def load_definitions(path, names, nested_class=None):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    nodes = tree.body
    if nested_class:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == nested_class)
    selected = [node for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != names:
        raise ImportError(f"Required original LFIRL definitions missing in {path}")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), globals())


Tensor = torch.Tensor
ObsType = Union[Tensor, Dict[str, Tensor]]
load_definitions(SOURCE / "train_lfirl.py", NAMES)
load_definitions(SOURCE / "models/diffusion_model.py", {"score_first_action"}, "DiffusionPolicyTimeseries")
