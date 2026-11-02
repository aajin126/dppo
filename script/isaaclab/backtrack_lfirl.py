"""Probe a saved LFIRL update direction while retaining the original BC fallback.

This is a finite, physical validation experiment, not TRPO or a guarantee of
monotonic learning. Negative coefficients are diagnostics and cannot be adopted.
The existing evaluator, normalizer, BC network and DDIM sampler are reused.
"""

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import hydra
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PREVIOUS = ROOT / 'log/furniture-lfirl/continued_improvement_2026-11-02'
BASE = PREVIOUS / 'upstream_control_value2'
SOURCE = PREVIOUS / 'upstream_control_actor5'
REFERENCE = ROOT / 'log/furniture-lfirl/continued_improvement_2026-11-01/bc_upstream_gripper_memory'
HELPERS = ROOT / 'log/furniture-lfirl/score_contract_2026-11-02/method_source'
COEFFICIENTS = (-.1, .05, .1, .25)
SEEDS = (2026110261, 2026110262)
FRESH_SEEDS = (2026110281, 2026110282)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def blend_policy(base, candidate, coefficient):
    """Change actor_ft only, with exact original/updated endpoints."""
    if not math.isfinite(coefficient) or not -.1 <= coefficient <= 1.:
        raise ValueError('Coefficient must be finite and between -0.1 and 1')
    if base.keys() != candidate.keys():
        raise ValueError('Policy checkpoint keys differ')
    result = dict(base)
    found = False
    for key, value in base.items():
        other = candidate[key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f'Policy tensor contract differs: {key}')
        if key.startswith(('actor.', 'network.')) and not torch.equal(value, other):
            raise ValueError('Policies do not share the same frozen BC')
        if key.startswith('actor_ft.'):
            found = True
            if coefficient == 0.:
                mixed = value.clone()
            elif coefficient == 1.:
                mixed = other.clone()
            else:
                mixed = value + coefficient * (other - value)
            if not torch.isfinite(mixed).all():
                raise ValueError(f'Nonfinite interpolated actor: {key}')
            result[key] = mixed
    if not found:
        raise ValueError('Checkpoint has no trainable denoiser')
    return result


def load_cache_module():
    spec = importlib.util.spec_from_file_location('lfirl_backtrack_cache', HELPERS / 'evaluation_cache.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    base = torch.load(BASE / 'checkpoint/state_1.pt', map_location='cpu', weights_only=True)
    candidate = torch.load(SOURCE / 'checkpoint/state_6.pt', map_location='cpu', weights_only=True)
    companion = torch.load(BASE / 'checkpoint/lfirl_1.pt', map_location='cpu', weights_only=True)
    if companion['policy_gripper_command_memory'] != 'upstream_raw_action':
        raise ValueError('Expected source-faithful gripper control')
    companion = {key: value for key, value in companion.items()
                 if key not in ('training_state', 'success_replay', 'best_evaluation')}
    companion.update(training_state=None, success_replay=None, best_evaluation=None,
                     evaluation_only=True, training_resume_allowed=False)
    original_cfg = OmegaConf.load(SOURCE / 'config.yaml')
    original_cfg.model.device = 'cpu'
    model = hydra.utils.instantiate(original_cfg.model)
    condition = {'state': torch.zeros(8, 1, 58)}
    endpoint_checks = []
    for alpha, state in ((0., base['model']), (1., candidate['model'])):
        mixed = blend_policy(base['model'], candidate['model'], alpha)
        for deterministic in (False, True):
            with torch.no_grad():
                model.load_state_dict(state, strict=True)
                torch.manual_seed(2026110301)
                expected = model(cond=condition, deterministic=deterministic, return_chain=True)
                model.load_state_dict(mixed, strict=True)
                torch.manual_seed(2026110301)
                observed = model(cond=condition, deterministic=deterministic, return_chain=True)
            torch.testing.assert_close(expected.chains, observed.chains, rtol=0, atol=0)
        endpoint_checks.append({'coefficient': alpha, 'sampler_identical': True})

    entries = []
    for alpha in COEFFICIENTS:
        name = f'alpha_{alpha:+.2f}'.replace('+', 'p').replace('-', 'm').replace('.', '_')
        run = output / name
        checkpoint = run / 'checkpoint'
        checkpoint.mkdir(parents=True, exist_ok=True)
        state = dict(base)
        state['model'] = blend_policy(base['model'], candidate['model'], alpha)
        state['itr'] = 1
        for iteration in (0, 1):
            model_path = checkpoint / f'state_{iteration}.pt'
            expected = base if iteration == 0 else state
            if model_path.exists():
                existing = torch.load(model_path, map_location='cpu', weights_only=True)
                if any(not torch.equal(value, existing['model'][key])
                       for key, value in expected['model'].items()):
                    raise ValueError(f'Preserve inconsistent existing checkpoint: {model_path}')
            else:
                torch.save(expected, model_path)
            reward_path = checkpoint / f'lfirl_{iteration}.pt'
            if not reward_path.exists():
                torch.save(dict(companion, iteration=iteration), reward_path)
        cfg = OmegaConf.load(SOURCE / 'config.yaml')
        cfg.logdir = str(run)
        cfg.wandb = None
        cfg.lfirl.resume_checkpoint = None
        cfg.train.n_train_itr = 2
        cfg_path = run / 'config.yaml'
        if cfg_path.exists():
            if OmegaConf.to_container(OmegaConf.load(cfg_path)) != OmegaConf.to_container(cfg):
                raise ValueError('Preserve existing evaluation configuration')
        else:
            OmegaConf.save(cfg, cfg_path)
        write_json(checkpoint / 'best.json', {
            'iteration': 1, 'criterion': 'Prespecified update-direction diagnostic, not selected best',
            'coefficient': alpha, 'extension_beyond_original_lfirl': True,
            'diagnostic_only': alpha < 0,
        })
        entries.append({'run': str(run), 'coefficient': alpha, 'diagnostic_only': alpha < 0})
    write_json(output / 'experiment_plan.json', {
        'method': 'Controlled backtracking along a saved LFIRL/DPPO actor update',
        'source_bc': str(BASE / 'checkpoint/state_1.pt'),
        'source_updated_policy': str(SOURCE / 'checkpoint/state_6.pt'),
        'coefficients': list(COEFFICIENTS), 'episodes_per_evaluation': 1024,
        'negative_coefficient_is_diagnostic_only': True,
        'reward_changed': False, 'sampler_changed': False, 'additional_training_steps': 0,
        'ordinary_start_seeds': list(SEEDS), 'reserved_final_seeds': list(FRESH_SEEDS),
        'endpoint_checks': endpoint_checks, 'candidates': entries,
    })
    retained = output / 'retained_policy.json'
    if not retained.exists():
        write_json(retained, {
            'source': str(BASE / 'checkpoint/state_1.pt'), 'kind': 'original_bc',
            'candidate_adopted': False, 'reason': 'Retain BC until independent validation supports improvement',
        })
    return entries


def evaluate(run, output, policy, seeds, reference=None, restore_self=False):
    cache = load_cache_module()
    args = argparse.Namespace(output=output, run_dir=run, seeds=list(seeds), num_envs=512,
                              policy=policy, iterations=2, restore_reference=reference is not None,
                              restore_self=restore_self)
    if cache.inspect_evaluation(args) == 'reuse':
        return
    command = ['bash', str(HELPERS / 'evaluate.sh'), '--run-dir', str(run), '--output', str(output),
               '--policy', policy, '--iterations', '2', '--num-envs', '512',
               '--seeds', *map(str, seeds)]
    if reference is not None:
        command.extend(['--reference-dir', str(reference), '--restore-reference'])
    if restore_self:
        command.append('--restore-self')
    for attempt in (1, 2):
        with (run / f'{output.parent.name}_{output.name}_process.log').open('a') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            cache.inspect_evaluation(args, verify_only=True)
            return
        if result.returncode not in (139, -11) or attempt == 2:
            raise RuntimeError(f'Evaluation failed, exit={result.returncode}: {run}')
        cache.inspect_evaluation(args)


def compare(reference, candidate, output):
    subprocess.run([sys.executable, str(HELPERS / 'compare_paired.py'), str(reference),
                    str(candidate), '--output', str(output)], check=True)
    return json.loads(output.read_text())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for key in ('DISPLAY', 'WAYLAND_DISPLAY'):
        os.environ.pop(key, None)
    os.environ['PYTHONUNBUFFERED'] = '1'
    entries = prepare(output)
    if args.prepare_only:
        print('Prepared four checkpoints; real BC/updated sampler endpoints verified.')
        return
    rows = []
    for entry in entries:
        run = Path(entry['run'])
        evaluation = run / 'evaluation'
        evaluate(run, evaluation, 'final', SEEDS, reference=REFERENCE)
        result = compare(REFERENCE, evaluation, run / 'vs_bc.json')
        subprocess.run([sys.executable, str(HELPERS / 'analyze_failures.py'), str(REFERENCE),
                        str(evaluation), '--reward', str(OmegaConf.load(run / 'config.yaml').lfirl.reward_checkpoint),
                        '--output', str(run / 'failure_analysis.json')], check=True)
        report = json.loads((evaluation / 'report.json').read_text())
        final_matches = sum(sum(row['final_pose_matches']) for row in report['rollouts'])
        rows.append(dict(entry, successes=result['candidate_assembly_successes'],
                         final_pose_matches=final_matches, paired_p=result['paired_exact_two_sided_p_value']))
        write_json(output / 'development_results.json', {'results': rows, 'completed': False})
    eligible = [row for row in rows if not row['diagnostic_only']
                and row['successes'] > 15 and row['final_pose_matches'] >= 10]
    selected = max(eligible, key=lambda row: (row['successes'], row['final_pose_matches'],
                                             -row['coefficient'])) if eligible else None
    write_json(output / 'development_results.json', {'results': rows, 'completed': True,
                                                       'selected': selected})
    verified = False
    if selected:
        run = Path(selected['run'])
        fresh = run / 'fresh_final'
        fresh.mkdir(exist_ok=True)
        evaluate(run, fresh / 'bc', 'bc', FRESH_SEEDS, restore_self=True)
        evaluate(run, fresh / 'final', 'final', FRESH_SEEDS, reference=fresh / 'bc')
        result = compare(fresh / 'bc', fresh / 'final', fresh / 'vs_bc.json')
        baseline = json.loads((fresh / 'bc/report.json').read_text())
        candidate = json.loads((fresh / 'final/report.json').read_text())
        pose_count = lambda report: sum(sum(row['final_pose_matches']) for row in report['rollouts'])
        verified = bool(result['improvement_supported_in_this_comparison']
                        and pose_count(candidate) >= pose_count(baseline))
        write_json(fresh / 'conclusion.json', {'policy_improvement_verified': verified,
                                               'comparison': result, 'selection_is_independent': True})
        if verified:
            write_json(output / 'retained_policy.json', {
                'source': str(run / 'checkpoint/state_1.pt'), 'kind': 'validated_backtracked_lfirl_policy',
                'candidate_adopted': True, 'comparison': str(fresh / 'conclusion.json'),
            })
    write_json(output / 'completed.json', {'completed': True, 'policy_improvement_verified': verified,
                                          'original_bc_retained': not verified,
                                          'additional_training_steps': 0})


if __name__ == '__main__':
    main()
