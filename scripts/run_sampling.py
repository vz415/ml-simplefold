#!/usr/bin/env python3
"""Hydra entry point for Slurm sampling; config inspection is safe locally."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
MODELS = {f'simplefold_{size}' for size in ('100M', '360M', '700M', '1.1B', '1.6B', '3B')}


def validate_config(cfg):
    """Check launch settings without importing Torch or opening weights."""
    if cfg.model not in MODELS or cfg.sampling.mode not in ('standard', 'fk'):
        raise ValueError('Unknown model or sampling mode')
    for key in ('num_steps', 'samples'):
        value = cfg[key]
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg.seed) is not int or cfg.seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not math.isfinite(cfg.tau) or cfg.tau < 0:
        raise ValueError('tau must be finite and nonnegative')
    for key, subdir in (('checkpoint_dir', 'checkpoints'), ('ccd_dir', 'ccd')):
        expected = os.path.abspath(os.path.join(cfg.paths.cache_dir, subdir))
        if os.path.abspath(cfg.paths[key]) != expected:
            raise ValueError(f'paths.{key} is derived; override paths.cache_dir instead')
    timeout = int(cfg.storage_timeout_seconds)
    if timeout <= 0 or str(timeout) != str(cfg.storage_timeout_seconds):
        raise ValueError('storage_timeout_seconds must be a positive integer')
    if cfg.sampling.mode == 'fk':
        sampling = cfg.sampling
        if cfg.samples < 2 or cfg.num_steps < 2:
            raise ValueError('FK needs at least two samples and two steps')
        if not math.isfinite(sampling.beta) or sampling.beta < 0:
            raise ValueError('sampling.beta must be finite and nonnegative')
        if not 0 <= sampling.ess_threshold <= 1:
            raise ValueError('sampling.ess_threshold must be in [0, 1]')
        if not sampling.checkpoint_times or any(
                not 0 < t < .99 for t in sampling.checkpoint_times):
            raise ValueError('FK checkpoint times must be within (0, 0.99)')
        for key in ('brownian_seed', 'resampling_seed'):
            if type(sampling[key]) is not int or sampling[key] < 0:
                raise ValueError(f'sampling.{key} must be a nonnegative integer')
        if type(sampling.common_start) is not bool:
            raise ValueError('sampling.common_start must be boolean')


def run_python(script, *args):
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *map(str, args)], check=True)


@hydra.main(version_base='1.3', config_path='../configs', config_name='sample')
def main(cfg: DictConfig):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Protein sampling is permitted only inside a Slurm GPU job')
    validate_config(cfg)
    paths = {k: Path(to_absolute_path(v)) for k, v in cfg.paths.items()}
    fasta = Path(to_absolute_path(cfg.fasta))
    output = Path(HydraConfig.get().runtime.output_dir)
    if (output / 'experiment.json').exists():
        raise RuntimeError(f'Run already exists: {output}; use a new job/output directory')
    if not fasta.exists():
        raise FileNotFoundError(fasta)
    os.environ['TORCH_HOME'] = str(paths['cache_dir'] / 'torch')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU inference')
    # Hydra keeps its composition/override receipts; save a fully resolved copy too.
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    (output / 'resolved-config.yaml').write_text(resolved)
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    (output / 'git-commit.txt').write_text(commit + '\n')
    with (output / 'pip-freeze.txt').open('w') as receipt:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=receipt, check=True)
    metadata = {
        'job_id': os.environ['SLURM_JOB_ID'], 'name': cfg.name, 'model': cfg.model,
        'sampling_mode': ('paired_frozen_weight_fk' if cfg.sampling.get('run_baseline', True)
                          else 'frozen_weight_fk') if cfg.sampling.mode == 'fk' else 'standard',
        'encoder': 'ESM2-3B', 'backend': 'torch', 'fasta_path': str(fasta),
        'fasta_sha256': hashlib.sha256(fasta.read_bytes()).hexdigest() if fasta.is_file() else None,
        'num_steps': cfg.num_steps, 'nsample_per_protein': cfg.samples,
        'tau': cfg.tau, 'seed': cfg.seed, 'log_timesteps': True,
        'compute_plddt': False, 'output_format': 'pdb', 'output_dir': str(output),
        'source_commit': commit, 'cuda_device': torch.cuda.get_device_name(0),
        'slurm_environment': {key: os.environ[key] for key in (
            'SLURM_JOB_NAME', 'SLURM_JOB_ACCOUNT', 'SLURM_JOB_PARTITION',
            'SLURM_JOB_QOS', 'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE',
            'SLURM_MEM_PER_CPU', 'SLURM_JOB_GPUS') if key in os.environ},
        'resolved_config_sha256': hashlib.sha256(resolved.encode()).hexdigest(),
        'hydra_overrides': list(HydraConfig.get().overrides.task),
    }
    (output / 'experiment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(resolved, flush=True)
    run_python('download_weights.py', '--artifact_dir', paths['cache_dir'],
               '--simplefold_model', cfg.model)
    if cfg.sampling.mode == 'fk':
        from sample_complex_fk import run_fk
        args = SimpleNamespace(
            model=cfg.model, fasta_path=fasta, num_steps=cfg.num_steps,
            particles=cfg.samples, tau=cfg.tau, seed=cfg.seed,
            ckpt_dir=paths['checkpoint_dir'], cache_dir=paths['ccd_dir'],
            reference_cif=paths['reference_cif'], reference_pdb=paths['reference_pdb'],
            output_dir=output)
        conditions = run_fk(args, OmegaConf.to_container(cfg.sampling, resolve=True))
    else:
        subprocess.run([
            'simplefold', '--backend', 'torch', '--simplefold_model', cfg.model,
            '--ckpt_dir', str(paths['checkpoint_dir']), '--cache_dir', str(paths['ccd_dir']),
            '--fasta_path', str(fasta), '--output_dir', str(output),
            '--num_steps', str(cfg.num_steps), '--tau', str(cfg.tau),
            '--nsample_per_protein', str(cfg.samples), '--seed', str(cfg.seed),
            '--output_format', 'pdb'], check=True)
        conditions = [output]
    for condition in conditions:
        run_python('validate_predictions.py', '--fasta_path', fasta,
                   '--prediction_dir', condition / f'predictions_{cfg.model}',
                   '--num_samples', cfg.samples)
    print(f'Sampling and validation completed: {output}', flush=True)


if __name__ == '__main__':
    # Stop before Hydra creates a run directory on a local machine. Inspection
    # flags print config/help without invoking the task or opening model files.
    inspection_flags = {'--cfg', '-c', '--help', '-h', '--hydra-help', '--info', '-i', '--version'}
    inspection = any(arg.split('=', 1)[0] in inspection_flags for arg in sys.argv[1:])
    if not os.environ.get('SLURM_JOB_ID') and not inspection:
        raise SystemExit('Use a Slurm GPU job for sampling; inspect locally with --cfg job --resolve.')
    main()
