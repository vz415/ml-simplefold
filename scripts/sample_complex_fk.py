#!/usr/bin/env python3
"""Run frozen-weight complex FK sampling on a Slurm GPU only."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'simplefold'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def expand_shared_batch(batch, particles):
    """Share singleton conditioning storage instead of copying large features."""
    import torch
    return {key: value.expand(particles, *value.shape[1:])
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 1
            else value for key, value in batch.items()}


def sampling_conditions(config):
    """Keep legacy paired pilots; new FK-only runs reuse an existing baseline."""
    fk = ('fk', float(config['beta']))
    return [('baseline', 0.), fk] if config.get('run_baseline', True) else [fk]


def make_reward(args, config):
    if config.get('reward_kind') == '2h62_partial_topology':
        from fk_2h62_topology_reward import TopologyComplexReward
        return TopologyComplexReward(args.reference_cif, args.reference_pdb, config['reward'])
    if config.get('reward_kind', '2goo_reference') == '2h62_partial':
        from fk_2h62_reward import PartialComplexReward
        return PartialComplexReward(args.reference_cif, args.reference_pdb, config['reward'])
    from fk_complex_reward import ComplexReward
    return ComplexReward(args.reference_cif, args.reference_pdb, config['reward'])


def score_existing_baseline(reward, prediction_dir, output_dir, expected_samples=None):
    """Analyze available baseline PDBs without creating any new structures."""
    paths = sorted(Path(prediction_dir).glob('*_sampled_*.pdb'),
                   key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    status = 'scored' if paths else 'not_available_at_launch'
    if paths and expected_samples is not None and (
            len(paths) != expected_samples or
            not (Path(prediction_dir).parent / 'validation.json').is_file()):
        status = 'not_ready_at_launch'
    receipt = {'prediction_dir': str(prediction_dir), 'status': status,
               'available_prediction_count': len(paths), 'expected_samples': expected_samples,
               'samples': [{'path': str(p), 'sha256': digest(p), **reward.score(p)}
                           for p in paths] if status == 'scored' else []}
    write_json(Path(output_dir) / 'existing-baseline-scores.json', receipt)


def run_fk(args, config):
    """Run the requested conditions using settings resolved by run_sampling.py."""
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Protein inference is permitted only inside a Slurm GPU job')
    if args.particles < 2 or args.num_steps < 2:
        raise ValueError('Need at least two particles and two steps')

    import numpy as np
    import torch
    import lightning.pytorch as pl
    from inference import initialize_folding_model, initialize_esm_model, initialize_others
    from model.torch.fk_sampler import FKSampler, make_initial_noise
    from utils.datamodule_utils import process_one_inference_structure
    from utils.fasta_utils import check_fasta_inputs, download_fasta_utilities, process_fastas
    from utils.boltz_utils import process_structure, save_structure, center_random_augmentation
    from boltz_data_pipeline.write.pdb import to_pdb

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU inference')
    if any(not isinstance(seed, int) or seed < 0 for seed in
           (args.seed, config['brownian_seed'], config['resampling_seed'])):
        raise ValueError('RNG seeds must be nonnegative integers')
    # Fail on incompatible reference inputs before loading either large model.
    reward = make_reward(args, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if config.get('baseline_prediction_dir'):
        score_existing_baseline(reward, config['baseline_prediction_dir'], args.output_dir,
                                expected_samples=args.particles)
    pl.seed_everything(args.seed, workers=True)
    checkpoint_path = args.ckpt_dir / f'{args.model}.ckpt'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    # Use the existing inference setup and atom mapping, with no training path.
    model_batch_size = config['model_batch_size']
    chunked = model_batch_size is not None
    inference_args = argparse.Namespace(
        backend='torch', simplefold_model=args.model, ckpt_dir=str(args.ckpt_dir),
        nsample_per_protein=1 if chunked else args.particles, num_steps=args.num_steps,
        tau=args.tau, plddt=False)
    model, device = initialize_folding_model(inference_args)
    model.eval().requires_grad_(False)
    if chunked:
        model.to('cpu')
        torch.cuda.empty_cache()
        print('Folding model offloaded for singleton ESM/input preparation.', flush=True)
    esm_model, esm_dict, af2_to_esm = initialize_esm_model(inference_args, device)
    esm_model.eval().requires_grad_(False)
    tokenizer, featurizer, processor, flow, base_sampler = initialize_others(inference_args, device)
    download_fasta_utilities(args.cache_dir)
    data = check_fasta_inputs(args.fasta_path)
    if len(data) != 1:
        raise ValueError('This pilot requires exactly one multichain FASTA target')
    process_fastas(data=data, out_dir=args.output_dir, ccd_path=args.cache_dir / 'ccd.pkl')
    structure_files = sorted((args.output_dir / 'structures').glob('*.npz'))
    if len(structure_files) != 1:
        raise ValueError('Expected one prepared structure')
    structure_file = structure_files[0]
    record_file = args.output_dir / 'records' / f'{structure_file.stem}.json'
    with torch.no_grad():
        batch, structure, record = process_one_inference_structure(
            structure_file, record_file, tokenizer, featurizer, processor,
            esm_model, esm_dict, af2_to_esm)
    # ESM representations are identical for both conditions and are cached in
    # batch; release encoder GPU memory before running the two folding paths.
    del esm_model
    torch.cuda.empty_cache()
    if chunked:
        model.to(device)
        batch = expand_shared_batch(batch, args.particles)
        print(f'Shared inputs prepared; {args.particles} particles, '
              f'model batches of {model_batch_size}.', flush=True)
    print(f'GPU memory before sampling: allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB, '
          f'reserved={torch.cuda.memory_reserved()/2**30:.2f} GiB.', flush=True)
    steps = base_sampler.steps
    times = config['checkpoint_times']
    if not times or any(not 0 < t < base_sampler.w_cutoff for t in times):
        raise ValueError('Reward checkpoints must be inside the stochastic interval')
    indices = sorted({int(torch.argmin(torch.abs(steps[:-1] - t))) for t in times})
    initial_generator = torch.Generator(device=device).manual_seed(args.seed)
    initial = make_initial_noise(batch['coords'].shape, generator=initial_generator,
                                 device=device, dtype=batch['coords'].dtype,
                                 common_start=bool(config['common_start']))
    torch.save(initial.cpu(), args.output_dir / 'initial-state.pt')
    # Small conditioning snapshot allows future branches to recover the exact
    # CCD conformers/masks/atom indices without saving repeated ESM tensors.
    conditioning = {k: (v[:1].cpu() if v.ndim and v.shape[0] == args.particles else v.cpu())
                    for k, v in batch.items() if isinstance(v, torch.Tensor) and k != 'esm_s'}
    torch.save(conditioning, args.output_dir / 'conditioning-without-esm.pt')
    parameter_versions = {name: p._version for name, p in model.named_parameters()}
    metadata = {
        'job_id': os.environ['SLURM_JOB_ID'], 'model': args.model, 'weights_fixed': True,
        'optimizer': None, 'compute_plddt': False, 'scale_angstrom': processor.scale,
        'num_steps': args.num_steps, 'tau': args.tau, 'particles': args.particles,
        'model_batch_size': model_batch_size,
        'shared_conditioning_storage': chunked,
        'common_start': bool(config['common_start']), 'initial_seed': args.seed,
        'brownian_seed': config['brownian_seed'], 'resampling_seed': config['resampling_seed'],
        'checkpoint_indices': indices, 'checkpoint_times': [float(steps[i]) for i in indices],
        'config': config, 'config_sha256': digest(args.output_dir / 'resolved-config.yaml'),
        'fasta_sha256': digest(args.fasta_path), 'reference_cif_sha256': digest(args.reference_cif),
        'reference_pdb_sha256': digest(args.reference_pdb),
        'checkpoint': str(checkpoint_path), 'checkpoint_bytes': checkpoint_path.stat().st_size,
        'source_commit': (args.output_dir / 'git-commit.txt').read_text().strip(),
        'initial_potential': 1.0,
        'terminal_outputs': 'Weighted particles; terminal resampling is not performed',
        'initialization': ('New common Gaussian latent' if config['common_start']
                           else 'Independent Gaussian particle latents'),
        'conditions': [label for label, _ in sampling_conditions(config)],
        'existing_baseline_prediction_dir': config.get('baseline_prediction_dir'),
        'baseline_comparison': ('Matched initial states and Brownian stream'
                                if config.get('run_baseline', True) else
                                'Historical baseline; no claim of paired initial states or noise'),
    }
    write_json(args.output_dir / 'fk-experiment.json', metadata)
    results = {}
    for label, beta in sampling_conditions(config):
        condition_dir = args.output_dir / label
        prediction_dir = condition_dir / f'predictions_{args.model}'
        state_dir = condition_dir / 'source-states'
        prediction_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        events = []
        with tempfile.TemporaryDirectory(prefix='simplefold-fk-score-') as scratch:
            scratch_dir = Path(scratch)

            def score_fn(coords, t):
                started = time.monotonic()
                if t < float(steps[indices[0]]) - 1e-6:
                    events.append({'time': t, 'kind': 'constant_initial_potential',
                                   'rewards': [0.] * args.particles})
                    return np.zeros(args.particles)
                coords = center_random_augmentation(
                    coords, batch['atom_pad_mask'], augmentation=False, centering=True)
                rows, cached = [], {}
                for i in range(args.particles):
                    converted = process_structure(deepcopy(structure), coords[i] * processor.scale,
                                                  batch['atom_pad_mask'][i], record)
                    pdb_text = to_pdb(converted)
                    pdb_hash = hashlib.sha256(pdb_text.encode()).hexdigest()
                    if pdb_hash not in cached:
                        path = scratch_dir / f'particle_{i}.pdb'
                        path.write_text(pdb_text)
                        cached[pdb_hash] = reward.score(path)
                    rows.append(cached[pdb_hash])
                values = [r['reward'] for r in rows]
                events.append({'time': t, 'kind': 'terminal' if t == 1. else 'clean_estimate',
                               'particles': rows, 'rewards': values,
                               'seconds': time.monotonic() - started})
                write_json(condition_dir / 'reward-events.json', events)
                print(f'{label}: scored t={t:.5f}, reward mean={np.mean(values):.4f}, '
                      f'best={max(values):.4f}, seconds={time.monotonic()-started:.1f}', flush=True)
                return values

            def checkpoint_fn(index, t, state):
                torch.save({'step': index, 'time': t, 'coords': state.cpu(),
                            'scale_angstrom': processor.scale, 'before_resampling': True},
                           state_dir / f'step_{index:04d}.pt')

            sampler = FKSampler(num_timesteps=args.num_steps, t_start=base_sampler.t_start,
                                tau=args.tau, log_timesteps=base_sampler.log_timesteps,
                                w_cutoff=base_sampler.w_cutoff, beta=beta,
                                checkpoint_indices=indices, ess_threshold=config['ess_threshold'],
                                model_batch_size=model_batch_size)
            started = time.monotonic()
            print(f'{label}: starting {args.num_steps} steps for {args.particles} particles.',
                  flush=True)
            result = sampler.sample(
                model, flow, initial.clone(), batch, score_fn=score_fn,
                brownian_generator=torch.Generator(device=device).manual_seed(config['brownian_seed']),
                resampling_generator=torch.Generator(device=device).manual_seed(config['resampling_seed']),
                checkpoint_fn=checkpoint_fn)
        out_dict = processor.postprocess(result, batch)
        for i in range(args.particles):
            converted = process_structure(deepcopy(structure), out_dict['denoised_coords'][i],
                                          batch['atom_pad_mask'][i], record)
            save_structure(converted, prediction_dir, f'{record.id}_sampled_{i}', output_format='pdb')
        # Re-score the written final structures. This also verifies that saved
        # PDB rounding agrees with the terminal scores used by the sampler.
        final_scores = [reward.score(prediction_dir / f'{record.id}_sampled_{i}.pdb')
                        for i in range(args.particles)]
        final_rewards = [r['reward'] for r in final_scores]
        if not np.allclose(final_rewards, result['final_rewards'].cpu().numpy(), atol=1e-9, rtol=0):
            raise RuntimeError('Saved PDB rewards differ from sampler terminal rewards')
        frozen = (not any(p.requires_grad for p in model.parameters())
                  and parameter_versions == {name: p._version for name, p in model.named_parameters()})
        if not frozen:
            raise RuntimeError('Frozen-model check failed')
        serialized = {k: v.cpu().tolist() if isinstance(v, torch.Tensor) else v
                      for k, v in result.items() if k not in ('denoised_coords', 'coords')}
        serialized.update(condition=label, beta=beta, weights_fixed_verified=frozen,
                          elapsed_seconds=time.monotonic()-started, final_scores=final_scores,
                          weighted_mean_reward=float(np.dot(final_rewards, serialized['weights'])),
                          unweighted_mean_reward=float(np.mean(final_rewards)),
                          best_sample=int(np.argmax(final_rewards)) + 1)
        write_json(condition_dir / 'fk-summary.json', serialized)
        results[label] = {'best_reward': max(final_rewards),
                          'mean_reward': serialized['unweighted_mean_reward'],
                          'weighted_mean_reward': serialized['weighted_mean_reward'],
                          'resampling_events': sum(h['resampled'] for h in result['history']),
                          'elapsed_seconds': serialized['elapsed_seconds']}
    write_json(args.output_dir / 'fk-comparison.json', results)
    print(json.dumps(results, indent=2), flush=True)
    return [args.output_dir / label for label in results]


if __name__ == '__main__':
    raise SystemExit('Use: python scripts/run_sampling.py experiment=2goo_fk')
