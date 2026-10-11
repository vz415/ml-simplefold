#!/usr/bin/env python3
"""Refine selected protein coordinates through Slurm, without loading a model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import gemmi
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fk_2h62_l61_reward import L61ComplexReward
from scripts.fk_2h62_reward import COMPONENTS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_input(original, reference_cif, reference_pdb, settings, output):
    """Preserve coordinates and sequences; declare reference disulfide topology."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reward = L61ComplexReward(reference_cif, reference_pdb, settings)
    chains = reward._read_prediction(original)
    structure = gemmi.read_structure(str(original))
    residues = {chain.name: [residue for residue in chain if residue.name != 'HOH']
                for chain in structure[0]}
    pose_indices, index = {}, 0
    for chain, records in residues.items():
        for position, residue in zip(chains[chain].full_positions, records, strict=True):
            index += 1
            pose_indices[(chain, position)] = (index, residue)
    pairs, headers, topology = [], [], []
    for serial, (first, second) in enumerate(reward.fixed_disulfides, 1):
        i, a = pose_indices[first]
        j, b = pose_indices[second]
        if a.name != 'CYS' or b.name != 'CYS' or 'SG' not in a or 'SG' not in b:
            raise ValueError(f'Expected native cysteine atoms missing: {first}, {second}')
        pairs.append(f'{i} {j}\n')
        headers.append(f'SSBOND {serial:3d} CYS {first[0]} {a.seqid.num:4d}    CYS {second[0]} {b.seqid.num:4d}\n')
        topology.append(dict(pose_indices=[i, j],
                             original_positions=[[first[0], first[1] + 1], [second[0], second[1] + 1]],
                             pdb_residues=[[first[0], str(a.seqid)], [second[0], str(b.seqid)]]))
    # No coordinate regeneration or terminal extension during preparation.
    lines = [line for line in Path(original).read_text().splitlines(keepends=True)
             if line.startswith(('ATOM  ', 'TER   '))]
    prepared = output / 'input.pdb'
    prepared.write_text(''.join(headers + lines) + 'END\n')
    disulfides = output / 'disulfides.txt'
    disulfides.write_text(''.join(pairs))
    (output / 'topology.json').write_text(json.dumps(dict(
        residue_count=index, disulfides=topology,
        chain_components=COMPONENTS,
        sequence_spans=settings.get('prediction_sequence_spans', {}),
        preparation='Original heavy-atom coordinates retained; native disulfides explicitly declared.'), indent=2) + '\n')
    return prepared, disulfides


def relax_script(settings):
    lines = [f"repeat {settings['repeats']}"]
    for i, weight in enumerate(settings['repulsive_weights']):
        tolerance = (settings['final_minimizer_tolerance']
                     if i == len(settings['repulsive_weights']) - 1
                     else settings['minimizer_tolerance'])
        # Explicit 1.0 keeps constraints active in every custom-script stage.
        lines.append(f'ramp_repack_min {weight:g} {tolerance:g} 1.0')
    return '\n'.join(lines + ['accept_to_best', 'endrepeat', ''])


def analyze(original, prepared, refined, cfg, source_receipt, output):
    subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_rosetta_refinement.py'),
                    '--original', str(original), '--prepared', str(prepared),
                    '--refined', str(refined), '--reference-cif', str(cfg.reference_cif),
                    '--reference-pdb', str(cfg.reference_pdb),
                    '--source-receipt', str(source_receipt), '--output', str(output)], check=True)


@hydra.main(version_base='1.3', config_path='../configs/refinement', config_name='2h62_rosetta_pilot')
def main(cfg: DictConfig):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Rosetta refinement runs only through Slurm')
    case = cfg.cases[int(cfg.case_index)]
    source = Path(cfg.artifact_dir) / 'runs' / case.source_job
    original = source / 'fk/predictions_simplefold_3B' / case.source_file
    source_receipt = source / 'fk-experiment.json'
    settings = json.loads(source_receipt.read_text())['config']['reward']
    output = Path(HydraConfig.get().runtime.output_dir)
    if (output / 'refinement-experiment.json').exists():
        raise RuntimeError(f'Refinement receipt already exists: {output}')
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    (output / 'resolved-config.yaml').write_text(resolved)
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    (output / 'git-commit.txt').write_text(commit + '\n')
    shutil.copy2(original, output / 'original.pdb')
    shutil.copy2(source_receipt, output / 'source-fk-experiment.json')
    prepared, disulfides = prepare_input(original, cfg.reference_cif, cfg.reference_pdb,
                                         settings, output / 'prepared')
    protocol = output / 'relax.script'
    protocol.write_text(relax_script(OmegaConf.to_container(cfg.rosetta, resolve=True)))
    relaxed = output / 'relaxed'
    relaxed.mkdir()
    executable = shutil.which(cfg.rosetta.executable)
    if executable is None:
        raise FileNotFoundError(cfg.rosetta.executable)
    command = [executable, '-database', cfg.rosetta.database,
               '-in:file:s', str(prepared), '-in:fix_disulf', str(disulfides),
               '-in:detect_disulf', 'false',
               '-score:weights', cfg.rosetta.score_weights,
               '-relax:cartesian', '-relax:constrain_relax_to_start_coords',
               '-relax:coord_cst_stdev', str(cfg.rosetta.backbone_constraint_sd_angstrom),
               '-relax:ramp_constraints', 'false', '-relax:jump_move', 'false',
               '-relax:script', str(protocol), '-use_input_sc', '-nstruct', '1',
               '-run:constant_seed', '-run:jran', str(cfg.rosetta.seed),
               '-out:file:renumber_pdb', 'false', '-out:path:all', str(relaxed),
               '-out:file:scorefile', str(output / 'rosetta-score.sc')]
    receipt = dict(job_id=os.environ['SLURM_JOB_ID'],
                   array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                   name=cfg.name, case_index=int(cfg.case_index),
                   source_job=case.source_job, source_sample=case.source_sample,
                   source_file=str(original), source_sha256=digest(original),
                   source_receipt_sha256=digest(source_receipt), source_commit=commit,
                   prepared_sha256=digest(prepared), command=command,
                   config=OmegaConf.to_container(cfg, resolve=True),
                   hydra_overrides=list(HydraConfig.get().overrides.task),
                   executable=executable, executable_sha256=digest(executable),
                   topology_file='prepared/topology.json', status='running')
    (output / 'refinement-experiment.json').write_text(json.dumps(receipt, indent=2) + '\n')
    # Fail before expensive refinement if coordinate/topology preparation changed metrics.
    analyze(original, prepared, prepared, cfg, source_receipt, output / 'preparation-check.json')
    preparation = json.loads((output / 'preparation-check.json').read_text())
    delta = preparation['comparisons']['original_to_prepared']
    if (abs(delta['reward_delta']) > 1e-10 or
            any(abs(value) > 1e-10 for value in delta['geometry_deltas'].values()
                if isinstance(value, (int, float))) or
            preparation['stages']['prepared']['drift_from_original']['assembly_CA_RMSD_angstrom'] > 1e-10):
        raise RuntimeError('Coordinate preparation changed the original structure metrics')
    print(f'Refining source {case.source_job}, displayed sample {case.source_sample}', flush=True)
    start = time.monotonic()
    with (output / 'rosetta.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    predictions = list(relaxed.glob('*.pdb'))
    if len(predictions) != 1:
        raise RuntimeError(f'Expected one refined structure, found {len(predictions)}')
    analyze(original, prepared, predictions[0], cfg, source_receipt, output / 'comparison.json')
    receipt.update(status='complete', rosetta_elapsed_seconds=time.monotonic() - start,
                   refined_file=str(predictions[0].relative_to(output)),
                   refined_sha256=digest(predictions[0]))
    (output / 'refinement-experiment.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Refinement and comparison complete: {output}', flush=True)


if __name__ == '__main__':
    inspection = any(arg.split('=', 1)[0] in {'--cfg', '-c', '--help', '-h', '--info', '-i'}
                     for arg in sys.argv[1:])
    if not os.environ.get('SLURM_JOB_ID') and not inspection:
        raise SystemExit('Submit scripts/hpc_refine.slurm; config inspection is safe locally.')
    main()
