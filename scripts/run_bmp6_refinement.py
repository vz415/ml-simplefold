#!/usr/bin/env python3
"""Restrained BMP6 Cartesian cleanup; execution is restricted to Slurm.

Coordinates are restrained to the candidate, never to the experimental target.
The reference supplies only the observed-core mask and annotated native topology.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from xml.sax.saxutils import escape

import gemmi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_rosetta_refinement import canonicalize
from scripts.compare_structures import amino_acid


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verified_file(record, base):
    path = Path(record['path'])
    if not path.is_absolute():
        path = Path(base) / path
    if digest(path) != record['sha256']:
        raise ValueError(f'Input checksum mismatch: {path}')
    return path.resolve()


def prepare_input(original, topology, output):
    """Validate the entire construct, keep ATOM bytes, explicitly declare bonds."""
    structure = gemmi.read_structure(str(original))
    if len(structure) != 1:
        raise ValueError('Exactly one model is required')
    expected = topology['chain_masks']
    if [c.name for c in structure[0]] != topology['construct']['chain_order']:
        raise ValueError('Input chain order differs from the annotated construct')
    indices, sequences, seen, core, serials = {}, {}, set(), [], set()
    index = 0
    for chain in structure[0]:
        mask = expected[chain.name]
        sequence = ''
        for position, residue in enumerate(chain, 1):
            aa = amino_acid(residue.name)
            if aa is None or residue.seqid.num != position or residue.seqid.icode.strip():
                raise ValueError('Expected contiguous 1-based canonical protein residues')
            sequence += aa
            index += 1
            indices[(chain.name, position)] = (index, residue)
            if position in mask['observed_core_positions_1based']:
                core.append(index)
            names = set()
            for atom in residue:
                if atom.name in names or atom.serial in serials:
                    raise ValueError('Duplicate atom name or serial')
                names.add(atom.name)
                serials.add(atom.serial)
                if atom.altloc not in ('\x00', ' ') or atom.occ <= 0 or not all(
                        math.isfinite(v) for v in (*atom.pos, atom.occ)):
                    raise ValueError('Ambiguous or invalid atom coordinates/occupancy')
            if not {'N', 'CA', 'C', 'O'} <= names:
                raise ValueError(f'Missing backbone atom: {chain.name}:{position}')
        if len(sequence) != mask['length'] or hashlib.sha256(sequence.encode()).hexdigest() != mask['sequence_sha256']:
            raise ValueError(f'Construct sequence mismatch: {chain.name}')
        sequences[chain.name] = sequence
    if index != topology['construct']['total_residues'] or len(core) != topology['construct']['source_observed_residues']:
        raise ValueError('Construct/core residue count mismatch')
    pairs, headers, bonds = [], [], []
    for serial, bond in enumerate(topology['native_disulfides'], 1):
        endpoints = [(bond[side]['chain'], bond[side]['input_position_1based']) for side in ('left', 'right')]
        pose_pair = []
        for key in endpoints:
            pose_index, residue = indices[key]
            if key in seen or residue.name != 'CYS' or 'SG' not in residue:
                raise ValueError(f'Invalid native disulfide endpoint: {key}')
            seen.add(key)
            pose_pair.append(pose_index)
        left, right = endpoints
        headers.append(f'SSBOND {serial:3d} CYS {left[0]} {left[1]:4d}    CYS {right[0]} {right[1]:4d}\n')
        pairs.append(f'{pose_pair[0]} {pose_pair[1]}\n')
        bonds.append(dict(endpoints=endpoints, pose_indices=pose_pair, annotation_id=bond['id']))
    cysteines = {key for key, (_, residue) in indices.items() if residue.name == 'CYS'}
    if seen != cysteines or len(bonds) != topology['disulfide_count']:
        raise ValueError('Native topology does not cover the expected cysteines')
    lines = [line for line in Path(original).read_text().splitlines(keepends=True)
             if line.startswith(('ATOM  ', 'TER   '))]
    if not lines or sum(line.startswith('ATOM  ') for line in lines) != len(serials):
        raise ValueError('Unexpected non-ATOM protein records')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / 'input.pdb'
    prepared.write_text(''.join(headers + lines) + 'END\n')
    (output / 'disulfides.txt').write_text(''.join(pairs))
    metadata = dict(residue_count=index, core_pose_indices=core,
                    unrestrained_backbone_residues=index-len(core),
                    expected_sequences=sequences, disulfides=bonds,
                    coordinate_preparation='Original ATOM records preserved byte-for-byte')
    write_json(output / 'preparation.json', metadata)
    return metadata


def protocol_xml(core, script, sd=0.5):
    resnums = ','.join(map(str, core))
    return f'''<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="cart" weights="ref2015_cart">
      <Reweight scoretype="coordinate_constraint" weight="1.0"/>
      <Reweight scoretype="cart_bonded" weight="0.5"/>
      <Reweight scoretype="pro_close" weight="0.0"/>
    </ScoreFunction>
  </SCOREFXNS>
  <RESIDUE_SELECTORS><Index name="core" resnums="{resnums}"/></RESIDUE_SELECTORS>
  <TASKOPERATIONS>
    <RestrictToRepacking name="repack"/>
    <OperateOnResidueSubset name="core_for_constraints" selector="core" flip_subset="true">
      <PreventRepackingRLT/>
    </OperateOnResidueSubset>
  </TASKOPERATIONS>
  <MOVERS>
    <VirtualRoot name="root" removable="true"/>
    <AtomCoordinateCstMover name="core_cst" coord_dev="{sd}" sidechain="false"
                            task_operations="core_for_constraints"/>
    <FastRelax name="relax" scorefxn="cart" repeats="1" cartesian="true"
               disable_design="true" task_operations="repack"
               relaxscript="{escape(str(script), {chr(34): '&quot;'})}" ramp_down_constraints="false">
      <MoveMap bb="true" chi="true" jump="false"/>
    </FastRelax>
    <VirtualRoot name="unroot" remove="true"/>
  </MOVERS>
  <PROTOCOLS>
    <Add mover="root"/><Add mover="core_cst"/><Add mover="relax"/><Add mover="unroot"/>
  </PROTOCOLS>
  <OUTPUT scorefxn="cart"/>
</ROSETTASCRIPTS>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--case-index', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not args.prepare_only and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run refinement only through Slurm')
    config = json.loads(args.config.read_text())
    case = config['cases'][args.case_index]
    paths = {key: verified_file(value, args.config.parent)
             for key, value in (config['files'] | case['files']).items()}
    topology = json.loads(paths['topology'].read_text())
    from scripts.analyze_bmp6_refinement import strict_atoms
    _, atom_checks = strict_atoms(paths['original'], json.loads(paths['mapping'].read_text())['chains'])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, output / 'config.json')
    for name, path in paths.items():
        shutil.copy2(path, output / (name + path.suffix))
    metadata = prepare_input(paths['original'], topology, output / 'prepared')
    script = output / 'relax.script'
    script.write_text('repeat 1\nramp_repack_min 0.02 0.01 1.0\nramp_repack_min 0.25 0.01 1.0\n'
                      'ramp_repack_min 0.55 0.01 1.0\nramp_repack_min 1 0.00001 1.0\naccept_to_best\nendrepeat\n')
    xml = output / 'protocol.xml'
    xml.write_text(protocol_xml(metadata['core_pose_indices'], script, config['core_backbone_sd_angstrom']))
    receipt = dict(status='prepared', job_id=os.environ.get('SLURM_JOB_ID'), case=case,
                   source_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                   created_at=datetime.now(timezone.utc).isoformat(),
                   input_hashes={key: digest(path) for key, path in paths.items()},
                   original_atom_checks=atom_checks,
                   prepared_sha256=digest(output / 'prepared/input.pdb'),
                   protocol_sha256=digest(xml), relax_script_sha256=digest(script),
                   protocol=dict(core_backbone_sd_angstrom=config['core_backbone_sd_angstrom'],
                                 core_residues=len(metadata['core_pose_indices']),
                                 terminal_residues=metadata['unrestrained_backbone_residues'],
                                 sidechains='repack only, fixed sequences', jumps=False,
                                 coordinate_targets='candidate starting coordinates',
                                 topology_design_note='Use annotated pairs/core masks only; optional staged design in topology JSON is not executed.'))
    receipt_path = output / 'refinement-experiment.json'
    write_json(receipt_path, receipt)
    if args.prepare_only:
        print(f'Preparation verified: {output}')
        return
    started = time.monotonic()
    try:
        executable = shutil.which(config['executable'])
        if executable is None:
            raise FileNotFoundError(config['executable'])
        relaxed = output / 'relaxed'
        relaxed.mkdir()
        command = [executable, '-database', config['database'], '-parser:protocol', str(xml),
                   '-in:file:s', str(output / 'prepared/input.pdb'),
                   '-in:fix_disulf', str(output / 'prepared/disulfides.txt'), '-in:detect_disulf', 'false',
                   '-use_input_sc', '-nstruct', '1', '-run:constant_seed', '-run:jran', str(config['seed']),
                   '-out:file:renumber_pdb', 'false', '-out:path:all', str(relaxed),
                   '-out:file:scorefile', str(output / 'rosetta-score.sc')]
        receipt.update(status='running', command=command, executable=executable,
                       executable_sha256=digest(executable))
        write_json(receipt_path, receipt)
        with (output / 'rosetta.log').open('w') as log:
            subprocess.run(command, cwd=relaxed, stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt['rosetta_elapsed_seconds'] = time.monotonic() - started
        predictions = list(relaxed.glob('*.pdb'))
        if len(predictions) != 1:
            raise ValueError(f'Expected exactly one refined structure; got {len(predictions)}')
        refined = output / 'refined.pdb'
        mapping = canonicalize(predictions[0], refined, metadata['expected_sequences'])
        # Canonicalize preserves Rosetta atoms but drops its metadata; restore known topology.
        headers = [line for line in (output / 'prepared/input.pdb').read_text().splitlines(keepends=True)
                   if line.startswith('SSBOND')]
        refined.write_text(''.join(headers) + refined.read_text())
        write_json(output / 'canonicalization.json', mapping)
        analysis = [sys.executable, str(ROOT / 'scripts/analyze_bmp6_refinement.py'),
                    '--original', str(paths['original']), '--refined', str(refined),
                    '--original-evaluation', str(paths['original_evaluation']),
                    '--sample-index', str(case['source_sample']),
                    '--dockq-python', config['dockq_python'], '--output', str(output / 'comparison.json')]
        for key in ('mapping', 'reference', 'spec', 'parent_spec', 'fasta', 'topology'):
            analysis += ['--' + key.replace('_', '-'), str(paths[key])]
        receipt.update(status='analyzing', raw_refined_sha256=digest(predictions[0]), refined_sha256=digest(refined),
                       analysis_command=analysis)
        write_json(receipt_path, receipt)
        subprocess.run(analysis, check=True)
        receipt.update(status='complete', elapsed_seconds=time.monotonic() - started,
                       comparison_sha256=digest(output / 'comparison.json'))
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        write_json(receipt_path, receipt)
    print(f'Refinement and analysis complete: {output}', flush=True)


if __name__ == '__main__':
    main()
