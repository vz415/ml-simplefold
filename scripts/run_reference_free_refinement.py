"""Restrained Rosetta relaxation with explicit sequence-mapped disulfides.

No experimental docking reference or folding model is used. Known conserved
covalent connectivity is supplied separately from the predicted coordinates.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import gemmi
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.structure_clashes import analyze_clashes
from scripts.compare_structures import kabsch
from scripts.run_rosetta_refinement import relax_script
from scripts.analyze_rosetta_refinement import canonicalize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def with_topology(source, topology, output):
    structure = gemmi.read_structure(str(source))
    headers, pose_pairs = [], []
    offsets, offset = {}, 0
    for chain in structure[0]:
        sequence = gemmi.one_letter_code([r.name for r in chain])
        assert sequence == topology['chain_sequences'][chain.name]
        offsets[chain.name] = offset
        offset += len(chain)
    for serial, pair in enumerate(topology['pairs'], 1):
        (first, i), (second, j) = pair
        for chain, number in pair:
            residue = structure[0][chain][number-1]
            assert residue.name == 'CYS' and 'SG' in residue
        headers.append(f'SSBOND {serial:3d} CYS {first} {i:4d}    CYS {second} {j:4d}\n')
        pose_pairs.append([offsets[first]+i, offsets[second]+j])
    lines = [line+'\n' for line in Path(source).read_text().splitlines()
             if line.startswith(('ATOM  ', 'TER   '))]
    Path(output).write_text(''.join(headers+lines)+'END\n')
    return pose_pairs


def coordinates(path, atom_name='CA'):
    structure = gemmi.read_structure(str(path))
    return np.array([[a.pos.x,a.pos.y,a.pos.z] for c in structure[0]
                     for r in c for a in r if a.name == atom_name])


def measure(path, topology, groups):
    structure = gemmi.read_structure(str(path))
    distances = []
    for (a,i),(b,j) in topology['pairs']:
        first,second = structure[0][a][i-1]['SG'][0],structure[0][b][j-1]['SG'][0]
        distances.append(dict(first=[a,i],second=[b,j],
                             distance_angstrom=first.pos.dist(second.pos)))
    return dict(clashes=analyze_clashes(path, chain_components=groups),
                disulfides=distances,
                disulfide_outliers_0_1A=sum(abs(d['distance_angstrom']-2.05)>.1 for d in distances))


@hydra.main(version_base='1.3', config_path='../configs/refinement', config_name='bmp4_rosetta_sample2')
def main(cfg):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit refinement through Slurm')
    output = Path(HydraConfig.get().runtime.output_dir)
    original = Path(cfg.input_pdb)
    topology = json.loads(Path(cfg.topology_file).read_text())
    assert digest(original) == topology['source_pdb_sha256']
    groups = OmegaConf.to_container(cfg.chain_components, resolve=True)
    (output/'resolved-config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    shutil.copy2(original, output/'original.pdb')
    shutil.copy2(cfg.topology_file, output/'disulfide-topology.json')
    prepared = output/'prepared.pdb'
    pairs = with_topology(original, topology, prepared)
    assert np.array_equal(coordinates(original), coordinates(prepared))
    disulfides = output/'disulfides.txt'
    disulfides.write_text(''.join(f'{i} {j}\n' for i,j in pairs))
    protocol = output/'relax.script'
    protocol.write_text(relax_script(OmegaConf.to_container(cfg.rosetta, resolve=True)))
    relaxed = output/'relaxed'
    relaxed.mkdir(exist_ok=True)
    executable = shutil.which(cfg.rosetta.executable)
    command = [executable, '-database', cfg.rosetta.database, '-in:file:s', str(prepared),
               '-in:fix_disulf', str(disulfides), '-in:detect_disulf', 'false',
               '-score:weights', cfg.rosetta.score_weights,
               '-relax:cartesian', '-relax:constrain_relax_to_start_coords',
               '-relax:coord_cst_stdev', str(cfg.rosetta.backbone_constraint_sd_angstrom),
               '-relax:ramp_constraints', 'false', '-relax:jump_move', 'false',
               '-relax:script', str(protocol), '-use_input_sc', '-nstruct', '1',
               '-run:constant_seed', '-run:jran', str(cfg.rosetta.seed),
               '-out:file:renumber_pdb', 'false', '-out:path:all', str(relaxed),
               '-out:file:scorefile', str(output/'rosetta-score.sc')]
    commit = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    receipt = dict(status='running', job_id=os.environ['SLURM_JOB_ID'],
                   name=cfg.name, source_job=cfg.source_job, source_sample=cfg.source_sample,
                   source_saved_index=cfg.source_sample, source_sha256=digest(original),
                   source_commit=commit, topology_sha256=digest(cfg.topology_file),
                   prepared_sha256=digest(prepared), command=command,
                   config=OmegaConf.to_container(cfg,resolve=True),
                   executable_sha256=digest(executable), reference_free=True)
    write_json(output/'refinement-experiment.json',receipt)
    before = measure(prepared,topology,groups)
    write_json(output/'before.json',before)
    start=time.monotonic()
    with (output/'rosetta.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    result, = relaxed.glob('*.pdb')
    canonical = output/'refined.pdb'
    temporary = output/'refined-canonical.pdb'
    canonicalize(result,temporary,topology['chain_sequences'])
    with_topology(temporary,topology,canonical)
    after = measure(canonical,topology,groups)
    moving,fixed=coordinates(canonical),coordinates(original)
    rotation,translation,distances=kabsch(moving,fixed)
    comparison=dict(before=before,after=after,
                    raw_original=measure(original,topology,groups),
                    assembly_CA_RMSD_fixed_frame_angstrom=float(np.sqrt(np.mean(np.sum((moving-fixed)**2,axis=1)))),
                    assembly_CA_RMSD_after_global_fit_angstrom=float(np.sqrt(np.mean(distances**2))),
                    overlap_reduction=before['clashes']['clash_count']-after['clashes']['clash_count'],
                    note='Prepared and refined clash counts share declared chemical topology. Raw-original counts can differ because its disulfides were not declared. No DockQ, native-contact accuracy or affinity claim.')
    write_json(output/'comparison.json',comparison)
    receipt.update(status='complete', rosetta_elapsed_seconds=time.monotonic()-start,
                   refined_file='refined.pdb',refined_sha256=digest(canonical))
    write_json(output/'refinement-experiment.json',receipt)
    print(f'Refinement complete: {output}',flush=True)


if __name__=='__main__':
    main()
