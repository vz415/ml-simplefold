#!/usr/bin/env python3
"""Install a separate CPU validation environment through the approved Slurm launcher."""
import json
import hashlib
import os
from pathlib import Path
import subprocess

import hydra
from omegaconf import OmegaConf


def prepare_rotarama(python):
    subprocess.run([python, '-m', 'mmtbx.command_line.rebuild_rotarama_cache'], check=True)
    subprocess.run([python, '-c',
        'from mmtbx.rotamer.rotamer_eval import RotamerEval; '
        'from mmtbx.rotamer.ramachandran_eval import RamachandranEval; '
        'RotamerEval(); RamachandranEval(); print("Rotarama cache ready", flush=True)'], check=True)


@hydra.main(version_base='1.3', config_path='../configs', config_name='training_geometry')
def main(cfg):
    assert os.environ.get('SLURM_JOB_ID'), 'Install remote tools through Slurm'
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved-config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    python = str(Path(cfg.geometry_env) / 'bin/python')
    if cfg.get('rebuild_cache_only', False):
        prepare_rotarama(python)
        (output / 'completion.json').write_text(json.dumps(dict(state='cache_ready', geometry_env=cfg.geometry_env), indent=2)+'\n')
        return
    env = dict(os.environ, CONDA_PKGS_DIRS=str(Path.cwd() / 'artifacts/conda-cache-geometry'))
    command = ['conda', 'create', '--yes', '--solver=libmamba', '--override-channels',
               '--channel', 'conda-forge', '--prefix', cfg.geometry_env,
               f'python={cfg.python_version}', f'cctbx-base={cfg.cctbx_version}',
               'gemmi=0.7.3', 'hydra-core=1.3.2']
    if not (Path(cfg.geometry_env) / 'bin/python').exists():
        subprocess.run(command, check=True, env=env)
    asset = Path(cfg.artifact_root) / 'tools/molprobity' / cfg.chem_data_url.rsplit('/', 1)[-1]
    asset.parent.mkdir(parents=True, exist_ok=True)
    if not asset.exists():
        partial = asset.with_suffix('.partial')
        subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--connect-timeout',
                        '30', '--continue-at', '-', '--output', str(partial), cfg.chem_data_url], check=True)
        partial.rename(asset)
    checksum = hashlib.sha256()
    with asset.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    assert checksum.hexdigest() == cfg.chem_data_sha256, 'Chemistry data download checksum mismatch'
    subprocess.run(['conda', 'install', '--yes', '--prefix', cfg.geometry_env, str(asset)], check=True, env=env)
    subprocess.run([python, '-c', 'import mmtbx.validation, iotbx.pdb, gemmi, hydra; print("Geometry imports passed", flush=True)'], check=True)
    prepare_rotarama(python)
    packages = subprocess.check_output(['conda', 'list', '--explicit', '--prefix', cfg.geometry_env], text=True)
    (output / 'conda-explicit.txt').write_text(packages)
    (output / 'completion.json').write_text(json.dumps(dict(state='installed', geometry_env=cfg.geometry_env, command=command), indent=2)+'\n')
    print('Geometry environment ready:', cfg.geometry_env, flush=True)


if __name__ == '__main__':
    main()
