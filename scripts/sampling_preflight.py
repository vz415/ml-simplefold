"""Bounded storage checks before Hydra creates logs or overwrites receipts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.experimental.callback import Callback
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


def prepare_output(paths, output):
    """Run only in the timeout-controlled child; every filesystem call is bounded."""
    for path in map(Path, paths):
        path.stat()
        print(f'Accessible storage: {path}', flush=True)
    output = Path(output)
    for name in ('.hydra', 'experiment.json', 'resolved-config.yaml', 'multirun.yaml'):
        if (output / name).exists():
            raise RuntimeError(f'Existing run receipt: {output / name}; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)


class StoragePreflight(Callback):
    def _prepare(self, config, sweep=False):
        if not os.environ.get('SLURM_JOB_ID'):
            raise SystemExit('Sampling requires a Slurm GPU job')
        try:
            from run_sampling import validate_config
            validate_config(config)
            if sweep:
                overrides = OverridesParser.create().parse_overrides(config.hydra.overrides.task)
                for override in overrides:
                    key = override.key_or_group
                    if override.is_sweep_override() and (key.startswith('paths.') or key in
                            ('paths', 'experiment', 'sampling', 'sampling.mode')):
                        raise ValueError('Keep storage and sampling mode shared within one sweep')
                subdir = OmegaConf.to_container(config.hydra.sweep, resolve=False)['subdir']
                if subdir != '${hydra.job.num}':
                    raise ValueError('Use the default hydra.job.num sweep subdirectories')
                raw_paths = OmegaConf.to_container(config.paths, resolve=False)
                raw_paths['sweep_dir'] = OmegaConf.to_container(config.hydra.sweep, resolve=False)['dir']
                for value in raw_paths.values():
                    # Inspect nested expressions from the inside out. A swept
                    # task value must not move a later job outside the checked root.
                    while references := re.findall(r'\$\{([^${}]+)\}', value):
                        for reference in references:
                            if not reference.startswith(('paths.', 'oc.env:')):
                                raise ValueError('Sweep storage/output paths must use only paths.* or oc.env references')
                            value = value.replace('${' + reference + '}', 'static')
            paths = config.paths
            checks = [paths.checkpoint_dir,
                      f'{paths.cache_dir}/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt',
                      f'{paths.ccd_dir}/ccd.pkl']
            if config.sampling.mode == 'fk':
                checks += [paths.reference_cif, paths.reference_pdb]
            output = config.hydra.sweep.dir if sweep else config.hydra.run.dir
            payload = json.dumps({'paths': [to_absolute_path(p) for p in checks],
                                  'output': to_absolute_path(output)})
            subprocess.run(['timeout', '-k', '5s', str(config.storage_timeout_seconds),
                            sys.executable, str(Path(__file__).absolute()), payload], check=True)
        except Exception as error:
            # Hydra catches ordinary callback exceptions and only warns.
            # SystemExit is required to stop before it writes to storage.
            raise SystemExit(f'Sampling preflight failed: {error}') from error

    def on_run_start(self, config, **kwargs):
        self._prepare(config)

    def on_multirun_start(self, config, **kwargs):
        self._prepare(config, sweep=True)


if __name__ == '__main__':
    payload = json.loads(sys.argv[1])
    prepare_output(payload['paths'], payload['output'])
