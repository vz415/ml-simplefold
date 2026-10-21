#!/usr/bin/env python3
"""Submit training -> GPU sampling -> CPU scoring, retaining one scientific run."""
import json
from pathlib import Path
import shlex
import subprocess

import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def stage_command(cfg, phase, commit, training_job=None, dependency=None):
    resources = cfg.stages[phase]
    command = ['sbatch', '--parsable', '--job-name=mixed-' + phase,
        '--partition=' + resources.partition, '--account=' + resources.account,
        '--qos=' + resources.qos, '--gres=' + resources.gres,
        f'--cpus-per-task={resources.cpus}', '--mem=' + resources.memory,
        '--time=' + resources.time]
    if dependency:
        command += ['--dependency=afterok:' + dependency, '--kill-on-invalid-dep=yes']
    command += [cfg.launcher, '--train-mixed', '--config-name=' + cfg.training_config,
                'phase=' + phase, 'expected_commit=' + commit, 'artifact_root=' + cfg.artifact_root]
    if phase == 'train':
        command += list(cfg.training_overrides)
    else:
        command += ['source_run=' + f'{cfg.artifact_root}/runs/{training_job}']
    return command


@hydra.main(version_base='1.3', config_path='../configs', config_name='mixed_pipeline')
def main(cfg):
    import yaml
    policy = yaml.safe_load((ROOT / '.agents/hpc-experiments.yaml').read_text())
    assert Path(cfg.launcher).name in policy['allowed_scripts']
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if cfg.submit:
        assert subprocess.check_output(['whoami'], text=True).strip() == 'ynkim4', 'Submit on hpc3y'
        assert Path.cwd() == ROOT, 'Run from the verified training checkout'
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
        upstream = subprocess.check_output(['git', 'rev-parse', '@{upstream}'], cwd=ROOT, text=True).strip()
        assert upstream == commit, 'Push this source revision first'
    receipt = dict(source_commit=commit, submitted=bool(cfg.submit), stages=[])
    destination = Path(cfg.receipt_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'resolved-config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    previous, training_job = None, None
    for phase in ['train', 'sample', 'score']:
        command = stage_command(cfg, phase, commit, training_job, previous)
        print(shlex.join(command), flush=True)
        job = (subprocess.check_output(command, cwd=ROOT, text=True).strip().split(';')[0]
               if cfg.submit else {'train': 'TRAIN_JOB', 'sample': 'SAMPLE_JOB', 'score': 'SCORE_JOB'}[phase])
        if cfg.submit:
            assert job.isdigit(), job
        if phase == 'train':
            training_job = job
        receipt['stages'].append(dict(phase=phase, job_id=job, command=command))
        # Persist after every submission so partial submission cannot lose job IDs.
        (destination / 'submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
        previous = job
    print(f'After completion: python scripts/check_training_run.py run_id={training_job} '
          f'completion_job_id={previous} wait_for_training=true', flush=True)


if __name__ == '__main__':
    main()
