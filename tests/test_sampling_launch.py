"""Configuration, provenance and launch routing checks; never run a model."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_sampling
from sampling_preflight import StoragePreflight, prepare_output


def config(*overrides, hydra=False):
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'configs')):
        return compose(config_name='sample', overrides=list(overrides), return_hydra_config=hydra)


class SamplingLaunchTests(unittest.TestCase):
    def test_profiles_and_overrides_compose(self):
        standard = config('experiment=2goo_3b')
        run_sampling.validate_config(standard)
        self.assertEqual((standard.model, standard.samples, standard.sampling.mode),
                         ('simplefold_3B', 10, 'standard'))
        fk = config('experiment=2goo_fk', 'seed=9', 'sampling.beta=1.5',
                    'paths.cache_dir=/tmp/test-simplefold-cache')
        run_sampling.validate_config(fk)
        self.assertEqual((fk.seed, fk.sampling.beta), (9, 1.5))
        self.assertEqual(fk.paths.checkpoint_dir, '/tmp/test-simplefold-cache/checkpoints')
        self.assertEqual(fk.paths.ccd_dir, '/tmp/test-simplefold-cache/ccd')

    def test_individual_cache_override_and_invalid_particle_count_rejected(self):
        with self.assertRaisesRegex(ValueError, 'derived'):
            run_sampling.validate_config(config('paths.checkpoint_dir=/some/other/place'))
        with self.assertRaisesRegex(ValueError, 'at least two'):
            run_sampling.validate_config(config('experiment=2goo_fk', 'samples=1'))

    def test_cli_preview_is_safe_and_execution_refused_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items() if k != 'SLURM_JOB_ID'}
            command = [sys.executable, str(ROOT / 'scripts/run_sampling.py'),
                       'experiment=2goo_fk', f'paths.artifact_dir={tmp}']
            preview = subprocess.run(command + ['--cfg', 'job', '--resolve'], env=env,
                                     capture_output=True, text=True, timeout=20)
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertIn('mode: fk', preview.stdout)
            refused = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('Slurm GPU job', refused.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def exercise_task(self, profile, nested=False):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(f'experiment={profile}', f'paths.artifact_dir={tmp}')
            output = Path(tmp) / 'runs' / '999'
            if nested:
                output /= '7'
            output.mkdir(parents=True)
            hydra_state = SimpleNamespace(runtime=SimpleNamespace(output_dir=str(output)),
                                          overrides=SimpleNamespace(task=[f'experiment={profile}']))
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
                                                             get_device_name=lambda i: 'test GPU'))
            fake_fk = SimpleNamespace(run_fk=lambda *args: None)
            with patch.dict(os.environ, {'SLURM_JOB_ID': '999'}), \
                    patch.dict(sys.modules, {'torch': fake_torch, 'sample_complex_fk': fake_fk}), \
                    patch.object(run_sampling.HydraConfig, 'get', return_value=hydra_state), \
                    patch.object(run_sampling.subprocess, 'check_output', return_value='abc123\n'), \
                    patch.object(run_sampling.subprocess, 'run') as launched, \
                    patch.object(run_sampling, 'run_python') as helper, \
                    patch('builtins.print'), \
                    patch.object(fake_fk, 'run_fk') as fk:
                run_sampling.main.__wrapped__(cfg)
                metadata = json.loads((output / 'experiment.json').read_text())
                resolved = OmegaConf.load(output / 'resolved-config.yaml')
                self.assertEqual(metadata['output_dir'], str(output))
                self.assertEqual(metadata['source_commit'], 'abc123')
                self.assertEqual(metadata['hydra_overrides'], [f'experiment={profile}'])
                self.assertEqual(resolved.seed, cfg.seed)
                calls = [call.args for call in helper.call_args_list]
                self.assertEqual(calls[0][0], 'download_weights.py')
                validations = [call for call in calls if call[0] == 'validate_predictions.py']
                if cfg.sampling.mode == 'fk':
                    args, sampling = fk.call_args.args
                    self.assertEqual(args.output_dir, output)
                    self.assertEqual(args.seed, cfg.seed)
                    self.assertEqual(sampling['beta'], cfg.sampling.beta)
                    self.assertEqual(len(validations), 2)
                    for condition, call in zip(('baseline', 'fk'), validations):
                        self.assertEqual(call[4], output / condition / f'predictions_{cfg.model}')
                else:
                    fk.assert_not_called()
                    self.assertEqual(len(validations), 1)
                    command = launched.call_args.args[0]
                    self.assertEqual(command[0], 'simplefold')
                    self.assertEqual(command[command.index('--output_dir') + 1], str(output))

    def test_standard_task_records_config_and_routes_output(self):
        self.exercise_task('2goo_3b')

    def test_fk_task_uses_hydra_runtime_output_in_sweep_subdirectory(self):
        self.exercise_task('2goo_fk', nested=True)

    def test_storage_preparation_preserves_existing_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'job'
            output.mkdir()
            receipt = output / 'multirun.yaml'
            receipt.write_text('preserve me\n')
            with self.assertRaisesRegex(RuntimeError, 'Existing run receipt'):
                prepare_output([tmp], output)
            self.assertEqual(receipt.read_text(), 'preserve me\n')
            with self.assertRaises(FileNotFoundError):
                prepare_output([Path(tmp) / 'missing'], Path(tmp) / 'new-job')
            self.assertFalse((Path(tmp) / 'new-job').exists())

    def test_storage_callback_failure_stops_before_hydra_outputs(self):
        cfg = config(hydra=True)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '999'}), \
                patch('sampling_preflight.subprocess.run',
                      side_effect=subprocess.CalledProcessError(124, 'timeout')):
            with self.assertRaisesRegex(SystemExit, 'preflight failed'):
                StoragePreflight().on_run_start(cfg)

    def test_sweep_preflight_rejects_swept_storage_and_colliding_directories(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '999'}):
            swept = config(hydra=True)
            swept.hydra.overrides.task.append('paths.cache_dir=/tmp/a,/tmp/b')
            with self.assertRaisesRegex(SystemExit, 'Keep storage'):
                StoragePreflight().on_multirun_start(swept)
            colliding = config('hydra.sweep.subdir=same-directory', hydra=True)
            with self.assertRaisesRegex(SystemExit, 'sweep subdirectories'):
                StoragePreflight().on_multirun_start(colliding)

    def test_sweep_allows_steering_strength_with_shared_storage_and_mode(self):
        cfg = config('experiment=2goo_fk', hydra=True)
        cfg.hydra.overrides.task.append('sampling.beta=1.0,2.0')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '999'}), \
                patch('sampling_preflight.subprocess.run') as launched:
            StoragePreflight().on_multirun_start(cfg)
        launched.assert_called_once()
        command = launched.call_args.args[0]
        self.assertEqual(command[:3], ['timeout', '-k', '5s'])
        self.assertIn('/runs/999', json.loads(command[-1])['output'])

    def test_sweep_rejects_task_dependent_output_or_cache_interpolation(self):
        for override in ('hydra.sweep.dir=/tmp/runs/${seed}',
                         'paths.cache_dir=/tmp/cache/${seed}'):
            cfg = config(override, hydra=True)
            with patch.dict(os.environ, {'SLURM_JOB_ID': '999'}), \
                    patch('sampling_preflight.subprocess.run') as launched:
                with self.assertRaisesRegex(SystemExit, 'paths must use only'):
                    StoragePreflight().on_multirun_start(cfg)
            launched.assert_not_called()


if __name__ == '__main__':
    unittest.main()
