"""Stage boundaries must preserve settings and never turn CPU scoring into inference."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hydra
from omegaconf import OmegaConf

from scripts.submit_mixed_pipeline import stage_command
from scripts import train_mixed
from scripts.train_mixed import ROOT, digest
from scripts.check_training_run import Check


class MixedPipelineTests(unittest.TestCase):
    def test_job_dependencies_and_gpu_cpu_resources(self):
        cfg = OmegaConf.load(ROOT / 'configs/mixed_pipeline.yaml')
        cfg.training_overrides = ['max_steps=42', 'evaluation.tau=0.8']
        train = stage_command(cfg, 'train', 'pinned')
        sample = stage_command(cfg, 'sample', 'pinned', '123', '123')
        score = stage_command(cfg, 'score', 'pinned', '123', '124')
        for command in [train, sample]:
            self.assertIn('--gres=gpu:A100:1', command)
        self.assertIn('--gres=none', score)
        self.assertIn('--dependency=afterok:123', sample)
        self.assertIn('--dependency=afterok:124', score)
        for command in [sample, score]:
            self.assertIn('source_run=' + cfg.artifact_root + '/runs/123', command)
            self.assertNotIn('max_steps=42', command)
            self.assertIn('expected_commit=pinned', command)
        self.assertIn('max_steps=42', train)

    def scoring_run(self, directory):
        selection = ROOT / 'configs/datasets/multimer_refined_v1.json'
        source = OmegaConf.create(dict(model_name='3B', selection=str(selection),
            max_steps=600, retention=dict(enabled=True)))
        (directory / 'resolved-config.yaml').write_text(OmegaConf.to_yaml(source))
        files = {
            'experiment.json': dict(synthetic_selection_sha256=digest(selection)),
            'training-summary.json': dict(model_name='3B', source_counts=dict(pdb=540, synthetic=60)),
            'sampling-completion.json': dict(state='sampling_completed'),
            'prediction-receipt.json': [dict(id='complex')],
        }
        retention = directory / 'retention'
        retention.mkdir()
        benchmark = dict(selection=str(selection), sampling=dict(tau=0.01))
        (retention / 'resolved-config.yaml').write_text(OmegaConf.to_yaml(benchmark))
        (retention / 'experiment.json').write_text(json.dumps(dict(selection_sha256=digest(selection))))
        files['training-completion.json'] = dict(state='training_completed', config_sha256=digest(directory / 'resolved-config.yaml'))
        files['sampling-completion.json']['retention_config_sha256'] = digest(retention / 'resolved-config.yaml')
        for name, values in files.items():
            (directory / name).write_text(json.dumps(values))
        return OmegaConf.create(dict(phase='score', source_run=str(directory), expected_commit=None))

    def test_cpu_scoring_uses_saved_settings_without_cuda_or_model_construction(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cfg = self.scoring_run(directory)
            cfg.evaluation = dict(tau=0.99)  # ignored: settings come from the sampled run
            with patch.dict(os.environ, SLURM_JOB_ID='cpu-job'), \
                 patch('torch.cuda.is_available', side_effect=AssertionError('CPU phase touched CUDA')), \
                 patch.object(hydra.utils, 'instantiate', side_effect=AssertionError('CPU phase built a model')), \
                 patch.object(train_mixed.subprocess, 'check_output', return_value='pinned'), \
                 patch('run_cameo_benchmark.score') as score:
                train_mixed.main.__wrapped__(cfg)
            self.assertEqual(score.call_args.args[0].sampling.tau, 0.01)
            completion = json.loads((directory / 'completion.json').read_text())
            self.assertEqual(completion['state'], 'completed')
            self.assertEqual(completion['source_counts'], dict(pdb=540, synthetic=60))
            self.assertEqual(json.loads((directory / 'scoring-completion.json').read_text())['job_id'], 'cpu-job')

    def test_cpu_scoring_failure_cannot_mark_run_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cfg = self.scoring_run(directory)
            with patch.dict(os.environ, SLURM_JOB_ID='cpu-job'), \
                 patch('run_cameo_benchmark.score', side_effect=RuntimeError('scoring failed')):
                with self.assertRaisesRegex(RuntimeError, 'scoring failed'):
                    train_mixed.main.__wrapped__(cfg)
            self.assertFalse((directory / 'completion.json').exists())
            self.assertFalse((directory / 'scoring-completion.json').exists())

    def test_checker_waits_for_cpu_job_while_retaining_training_run_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = OmegaConf.load(ROOT / 'configs/training_check.yaml')
            cfg.run_id = '123'
            cfg.completion_job_id = '125'
            cfg.artifact_dir = str(Path(temporary) / 'artifacts')
            cfg.output_dir = str(Path(temporary) / 'report')
            check = Check(cfg)
            with patch.object(check, 'remote', return_value='COMPLETED|') as remote:
                self.assertTrue(check.wait_for_training())
            self.assertIn('-j 125', remote.call_args.args[0])
            self.assertEqual(check.run.name, '123')


if __name__ == '__main__':
    unittest.main()
