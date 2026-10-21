"""Integrity, resume and report contracts for existing-run evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from scripts.check_training_run import Check, ROOT, balanced, checked_file, identity, native_method, save, verify_fold_scores
from scripts.build_training_geometry_report import write_report


class TrainingCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        cfg = OmegaConf.load(ROOT / 'configs/training_check.yaml')
        cfg.run_id = '42'
        cfg.artifact_dir = str(self.root / 'artifacts')
        cfg.output_dir = str(self.root / 'output')
        cfg.notes.enabled = False
        self.check = Check(cfg)
        self.row = dict(cohort='complex', arm='original', id='target', sample=0,
                        sha256='pinned', groups={'A': 'ligand'}, path='/old/path.pdb')

    def test_identity_ignores_staging_paths_and_order_but_not_chain_groups(self):
        other = dict(self.row, arm='mixed', sha256='other')
        self.assertEqual(identity([self.row, other]), identity([other, dict(self.row, path='/new/path')]))
        changed = dict(self.row, groups={'A': 'wrong_receptor'})
        self.assertNotEqual(identity([self.row]), identity([changed]))

    def test_pairing_detects_missing_samples_and_seed_mismatches(self):
        a = dict(id='target', sample=0, seed=4, condition='pretrained')
        b = dict(a, condition='fine_tuned')
        balanced([a, b])
        with self.assertRaises(AssertionError):
            balanced([a, dict(b, seed=5)])
        with self.assertRaises(AssertionError):
            balanced([a, b, dict(b, sample=1)])
        with self.assertRaises(AssertionError):
            balanced([a, a, b, b])

    def test_changed_file_cannot_pass_pinned_hash_verification(self):
        from scripts.check_training_run import digest
        path = self.root / 'a.pdb'
        path.write_text('original')
        pinned = digest(path)
        checked_file(self.root, 'a.pdb', pinned)
        path.write_text('different')
        with self.assertRaises(AssertionError):
            checked_file(self.root, 'a.pdb', pinned)
        with self.assertRaises(AssertionError):
            checked_file(self.root, '../outside', pinned)

    def test_resume_does_not_submit_another_geometry_job(self):
        self.check.record(geometry_job='123')
        with patch.object(self.check, 'find_geometry', return_value=None), \
             patch.object(self.check, 'submit_geometry') as submit, \
             patch.object(self.check, 'collect_geometry', return_value=None) as collect:
            self.assertIsNone(self.check.geometry([self.row]))
        submit.assert_not_called()
        collect.assert_called_once_with('123')

    def test_general_scores_must_match_the_current_predictions(self):
        rows = [dict(id='a', sample=0, seed=7, condition=arm, sha256=arm,
                     checkpoint_sha256=arm, tm_score=0.5) for arm in ['pretrained', 'fine_tuned']]
        base = self.root / 'retention'
        save(base / 'prediction-receipt.json', rows)
        save(base / 'benchmark-results.json', rows)
        save(base / 'benchmark-summary.json', dict(means_and_medians={
            arm: dict(tm_score=dict(mean=0.5, median=0.5)) for arm in ['pretrained', 'fine_tuned']}))
        verify_fold_scores(base)
        save(base / 'benchmark-results.json', [dict(r, sha256='wrong') for r in rows])
        with self.assertRaises(AssertionError):
            verify_fold_scores(base)

    def test_failed_job_stops_without_an_automatic_replacement(self):
        with patch.object(self.check, 'remote', return_value='OUT_OF_MEMORY|'), \
             patch.object(self.check, 'submit_geometry') as submit:
            with self.assertRaisesRegex(RuntimeError, 'not resubmitted'):
                self.check.collect_geometry('123')
        submit.assert_not_called()

    def test_completed_geometry_requires_matching_inputs_and_scientific_method(self):
        base = self.check.artifacts / 'remote-runs/123'
        save(base / 'completion.json', dict(state='completed', backend='molprobity',
             method_hash=native_method('2026.9')))
        save(base / 'input-manifest.json', [self.row])
        self.assertEqual(self.check.find_geometry([self.row]), base)
        self.assertIsNone(self.check.find_geometry([dict(self.row, sha256='new')]))
        save(base / 'completion.json', dict(state='completed', backend='molprobity', method_hash='stale'))
        self.assertIsNone(self.check.find_geometry([self.row]))

    def test_offline_missing_geometry_never_contacts_hpc(self):
        self.check.cfg.offline = True
        with patch.object(self.check, 'remote') as remote:
            with self.assertRaisesRegex(AssertionError, 'No matching native geometry'):
                self.check.geometry([self.row])
        remote.assert_not_called()

    def test_collection_does_not_transfer_weights_or_path_traversals(self):
        for relative in ['weights.ckpt', '../outside.pdb', 'trajectories/a.json']:
            with self.subTest(relative=relative), \
                 patch.object(self.check, 'remote', return_value=json.dumps([
                     dict(relative=relative, sha256='bad')])), \
                 patch.object(self.check, 'transfer') as transfer:
                with self.assertRaises(AssertionError):
                    self.check.collect()
                transfer.assert_not_called()

    def test_verified_local_outputs_do_not_touch_shared_storage(self):
        with patch.object(self.check, 'local_ready', return_value=True), \
             patch.object(self.check, 'remote') as remote:
            self.check.collect()
        remote.assert_not_called()
        self.assertEqual(self.check.state['downloaded_files'], 0)

    def test_two_arm_report_counts_and_labels_are_not_hardcoded(self):
        report = self.root / 'report-input'
        rows = [dict(self.row, cohort=cohort, arm=arm,
                     metrics={'molprobity_clashscore': 2})
                for cohort in ['complex', 'cameo'] for arm in ['original', 'mixed']]
        groups = [dict(cohort=r['cohort'], arm=r['arm'], structures=1,
                       metrics={'molprobity_clashscore': dict(mean=2, std=0, n=1)}) for r in rows]
        save(report / 'geometry-results.json', rows)
        save(report / 'geometry-summary.json', dict(groups=groups))
        (report / 'geometry-results.csv').write_text('cohort,arm\n')
        output = self.root / 'page'
        write_report(report, output, 'Two models <no third arm>', 'Cropped mixed')
        page = (output / 'index.html').read_text()
        self.assertIn('1 synthetic training complexes', page)
        self.assertIn('1 experimental CAMEO targets', page)
        self.assertIn('Cropped mixed', page)
        self.assertIn('&lt;no third arm&gt;', page)
        self.assertNotIn('Both tuned models used', page)
        self.assertIn('Per-structure metrics and provenance', page)


if __name__ == '__main__':
    unittest.main()
