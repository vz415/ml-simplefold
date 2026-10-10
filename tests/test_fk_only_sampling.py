"""Condition and reward routing regressions; never initialize a folding model."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.sample_complex_fk import make_reward, sampling_conditions, score_existing_baseline


class FKOnlySamplingTests(unittest.TestCase):
    def test_fk_only_does_not_include_another_baseline_condition(self):
        self.assertEqual(sampling_conditions({"beta": 2, "run_baseline": False}), [("fk", 2.)])

    def test_legacy_configs_keep_matched_baseline_and_fk(self):
        self.assertEqual(sampling_conditions({"beta": 2}), [("baseline", 0.), ("fk", 2.)])
        self.assertEqual(sampling_conditions({"beta": 3, "run_baseline": True}),
                         [("baseline", 0.), ("fk", 3.)])

    def test_zero_beta_fk_only_still_runs_exactly_one_named_condition(self):
        self.assertEqual(sampling_conditions({"beta": 0, "run_baseline": False}), [("fk", 0.)])

    def test_partial_reference_reward_dispatch_avoids_complete_2goo_reward(self):
        partial, complete = Mock(), Mock()
        args = argparse.Namespace(reference_cif=Path("observed.cif"), reference_pdb=Path("observed.pdb"))
        settings = {"positive_weights": {"observed_contacts": 1}}
        with patch.dict(sys.modules, {
            "fk_2h62_reward": SimpleNamespace(PartialComplexReward=partial),
            "fk_complex_reward": SimpleNamespace(ComplexReward=complete),
        }):
            result = make_reward(args, {"reward_kind": "2h62_partial", "reward": settings})
        self.assertIs(result, partial.return_value)
        partial.assert_called_once_with(args.reference_cif, args.reference_pdb, settings)
        complete.assert_not_called()

    def test_legacy_reward_dispatch_retains_complete_reference_reward(self):
        partial, complete = Mock(), Mock()
        args = argparse.Namespace(reference_cif=Path("reference.cif"), reference_pdb=Path("reference.pdb"))
        settings = {"positive_weights": {"native_contacts": 1}}
        with patch.dict(sys.modules, {
            "fk_2h62_reward": SimpleNamespace(PartialComplexReward=partial),
            "fk_complex_reward": SimpleNamespace(ComplexReward=complete),
        }):
            result = make_reward(args, {"reward": settings})
        self.assertIs(result, complete.return_value)
        complete.assert_called_once_with(args.reference_cif, args.reference_pdb, settings)
        partial.assert_not_called()

    def test_missing_baseline_is_recorded_without_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reward = Mock()
            score_existing_baseline(reward, root / 'pending-baseline', root)
            receipt = json.loads((root / 'existing-baseline-scores.json').read_text())
            self.assertEqual(receipt['status'], 'not_available_at_launch')
            self.assertEqual(receipt['samples'], [])
            reward.score.assert_not_called()

    def test_existing_baseline_scoring_preserves_pdbs_and_numeric_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / 'predictions'
            predictions.mkdir()
            for number in (10, 2, 0):
                (predictions / f'target_sampled_{number}.pdb').write_text(f'original {number}\n')
            before = {path: path.read_bytes() for path in predictions.iterdir()}
            reward = Mock()
            reward.score.return_value = {'reward': 1.5}
            score_existing_baseline(reward, predictions, root)
            receipt = json.loads((root / 'existing-baseline-scores.json').read_text())
            self.assertEqual([Path(row['path']).stem for row in receipt['samples']],
                             ['target_sampled_0', 'target_sampled_2', 'target_sampled_10'])
            self.assertTrue(all(len(row['sha256']) == 64 for row in receipt['samples']))
            self.assertEqual(before, {path: path.read_bytes() for path in predictions.iterdir()})

    def test_unvalidated_or_partial_baseline_is_not_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / 'predictions'
            predictions.mkdir()
            (predictions / 'target_sampled_0.pdb').write_text('still writing\n')
            reward = Mock()
            for expected in (1, 10):
                score_existing_baseline(reward, predictions, root, expected_samples=expected)
                receipt = json.loads((root / 'existing-baseline-scores.json').read_text())
                self.assertEqual(receipt['status'], 'not_ready_at_launch')
                self.assertEqual(receipt['available_prediction_count'], 1)
                reward.score.assert_not_called()


if __name__ == "__main__":
    unittest.main()
