"""Source/record identity must survive native loader fallback and batching."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/simplefold'))
import torch
from mixed_training import PlannedMixture, MixtureDataModule


class Records:
    def __init__(self, *names):
        self.samples = [(0, name) for name in names]

    def __getitem__(self, index):
        return dict(record={'id': self.samples[index][1]}, coords=torch.ones(3, 3))


class MixtureTests(unittest.TestCase):
    def test_replays_sources_records_and_repetitions_in_plan_order(self):
        datasets = dict(pdb=Records('pdb_a', 'pdb_b'), swissprot=Records('af_a'),
                        synthetic=Records('complex_a'))
        draws = [dict(source='pdb', record_id='pdb_b'),
                 dict(source='synthetic', record_id='complex_a'),
                 dict(source='swissprot', record_id='af_a'),
                 dict(source='pdb', record_id='pdb_b')]
        batches = list(MixtureDataModule(datasets, draws).train_dataloader())
        self.assertEqual([batch['record'][0]['id'] for batch in batches],
                         ['pdb_b', 'complex_a', 'af_a', 'pdb_b'])
        self.assertEqual([batch['training_source'].item() for batch in batches], [0, 2, 1, 0])
        self.assertEqual([batch['training_plan_index'].item() for batch in batches], [0, 1, 2, 3])

    def test_refuses_native_loader_silent_record_replacement(self):
        class Replacing(Records):
            def __getitem__(self, index):
                return dict(record={'id': 'wrong_record'})
        mixture = PlannedMixture({'pdb': Replacing('intended_record')},
                                 [dict(source='pdb', record_id='intended_record')])
        with self.assertRaises(AssertionError):
            mixture[0]


if __name__ == '__main__':
    unittest.main()
