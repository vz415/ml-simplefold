"""Tests of the declared per-copy FK objective, without model inference."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fk_complex_reward import combine_reward


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(component=c, hotspot_contact_recall=1.,
                          native_contact_recall=1., native_contact_precision=1.,
                          ligand_frame_CA_RMSD_angstrom=0.)
                     for c in ('BMPR1A', 'BMPR1A', 'ACVR2A', 'ACVR2A')]
        self.interfaces = {'receptors': self.rows,
                           'metrics': {'BMP2_dimer_CA_RMSD_angstrom': 0.}}
        self.clashes = {'clashes_per_1000_heavy_atoms': 0.}
        self.settings = {'positive_weights': {'ACVR2A_hotspot_contact_recall': 2.},
                         'penalties': {'clashes_per_1000_heavy_atoms':
                                       {'weight': .5, 'scale': 50., 'cap': 4.}}}

    def score(self):
        return combine_reward(self.interfaces, self.clashes, self.settings)

    def test_one_failed_copy_cannot_hide_behind_good_copy(self):
        self.assertEqual(self.score()['reward'], 2.)
        self.rows[2]['hotspot_contact_recall'] = 0.
        self.assertEqual(self.score()['reward'], 0.)

    def test_clashes_reduce_reward_and_penalty_is_bounded(self):
        self.clashes['clashes_per_1000_heavy_atoms'] = 50.
        self.assertEqual(self.score()['reward'], 1.5)
        self.clashes['clashes_per_1000_heavy_atoms'] = 10000.
        self.assertEqual(self.score()['reward'], 0.)

    def test_exchanging_identical_copy_labels_preserves_reward(self):
        self.rows[2]['hotspot_contact_recall'] = .3
        before = self.score()
        self.interfaces['receptors'] = list(reversed(copy.deepcopy(self.rows)))
        self.assertEqual(before, self.score())

    def test_missing_contact_precision_is_not_treated_as_success(self):
        self.rows[3]['native_contact_precision'] = None
        with self.assertRaisesRegex(ValueError, 'Undefined'):
            self.score()

    def test_explicitly_disconnected_receptor_is_ranked_with_zero_precision(self):
        self.rows[3].update(native_contact_precision=None, predicted_contacts=0)
        self.assertEqual(self.score()['features']['ACVR2A_native_contact_precision'], 0.)


if __name__ == '__main__':
    unittest.main()
