"""Typed interface-distance steering checks on synthetic C2 coordinates."""
import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from scripts.fk_2h62_reward import DEFAULT_SETTINGS, PartialComplexReward
from scripts.fk_2h62_topology_reward import TopologyComplexReward, huber_error
from tests.test_fk_2h62_reward import fixture, write_structure


class TopologyRewardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.predicted, self.sequences = fixture(self.root)
        settings = copy.deepcopy(DEFAULT_SETTINGS)
        settings["positive_weights"] = {key: 0. for key in settings["positive_weights"]}
        settings["penalties"] = {
            key: dict(spec, weight=0.) for key, spec in settings["penalties"].items()
        }
        settings["penalties"]["typed_interface_error"] = dict(weight=1., scale=1., cap=None)
        settings["topology"] = dict(
            distance_tolerance_angstrom=2., distance_scale_angstrom=5.,
            receptor_anchor_count=6, ligand_anchor_count_per_chain=4,
            worst_copy_weight=.5,
        )
        self.settings = settings
        self.reward = TopologyComplexReward(
            self.root / "observed.cif", self.root / "observed.pdb", settings,
        )
        # All chemistry penalty weights are zero in these topology tests. Their
        # separate fixed-geometry tests exercise that calculation; skipping it
        # here avoids quadratic clash work on densely packed synthetic atoms.
        geometry = patch("scripts.fk_2h62_reward.fixed_geometry", return_value={
            "clashes_per_1000_heavy_atoms": 0., "covalent_bond_strain": 0.,
        })
        geometry.start()
        self.addCleanup(geometry.stop)

    def score(self, coordinates=None, reward=None):
        path = self.root / "prediction.pdb"
        write_structure(path, self.predicted if coordinates is None else coordinates, self.sequences)
        return (self.reward if reward is None else reward).score(path)

    def translated(self, chain, vector):
        result = copy.deepcopy(self.predicted)
        result[chain] = {key: xyz + vector for key, xyz in result[chain].items()}
        return result

    def test_exact_ligand_swapped_native_pattern_has_zero_penalty(self):
        result = self.score()
        self.assertAlmostEqual(result["features"]["typed_interface_error"], 0., places=10)
        self.assertAlmostEqual(result["reward"], 0., places=10)
        self.assertEqual(len(result["copy_features"]), 4)
        self.assertEqual(len(result["assignment_candidates"]), 8)

    def test_missing_copy_displacement_is_informative_far_from_hotspots(self):
        native = self.score()
        ten = self.score(self.translated("D", np.array([0., 0., 10.])))
        twenty = self.score(self.translated("D", np.array([0., 0., 20.])))
        self.assertGreater(native["reward"], ten["reward"])
        self.assertGreater(ten["reward"], twenty["reward"])
        self.assertGreater(twenty["features"]["typed_interface_error"],
                           ten["features"]["typed_interface_error"])

    def test_two_copies_cannot_receive_full_credit_at_the_same_site(self):
        duplicate = copy.deepcopy(self.predicted)
        duplicate["D"] = copy.deepcopy(duplicate["C"])
        duplicate["F"] = copy.deepcopy(duplicate["E"])
        result = self.score(duplicate)
        self.assertLess(result["reward"], self.score()["reward"])
        self.assertGreater(result["features"]["typed_interface_error"], .1)
        for receptor in ("BMPR1A", "ACVR2B"):
            self.assertEqual(len(set(result["assignment"][f"{receptor}_site_to_prediction"])), 2)

    def test_swapping_receptor_types_on_the_missing_side_is_penalized(self):
        swapped = copy.deepcopy(self.predicted)
        centers = {
            chain: np.mean([xyz for (position, name), xyz in atoms.items() if name == "CA"], axis=0)
            for chain, atoms in self.predicted.items()
        }
        # Move each intact receptor fold to the other type's center. Sequences,
        # residue identities, and the measured side remain unchanged.
        for chain, other in (("D", "F"), ("F", "D")):
            swapped[chain] = {
                key: xyz + centers[other] - centers[chain]
                for key, xyz in swapped[chain].items()
            }
        result = self.score(swapped)
        self.assertLess(result["reward"], self.score()["reward"])
        self.assertGreater(result["features"]["typed_interface_error"], .1)

    def test_each_of_the_four_receptor_copies_contributes(self):
        native = self.score()["reward"]
        for chain in "CDEF":
            with self.subTest(chain=chain):
                shifted = self.score(self.translated(chain, np.array([0., 0., 20.])))
                self.assertLess(shifted["reward"], native)

    def test_global_rigid_transform_and_identical_copy_swaps_are_invariant(self):
        perturbed = self.translated("D", np.array([0., 0., 15.]))
        reference = self.score(perturbed)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        for group in ((("A", "B"),), (("C", "D"),), (("E", "F"),),
                      (("A", "B"), ("C", "D"), ("E", "F"))):
            with self.subTest(swaps=group):
                transformed = {
                    chain: {key: xyz @ rotation + [5., -7., 11.] for key, xyz in atoms.items()}
                    for chain, atoms in perturbed.items()
                }
                for first, second in group:
                    transformed[first], transformed[second] = transformed[second], transformed[first]
                result = self.score(transformed)
                self.assertAlmostEqual(result["reward"], reference["reward"], places=6)
                self.assertAlmostEqual(result["features"]["typed_interface_error"],
                                       reference["features"]["typed_interface_error"], places=6)

    def test_sub_tolerance_asymmetry_is_not_forced_to_exact_c2(self):
        # Triangle inequality bounds each distance change by this translation's
        # 1.5 Å norm, inside the declared 2 Å deadband.
        result = self.score(self.translated("D", np.array([0., 0., 1.5])))
        self.assertAlmostEqual(result["features"]["typed_interface_error"], 0., places=10)

    def test_huber_and_worst_copy_aggregation_match_declared_math(self):
        result = self.score(self.translated("D", np.array([0., 0., 15.])))
        values = []
        for row in result["copy_features"]:
            features = row["features"]
            error = features["typed_interface_copy_error_angstrom"]
            normalized = error / 5.
            huber = .5 * normalized ** 2 if normalized <= 1. else normalized - .5
            self.assertAlmostEqual(features["typed_interface_copy_error_normalized"], normalized)
            self.assertAlmostEqual(features["typed_interface_copy_huber_error"], huber)
            values.append(huber)
        expected = .5 * np.mean(values) + .5 * max(values)
        self.assertAlmostEqual(result["features"]["typed_interface_error"], expected)
        self.assertAlmostEqual(result["reward"], -expected)

    def test_huber_boundary_and_uncapped_far_errors(self):
        self.assertEqual(huber_error(0.), 0.)
        self.assertEqual(huber_error(.5), .125)
        self.assertEqual(huber_error(1.), .5)
        self.assertAlmostEqual(huber_error(1. - 1e-6), .5 - 1e-6, places=11)
        self.assertAlmostEqual(huber_error(1. + 1e-6), .5 + 1e-6, places=11)
        self.assertEqual(huber_error(10.), 9.5)
        self.assertGreater(huber_error(100.), huber_error(10.))

    def test_existing_hotspot_features_are_preserved(self):
        settings = copy.deepcopy(self.settings)
        settings["positive_weights"] = copy.deepcopy(DEFAULT_SETTINGS["positive_weights"])
        old_settings = copy.deepcopy(settings)
        del old_settings["penalties"]["typed_interface_error"]
        old = PartialComplexReward(self.root / "observed.cif", self.root / "observed.pdb", old_settings)
        new = TopologyComplexReward(self.root / "observed.cif", self.root / "observed.pdb", settings)
        before, after = self.score(reward=old), self.score(reward=new)
        for feature in DEFAULT_SETTINGS["positive_weights"]:
            self.assertAlmostEqual(after["features"][feature], before["features"][feature], places=6)
        self.assertAlmostEqual(after["reward"], 5., places=6)

    def test_missing_prediction_ca_cannot_hide_interface_error(self):
        missing = copy.deepcopy(self.predicted)
        del missing["D"][85, "CA"]
        with self.assertRaisesRegex(ValueError, "missing CA"):
            self.score(missing)

        # Also exercise the local distance scorer directly: missing anchors must
        # cost rather than disappear from the distance-error denominator.
        full_atoms = {
            (chain, position, name): xyz
            for chain, data in self.predicted.items()
            for (position, name), xyz in data.items()
        }
        for chain in ("D", "A", "B"):
            with self.subTest(missing_ca_chain=chain):
                atoms = {key: xyz for key, xyz in full_atoms.items()
                         if not (key[0] == chain and key[2] == "CA")}
                row = self.reward._copy_features(atoms, "D", {"A": "A", "D": "B"}, 1, "BMPR1A")
                self.assertGreater(row["features"]["typed_interface_copy_huber_error"], 0.)
                details = row["details"]["typed_interface"]
                self.assertLess(details["anchor_pair_coverage"], 1.)
                self.assertEqual(details["pair_count"], 6 * 8)
                self.assertEqual(len(details["receptor_anchor_keys"]), 6)
                self.assertEqual(len(details["ligand_anchor_keys"]), 8)


if __name__ == "__main__":
    unittest.main()
