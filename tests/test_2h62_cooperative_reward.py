"""Cooperative ACVR2B reward invariants on synthetic structures, no inference."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.fk_2h62_cooperative_reward import (
    CooperativeComplexReward, DEFAULT_COOPERATIVE_SETTINGS, harmonic_mean,
)
from scripts.fk_2h62_contact_reward import ContactComplexReward
from scripts.fk_2h62_l61_reward import DEFAULT_L61_SETTINGS
from test_fk_2h62_reward import fixture, write_structure


FEATURE = "ACVR2B_interface_score"
BMPR_FEATURES = (
    "BMPR1A_hotspot_contact_recall",
    "BMPR1A_native_contact_recall",
    "BMPR1A_native_contact_precision",
)


def atom_map(coordinates):
    return {(chain, *key): xyz for chain, atoms in coordinates.items()
            for key, xyz in atoms.items()}


class CooperativeRewardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.predicted, self.sequences = fixture(self.root)
        self.settings = copy.deepcopy(DEFAULT_COOPERATIVE_SETTINGS)
        # Synthetic coordinates are useful for exact contact invariants, not
        # chemistry: isolate positive features from the retained penalties.
        for specification in self.settings["penalties"].values():
            specification["weight"] = 0.
        self.reward = CooperativeComplexReward(
            self.root / "observed.cif", self.root / "observed.pdb", self.settings)

    def score(self, coordinates=None):
        path = self.root / "prediction.pdb"
        write_structure(path, self.predicted if coordinates is None else coordinates,
                        self.sequences)
        return self.reward.score(path)

    def knuckle(self, coordinates=None, receptor="E", site=0):
        return self.reward._copy_features(
            atom_map(self.predicted if coordinates is None else coordinates),
            receptor, {"A": "A", "D": "B"}, site, "ACVR2B")

    def test_harmonic_mean_requires_both_anchors_and_handles_empty_contacts(self):
        self.assertEqual(harmonic_mean(1., 0.), 0.)
        self.assertEqual(harmonic_mean(0., 1.), 0.)
        self.assertEqual(harmonic_mean(0., 0.), 0.)
        self.assertEqual(harmonic_mean(1., 1.), 1.)
        self.assertAlmostEqual(harmonic_mean(1., .1), 2. / 11.)

    def test_native_contact_pattern_gets_full_credit_at_both_distinct_sites(self):
        result = self.score()
        self.assertAlmostEqual(result["reward"], 5.5, places=6)
        self.assertEqual(len(result["assignment_candidates"]), 8)
        self.assertEqual(result["features"][FEATURE], 1.)
        self.assertEqual(result["contributions"][FEATURE], 3.)
        self.assertEqual(self.reward.metadata["positive_maximum"], 5.5)
        for row in result["copy_features"]:
            if row["component"] != "ACVR2B":
                continue
            contacts = row["details"]["ACVR2B_contacts"]
            self.assertGreater(contacts["native_contacts"], 0)
            self.assertEqual(contacts["recovered_contacts"], contacts["native_contacts"])
            self.assertEqual(contacts["predicted_contacts"], contacts["native_contacts"])
            for key in ("ACVR2B_native_contact_recall", "ACVR2B_native_contact_precision",
                        "ACVR2B_joint_hotspot_score", "ACVR2B_contact_f1", FEATURE):
                self.assertAlmostEqual(row["features"][key], 1.)

    def test_missing_y42_eliminates_joint_credit_despite_perfect_w60(self):
        changed = copy.deepcopy(self.predicted)
        for key in [key for key in changed["E"] if key[0] == 41]:
            del changed["E"][key]
        row = self.knuckle(changed)
        self.assertAlmostEqual(row["features"]["W60_packing"], 1.)
        self.assertEqual(row["features"]["Y42_engagement"], 0.)
        self.assertEqual(row["features"]["ACVR2B_joint_hotspot_score"], 0.)
        # Good whole-interface contacts can retain half the maximum score;
        # the proposed objective gates hotspot credit, not the whole module.
        self.assertGreater(row["features"]["ACVR2B_contact_f1"], 0.)
        self.assertLessEqual(row["features"][FEATURE], .5)
        native = self.knuckle()["details"]["ACVR2B_contacts"]
        contacts = row["details"]["ACVR2B_contacts"]
        self.assertEqual(contacts["native_contacts"], native["native_contacts"])
        self.assertLess(contacts["recovered_contacts"], native["recovered_contacts"])

    def test_empty_predicted_interface_has_zero_f1_and_null_diagnostic_precision(self):
        changed = copy.deepcopy(self.predicted)
        changed["E"] = {key: xyz + [0., 0., 100.] for key, xyz in changed["E"].items()}
        row = self.knuckle(changed)
        contacts = row["details"]["ACVR2B_contacts"]
        self.assertGreater(contacts["native_contacts"], 0)
        self.assertEqual(contacts["predicted_contacts"], 0)
        self.assertEqual(contacts["recovered_contacts"], 0)
        self.assertIsNone(contacts["native_contact_precision"])
        for key in ("ACVR2B_native_contact_recall", "ACVR2B_native_contact_precision",
                    "ACVR2B_contact_f1", FEATURE):
            self.assertEqual(row["features"][key], 0.)

    def test_inferred_site_exchanges_both_ligand_identities_without_receptor_mirroring(self):
        observed = self.knuckle()
        inferred = self.knuckle(receptor="F", site=1)
        self.assertEqual(observed["details"]["ACVR2B_contacts"]["ligand_reference_to_prediction"],
                         {"A": "A", "D": "B"})
        self.assertEqual(inferred["details"]["ACVR2B_contacts"]["ligand_reference_to_prediction"],
                         {"A": "B", "D": "A"})
        # Direct atom maps retain sub-millangstrom fixture coordinates while
        # experimental atoms were rounded when writing PDB. Full score() uses
        # PDB on both sides and is checked for exact maximum separately.
        self.assertAlmostEqual(inferred["features"][FEATURE], 1., places=3)
        wrong_site = self.knuckle(receptor="E", site=1)
        self.assertEqual(wrong_site["features"]["ACVR2B_contact_f1"], 0.)
        self.assertLess(wrong_site["features"][FEATURE], .01)

    def test_combined_copy_score_is_minimized_after_combining_features(self):
        # Opposite failures across copies: J/F=(1,.2) and (.2,1).
        # Each A=.6, so the module earns 1.8 rather than separately taking
        # min(J)=min(F)=.2, which would incorrectly give only .6.
        copies = [self.reward._copy_features(atom_map(self.predicted), receptor,
                                            {"A": "A", "D": "B"}, site, component)
                  for component, receptors in (("BMPR1A", "CD"), ("ACVR2B", "EF"))
                  for site, receptor in enumerate(receptors)]
        for row, joint, contact in zip(copies[2:], (1., .2), (.2, 1.)):
            row["features"].update(ACVR2B_joint_hotspot_score=joint,
                                   ACVR2B_contact_f1=contact,
                                   ACVR2B_interface_score=.5 * (joint + contact))
        features = self.reward._assignment_features(copies)
        self.assertAlmostEqual(features[FEATURE], .6)
        self.assertAlmostEqual(3. * features[FEATURE], 1.8)

    def test_bmpr1a_features_are_unchanged_and_l61_is_diagnostic_only(self):
        parent = ContactComplexReward(self.root / "observed.cif", self.root / "observed.pdb")
        for receptor, site in (("C", 0), ("D", 1)):
            arguments = atom_map(self.predicted), receptor, {"A": "A", "D": "B"}, site, "BMPR1A"
            self.assertEqual(self.reward._copy_features(*arguments), parent._copy_features(*arguments))
        for key in BMPR_FEATURES:
            self.assertEqual(DEFAULT_COOPERATIVE_SETTINGS["positive_weights"][key],
                             DEFAULT_L61_SETTINGS["positive_weights"][key])
        self.assertEqual(DEFAULT_COOPERATIVE_SETTINGS["penalties"]["ACVR2B_L61_geometry_error"]["weight"], 0.)
        for key, specification in DEFAULT_L61_SETTINGS["penalties"].items():
            if key != "ACVR2B_L61_geometry_error":
                self.assertEqual(DEFAULT_COOPERATIVE_SETTINGS["penalties"][key], specification)
        native = self.knuckle()["features"]["ACVR2B_L61_geometry_error"]
        changed = copy.deepcopy(self.predicted)
        changed["E"][60, "N"] += [0., 0., 30.]
        moved = self.knuckle(changed)["features"]["ACVR2B_L61_geometry_error"]
        self.assertGreater(moved, native)
        self.assertEqual(self.reward._penalty_contributions({
            key: moved if key == "ACVR2B_L61_geometry_error" else 0.
            for key in self.settings["penalties"]})["ACVR2B_L61_geometry_error"], 0.)

    def test_rigid_motion_and_type_preserving_copy_swaps_preserve_score(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        transformed = {chain: {key: xyz @ rotation + [5., -7., 11.]
                               for key, xyz in atoms.items()}
                       for chain, atoms in self.predicted.items()}
        for left, right in (("A", "B"), ("C", "D"), ("E", "F")):
            transformed[left], transformed[right] = transformed[right], transformed[left]
        result = self.score(transformed)
        self.assertAlmostEqual(result["reward"], 5.5, places=6)
        self.assertAlmostEqual(result["features"][FEATURE], 1.)
        for key in BMPR_FEATURES:
            self.assertEqual(result["features"][key], 1.)


if __name__ == "__main__":
    unittest.main()
