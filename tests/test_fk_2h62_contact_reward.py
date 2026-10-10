"""Contact-transfer reward invariants on synthetic coordinates, without inference."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from scripts.fk_2h62_contact_reward import ContactComplexReward, DEFAULT_CONTACT_SETTINGS
from scripts.fk_2h62_reward import PartialComplexReward
from test_fk_2h62_reward import fixture, write_structure


CONTACT_FEATURES = (
    "BMPR1A_hotspot_contact_recall",
    "BMPR1A_native_contact_recall",
    "BMPR1A_native_contact_precision",
)


def atom_map(coordinates):
    return {(chain, *key): xyz for chain, atoms in coordinates.items()
            for key, xyz in atoms.items()}


class ContactRewardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.predicted, self.sequences = fixture(self.root)
        self.settings = copy.deepcopy(DEFAULT_CONTACT_SETTINGS)
        for penalty in self.settings["penalties"].values():
            penalty["weight"] = 0.
        self.reward = self.make_reward()

    def make_reward(self):
        return ContactComplexReward(self.root / "observed.cif",
                                    self.root / "observed.pdb", self.settings)

    def score(self, coordinates=None):
        path = self.root / "prediction.pdb"
        write_structure(path, self.predicted if coordinates is None else coordinates,
                        self.sequences)
        return self.reward.score(path)

    def wrist(self, coordinates=None, receptor="C", site=0, reward=None):
        coordinates = self.predicted if coordinates is None else coordinates
        return (reward or self.reward)._copy_features(
            atom_map(coordinates), receptor, {"A": "A", "D": "B"}, site, "BMPR1A")

    def test_native_contacts_credit_both_sites_and_undefined_hotspot_is_excluded(self):
        result = self.score()
        self.assertAlmostEqual(result["reward"], 5.5, places=6)
        self.assertEqual(len(result["assignment_candidates"]), 8)
        for feature in CONTACT_FEATURES:
            self.assertEqual(result["features"][feature], 1.)
        wrists = [row for row in result["copy_features"] if row["component"] == "BMPR1A"]
        self.assertEqual({row["site"] for row in wrists}, {0, 1})
        for row in wrists:
            contacts = row["details"]["BMPR1A_contacts"]
            self.assertEqual(contacts["recovered_contacts"], contacts["native_contacts"])
            hotspots = {hotspot["name"]: hotspot for hotspot in contacts["hotspots"]}
            self.assertEqual(hotspots["F85"]["reference_full_sequence_position"], 85)
            self.assertEqual(hotspots["Q86"]["reference_full_sequence_position"], 86)
            # This fixture deliberately has no native F85 contacts. Undefined
            # recall must not halve the perfectly recovered Q86 hotspot score.
            self.assertEqual(hotspots["F85"]["native_contacts"], 0)
            self.assertIsNone(hotspots["F85"]["contact_recall"])
            self.assertEqual(hotspots["Q86"]["contact_recall"], 1.)
        self.assertEqual(self.reward.metadata["positive_maximum"], 5.5)
        self.assertEqual(result["contributions"]["Q86_geometry"], 0.)
        self.assertFalse(any("typed_interface" in key or "topology" in key
                             for key in result["features"]))

    def test_lost_or_duplicated_wrist_cannot_get_two_site_credit(self):
        for mode in ("displaced", "duplicated"):
            with self.subTest(mode=mode):
                modified = copy.deepcopy(self.predicted)
                if mode == "displaced":
                    modified["D"] = {key: xyz + [0., 0., 100.]
                                     for key, xyz in modified["D"].items()}
                else:
                    modified["D"] = copy.deepcopy(modified["C"])
                result = self.score(modified)
                for feature in CONTACT_FEATURES:
                    self.assertEqual(result["features"][feature], 0.)
                self.assertAlmostEqual(result["features"]["W60_packing"], 1.)
                self.assertAlmostEqual(result["features"]["Y42_engagement"], 1.)
                self.assertAlmostEqual(result["reward"], 3.)
                self.assertEqual(len(set(result["assignment"]["BMPR1A_site_to_prediction"])), 2)
                empty = [row["details"]["BMPR1A_contacts"] for row in result["copy_features"]
                         if row["component"] == "BMPR1A"
                         and row["features"]["BMPR1A_native_contact_recall"] == 0.]
                self.assertTrue(empty)
                for row in empty:
                    # A duplicate can retain contacts to the wrong ligand
                    # copy: that precision is defined and zero. A displaced
                    # receptor has no predicted contacts, hence null.
                    if row["predicted_contacts"]:
                        self.assertEqual(row["native_contact_precision"], 0.)
                    else:
                        self.assertIsNone(row["native_contact_precision"])

    def test_missing_hotspot_atoms_keep_full_native_contact_denominator(self):
        native = self.wrist()["details"]["BMPR1A_contacts"]
        modified = copy.deepcopy(self.predicted)
        for key in [key for key in modified["C"] if key[0] == 85]:
            del modified["C"][key]
        result = self.wrist(modified)
        contacts = result["details"]["BMPR1A_contacts"]
        self.assertEqual(contacts["native_contacts"], native["native_contacts"])
        self.assertLess(contacts["recovered_contacts"], native["recovered_contacts"])
        self.assertLess(result["features"]["BMPR1A_native_contact_recall"], 1.)
        self.assertEqual(result["features"]["BMPR1A_hotspot_contact_recall"], 0.)
        q86 = next(row for row in contacts["hotspots"] if row["name"] == "Q86")
        self.assertEqual(q86["native_contacts"], 45)
        self.assertEqual(q86["matched_heavy_atoms"], 0)
        self.assertEqual(q86["contact_recall"], 0.)
        self.assertEqual(q86["status"], "missing_prediction_residue_atoms")

    def test_unobserved_residues_and_atom_names_do_not_create_contact_credit(self):
        source = {"A": copy.deepcopy(self.predicted["A"]),
                  "D": copy.deepcopy(self.predicted["B"]),
                  "B": copy.deepcopy(self.predicted["C"]),
                  "C": copy.deepcopy(self.predicted["E"])}
        # The full construct keeps its sequence, while the experimental N-end
        # is unresolved. Prediction-only atoms there must not enter precision.
        for key in [key for key in source["B"] if key[0] == 0]:
            del source["B"][key]
        source_sequences = dict(A=self.sequences["A"], D=self.sequences["B"],
                                B=self.sequences["C"], C=self.sequences["E"])
        write_structure(self.root / "observed.cif", source, source_sequences, cif=True)
        write_structure(self.root / "observed.pdb", source, source_sequences)
        reward = self.make_reward()
        without_extra = copy.deepcopy(self.predicted)
        for key in [key for key in without_extra["C"] if key[0] == 0]:
            del without_extra["C"][key]
        with_extra = copy.deepcopy(without_extra)
        with_extra["C"][0, "CA"] = with_extra["B"][0, "CA"].copy()
        with_extra["C"][85, "XX"] = with_extra["B"][0, "CA"].copy()
        first = self.wrist(without_extra, reward=reward)
        second = self.wrist(with_extra, reward=reward)
        self.assertEqual(first["features"], second["features"])
        self.assertEqual(first["details"]["BMPR1A_contacts"],
                         second["details"]["BMPR1A_contacts"])

    def test_rigid_motion_and_type_preserving_permutations_leave_reward_invariant(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        for swaps in ((), (("A", "B"),), (("C", "D"),), (("E", "F"),),
                      (("A", "B"), ("C", "D"), ("E", "F"))):
            with self.subTest(swaps=swaps):
                transformed = {chain: {key: xyz @ rotation + [5., -7., 11.]
                                       for key, xyz in atoms.items()}
                               for chain, atoms in self.predicted.items()}
                for left, right in swaps:
                    transformed[left], transformed[right] = transformed[right], transformed[left]
                result = self.score(transformed)
                self.assertAlmostEqual(result["reward"], 5.5, places=6)
                for feature in CONTACT_FEATURES:
                    self.assertEqual(result["features"][feature], 1.)

    def test_acvr2b_copy_features_equal_original_reward(self):
        original = PartialComplexReward(self.root / "observed.cif", self.root / "observed.pdb")
        for ligand in ({"A": "A", "D": "B"}, {"A": "B", "D": "A"}):
            for receptor in ("E", "F"):
                for site in (0, 1):
                    arguments = atom_map(self.predicted), receptor, ligand, site, "ACVR2B"
                    self.assertEqual(original._copy_features(*arguments),
                                     self.reward._copy_features(*arguments))


class ContactConfigTests(unittest.TestCase):
    def test_profile_keeps_original_sampling_and_penalties(self):
        directory = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(version_base="1.3", config_dir=directory):
            configured = compose(config_name="sample",
                                 overrides=["experiment=2h62-inpaint-fk-contacts"])
            original = compose(config_name="sample", overrides=["experiment=2h62-inpaint-fk"])
        self.assertEqual(configured.sampling.reward_kind, "2h62_partial_contacts")
        self.assertEqual(dict(configured.sampling.reward.positive_weights),
                         DEFAULT_CONTACT_SETTINGS["positive_weights"])
        self.assertEqual(configured.sampling.reward.penalties, original.sampling.reward.penalties)
        self.assertEqual(configured.tau, .01)
        self.assertEqual(configured.model, "simplefold_3B")
        self.assertEqual(configured.samples, 10)
        self.assertEqual(configured.num_steps, 500)
        self.assertEqual(configured.seed, 42)
        self.assertEqual(configured.sampling.brownian_seed, 43)
        self.assertEqual(configured.sampling.resampling_seed, 44)
        self.assertFalse(configured.sampling.run_baseline)
        self.assertIsNone(configured.sampling.model_batch_size)
        self.assertEqual(configured.sampling.checkpoint_times, original.sampling.checkpoint_times)
        self.assertEqual(configured.sampling.beta, original.sampling.beta)
        self.assertEqual(configured.sampling.ess_threshold, original.sampling.ess_threshold)


if __name__ == "__main__":
    unittest.main()
