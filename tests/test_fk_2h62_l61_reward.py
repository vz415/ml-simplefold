"""Local L61 steering invariants on synthetic structures, without inference."""
import argparse
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from hydra import compose, initialize_config_dir

from scripts.fk_2h62_contact_reward import ContactComplexReward, DEFAULT_CONTACT_SETTINGS
from scripts.fk_2h62_l61_reward import DEFAULT_L61_SETTINGS, L61ComplexReward
from scripts.sample_complex_fk import make_reward
from test_fk_2h62_reward import fixture, write_structure


FEATURE = "ACVR2B_L61_geometry_error"


def atom_map(coordinates):
    return {(chain, *key): xyz for chain, atoms in coordinates.items()
            for key, xyz in atoms.items()}


class L61RewardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.predicted, self.sequences = fixture(self.root)
        self.settings = copy.deepcopy(DEFAULT_L61_SETTINGS)
        # Arbitrary synthetic coordinates establish reward invariants, not
        # chemistry. Isolate the new penalty from the old strain/clash terms.
        for key, specification in self.settings["penalties"].items():
            if key != FEATURE:
                specification["weight"] = 0.
        self.reward = L61ComplexReward(self.root / "observed.cif",
                                       self.root / "observed.pdb", self.settings)

    def score(self, coordinates=None, reward=None):
        path = self.root / "prediction.pdb"
        write_structure(path, self.predicted if coordinates is None else coordinates,
                        self.sequences)
        return (reward or self.reward).score(path)

    def knuckle(self, coordinates=None, receptor="E", site=0):
        return self.reward._copy_features(
            atom_map(self.predicted if coordinates is None else coordinates),
            receptor, {"A": "A", "D": "B"}, site, "ACVR2B")

    def test_native_geometry_is_zero_at_both_sites_and_maximum_stays_5_5(self):
        result = self.score()
        self.assertAlmostEqual(result["reward"], 5.5, places=6)
        self.assertAlmostEqual(result["features"][FEATURE], 0., places=12)
        self.assertEqual(len(result["assignment_candidates"]), 8)
        self.assertEqual(self.reward.metadata["positive_maximum"], 5.5)
        for row in result["copy_features"]:
            if row["component"] != "ACVR2B":
                continue
            geometry = row["details"]["L61_geometry"]
            self.assertEqual(geometry["status"], "ok")
            self.assertEqual(geometry["heavy_atom_coverage"], 1.)
            self.assertAlmostEqual(geometry["geometry_error"], 0., places=12)
        residues = self.reward.metadata["L61_geometry"]["reference_residues"]
        self.assertEqual([(row["reference_author_residue"], row["reference_residue_identity"])
                          for row in residues], [("60", "W"), ("61", "L"), ("88", "S")])

    def test_rigid_motion_and_type_preserving_copy_swaps_leave_reward_invariant(self):
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
                self.assertAlmostEqual(result["features"][FEATURE], 0., places=12)

    def test_l61_displacement_is_penalized_without_changing_hotspot_positives(self):
        modified = copy.deepcopy(self.predicted)
        modified["E"][60, "N"] += [0., 0., 40.]
        native = self.score()
        result = self.score(modified)
        self.assertGreater(result["features"][FEATURE], 1.)
        self.assertLess(result["reward"], native["reward"])
        for key in DEFAULT_CONTACT_SETTINGS["positive_weights"]:
            self.assertEqual(result["features"][key], native["features"][key])
        self.assertAlmostEqual(result["contributions"][FEATURE],
                               -.5 * result["features"][FEATURE])

    def test_approach_angle_changes_error_even_when_anchor_distance_is_fixed(self):
        modified = copy.deepcopy(self.predicted)
        origin = modified["E"][60, "N"]
        direction = modified["A"][87, "OG"] - origin
        target = self.reward.l61_angle_targets[0]
        # Pick 0 or 180 degrees, whichever is farther from the native angle.
        modified["E"][59, "C"] = origin + direction * (1. if target > 90. else -1.)
        native = self.knuckle()["details"]["L61_geometry"]
        changed = self.knuckle(modified)["details"]["L61_geometry"]
        self.assertEqual(changed["distance"], native["distance"])
        self.assertGreater(changed["approach_angles"][0]["error"], 0.)
        self.assertGreater(changed["geometry_error"], 0.)

    def test_worst_copy_error_is_used_without_averaging_away_a_failed_site(self):
        modified = copy.deepcopy(self.predicted)
        modified["F"][60, "N"] += [0., 0., 25.]
        result = self.score(modified)
        errors = [row["features"][FEATURE] for row in result["copy_features"]
                  if row["component"] == "ACVR2B"]
        self.assertAlmostEqual(min(errors), 0., places=12)
        self.assertGreater(max(errors), 1.)
        self.assertEqual(result["features"][FEATURE], max(errors))

    def test_missing_or_degenerate_geometry_has_finite_penalty_and_coverage(self):
        for mode in ("missing", "degenerate"):
            with self.subTest(mode=mode):
                modified = copy.deepcopy(self.predicted)
                if mode == "missing":
                    del modified["E"][60, "N"]
                else:
                    modified["E"][59, "C"] = modified["E"][60, "N"].copy()
                result = self.score(modified)
                self.assertTrue(np.isfinite(result["reward"]))
                self.assertGreaterEqual(result["features"][FEATURE], 10.)
                row = next(row for row in result["copy_features"]
                           if row["prediction_chain"] == "E")
                geometry = row["details"]["L61_geometry"]
                self.assertEqual(geometry["status"], "missing_or_degenerate_geometry")
                self.assertEqual(geometry["heavy_atom_coverage"], .8 if mode == "missing" else 1.)

    def test_zero_new_weight_reproduces_contact_reward_and_all_old_contributions(self):
        settings = copy.deepcopy(DEFAULT_L61_SETTINGS)
        settings["penalties"][FEATURE]["weight"] = 0.
        reward = L61ComplexReward(self.root / "observed.cif", self.root / "observed.pdb", settings)
        parent = ContactComplexReward(self.root / "observed.cif", self.root / "observed.pdb")
        modified = copy.deepcopy(self.predicted)
        modified["E"][60, "N"] += [0., 0., 30.]
        result, original = self.score(modified, reward), self.score(modified, parent)
        self.assertEqual(result["reward"], original["reward"])
        self.assertEqual(result["assignment"], original["assignment"])
        for key, value in original["features"].items():
            self.assertEqual(result["features"][key], value)
        for key, value in original["contributions"].items():
            self.assertEqual(result["contributions"][key], value)
        self.assertEqual(DEFAULT_L61_SETTINGS["positive_weights"],
                         DEFAULT_CONTACT_SETTINGS["positive_weights"])
        for key, specification in DEFAULT_CONTACT_SETTINGS["penalties"].items():
            self.assertEqual(DEFAULT_L61_SETTINGS["penalties"][key], specification)


class L61ConfigTests(unittest.TestCase):
    def test_hydra_profile_changes_only_reward_and_keeps_sampling_schedule(self):
        directory = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(version_base="1.3", config_dir=directory):
            configured = compose(config_name="sample", overrides=["experiment=2h62-inpaint-fk-l61"])
            previous = compose(config_name="sample", overrides=["experiment=2h62-inpaint-fk-contacts"])
        self.assertEqual(configured.sampling.reward_kind, "2h62_partial_contacts_l61")
        self.assertEqual(dict(configured.sampling.reward.positive_weights),
                         DEFAULT_CONTACT_SETTINGS["positive_weights"])
        for key, specification in previous.sampling.reward.penalties.items():
            self.assertEqual(configured.sampling.reward.penalties[key], specification)
        self.assertIsNone(configured.sampling.reward.penalties[FEATURE].cap)
        self.assertEqual(configured.sampling.reward.penalties[FEATURE].weight, .5)
        for key in ("model", "samples", "num_steps", "tau", "seed"):
            self.assertEqual(configured[key], previous[key])
        for key in ("checkpoint_times", "beta", "ess_threshold", "brownian_seed",
                    "resampling_seed", "common_start", "model_batch_size", "run_baseline"):
            self.assertEqual(configured.sampling[key], previous.sampling[key])

    def test_dispatch_routes_to_l61_without_constructing_another_reward(self):
        l61, parent = Mock(), Mock()
        args = argparse.Namespace(reference_cif=Path("observed.cif"), reference_pdb=Path("observed.pdb"))
        settings = {"l61_geometry": {"distance_scale_angstrom": 3.}}
        with patch.dict(sys.modules, {
            "fk_2h62_l61_reward": SimpleNamespace(L61ComplexReward=l61),
            "fk_2h62_contact_reward": SimpleNamespace(ContactComplexReward=parent),
        }):
            result = make_reward(args, {"reward_kind": "2h62_partial_contacts_l61", "reward": settings})
        self.assertIs(result, l61.return_value)
        l61.assert_called_once_with(args.reference_cif, args.reference_pdb, settings)
        parent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
