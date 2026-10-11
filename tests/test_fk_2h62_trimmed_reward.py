"""Terminal-trim controls retain native numbering and internal missing residues."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from scripts.fk_2h62_l61_reward import DEFAULT_L61_SETTINGS, L61ComplexReward
from scripts.fk_2h62_reward import atom_records, chemical_bonds
from test_fk_2h62_reward import fixture, write_structure


class TrimmedRewardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.predicted, self.sequences = fixture(self.root)
        self.start, self.end = 33, 117
        reference = {"A": self.predicted["A"], "B": self.predicted["C"],
                     "C": self.predicted["E"], "D": self.predicted["B"]}
        # Add an internal experimental gap, which must remain modeled.
        reference["B"] = {key: xyz for key, xyz in reference["B"].items()
                          if self.start <= key[0] < self.end and key[0] != 60}
        sequences = {"A": self.sequences["A"], "B": self.sequences["C"],
                     "C": self.sequences["E"], "D": self.sequences["B"]}
        write_structure(self.root / "observed.cif", reference, sequences, cif=True)
        write_structure(self.root / "observed.pdb", reference, sequences)
        settings = copy.deepcopy(DEFAULT_L61_SETTINGS)
        for specification in settings["penalties"].values():
            specification["weight"] = 0.
        self.original = L61ComplexReward(self.root / "observed.cif",
                                         self.root / "observed.pdb", settings)
        settings["prediction_sequence_spans"] = {"BMPR1A": [self.start, self.end]}
        self.trimmed = L61ComplexReward(self.root / "observed.cif",
                                        self.root / "observed.pdb", settings)
        write_structure(self.root / "original.pdb", self.predicted, self.sequences)
        trimmed = dict(self.predicted)
        sequences = dict(self.sequences)
        for chain in "CD":
            trimmed[chain] = {(position - self.start, name): xyz
                              for (position, name), xyz in self.predicted[chain].items()
                              if self.start <= position < self.end}
            sequences[chain] = sequences[chain][self.start:self.end]
        write_structure(self.root / "trimmed.pdb", trimmed, sequences)

    def test_terminal_trimming_preserves_contacts_hotspots_and_placement(self):
        original = self.original.score(self.root / "original.pdb")
        trimmed = self.trimmed.score(self.root / "trimmed.pdb")
        self.assertAlmostEqual(original["reward"], trimmed["reward"], places=10)
        for key in original["features"]:
            if key not in ("covalent_bond_strain", "clashes_per_1000_heavy_atoms"):
                self.assertAlmostEqual(original["features"][key], trimmed["features"][key],
                                       places=10, msg=key)
        self.assertEqual(original["interfaces"], trimmed["interfaces"])
        self.assertEqual(trimmed["coverage"]["experimental_heavy_atom_fraction"], 1.)
        self.assertLess(trimmed["geometry"]["heavy_atom_count"],
                        original["geometry"]["heavy_atom_count"])
        self.assertEqual([row["reference_full_sequence_position"]
                          for row in self.trimmed.wrist_hotspots], [85, 86])

    def test_trimmed_atom_keys_preserve_original_numbering_and_internal_gap(self):
        chains = self.trimmed._read_prediction(self.root / "trimmed.pdb")
        records, atoms, _ = atom_records(self.root / "trimmed.pdb", chains)
        for chain in "CD":
            self.assertEqual(chains[chain].full_positions, list(range(self.start, self.end)))
            for position in (60, 84, 85):
                np.testing.assert_array_equal(atoms[chain, position, "CA"],
                                               np.round(self.predicted[chain][position, "CA"], 3))
            self.assertNotIn((chain, self.start - 1, "CA"), atoms)
            self.assertNotIn((chain, self.end, "CA"), atoms)
        # Peptide adjacency is preserved through a gap in observed reference,
        # because that residue remains present in the modeled sequence.
        bonds = chemical_bonds(records, self.trimmed.fixed_disulfides)
        pairs = {(records[i].chain, records[i].number, records[j].number)
                 for (i, j), kind in bonds.items() if kind[0] == "peptide"}
        self.assertIn(("C", 60, 61), pairs)
        self.assertIn(("C", 61, 62), pairs)

    def test_wrong_sequence_still_fails_instead_of_silently_remapping(self):
        with self.assertRaisesRegex(ValueError, "sequence differs"):
            self.trimmed._read_prediction(self.root / "original.pdb")
        with self.assertRaisesRegex(ValueError, "sequence differs"):
            self.original._read_prediction(self.root / "trimmed.pdb")


class TrimmedExperimentConfigTests(unittest.TestCase):
    def test_trimmed_experiment_does_not_score_incompatible_full_length_baseline(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "configs/experiment/2h62-inpaint-fk-l61-bmpr1a-trimmed.yaml").read_text())
        self.assertFalse(config["sampling"]["run_baseline"])
        self.assertIsNone(config["sampling"]["baseline_prediction_dir"])


if __name__ == "__main__":
    unittest.main()
