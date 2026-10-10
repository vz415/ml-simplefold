"""Coordinate regressions for a partial native assembly, with no inference."""
from pathlib import Path
import unittest

import numpy as np

from scripts.analyze_2h62_baseline import natural_key, score_prediction
from scripts.compare_structures import ChainCA


def fixture():
    coordinates = {
        "A": np.array([[0., 0, 0], [2, 0, 0], [0, 2, 1]]),
        "B": np.array([[20., 0, 0], [21, 1, 0], [20, 2, 2]]),
    }
    coordinates["C"] = coordinates["A"] + [0, 0, 3]
    coordinates["D"] = coordinates["B"] + [0, 0, 3]
    def chain(name, xyz):
        return ChainCA("ACD", xyz, [0, 1, 2], ["1", "2", "3"], name, "ACD")
    references = {c: chain(c, xyz) for c, xyz in coordinates.items()}
    predicted = {
        "A": chain("A", coordinates["A"]), "B": chain("B", coordinates["B"]),
        "C": chain("C", coordinates["C"]), "D": chain("D", coordinates["C"] + [70, 0, 0]),
        "E": chain("E", coordinates["D"]), "F": chain("F", coordinates["D"] + [70, 0, 0]),
    }
    atoms = lambda chains: {(c, i, "CA"): xyz for c, data in chains.items()
                            for i, xyz in enumerate(data.coordinates)}
    return references, predicted, atoms(references), atoms(predicted)


class PartialAssemblyTests(unittest.TestCase):
    def test_identity_selects_observed_copies_and_excludes_extra_copies(self):
        refs, pred, native_atoms, pred_atoms = fixture()
        result, _, _ = score_prediction(refs, pred, native_atoms, pred_atoms)
        self.assertLess(result["BMP2_dimer_CA_RMSD_angstrom"], 1e-12)
        self.assertEqual(result["mapping"], dict(A="A", B="B", C="C", D="E"))
        self.assertEqual(result["unscored_receptor_chains"], ["D", "F"])
        self.assertEqual(len(result["receptors"]), 2)
        for receptor in result["receptors"]:
            self.assertEqual(receptor["native_contact_recovery"], 1)
            self.assertEqual(receptor["native_contact_precision"], 1)
            self.assertLess(receptor["placement_CA_RMSD_angstrom"], 1e-12)
        # Farther displacement of an unknown copy cannot reduce a native score.
        pred["D"].coordinates += [300, 0, 0]
        for key in pred_atoms:
            if key[0] == "D":
                pred_atoms[key] = pred_atoms[key] + [300, 0, 0]
        changed, _, _ = score_prediction(refs, pred, native_atoms, pred_atoms)
        self.assertEqual(result["receptors"], changed["receptors"])

    def test_equivalent_copies_and_rigid_transform_preserve_scores(self):
        refs, pred, native_atoms, _ = fixture()
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        swap = dict(A="B", B="A", C="D", D="C", E="F", F="E")
        transformed = {}
        for name, source in swap.items():
            data = pred[source]
            transformed[name] = ChainCA(data.sequence, data.coordinates @ rotation + [30, -40, 8],
                                        data.full_positions, data.author_residues, name, data.full_sequence)
        atoms = {(c, i, "CA"): xyz for c, data in transformed.items()
                 for i, xyz in enumerate(data.coordinates)}
        result, fitted, _ = score_prediction(refs, transformed, native_atoms, atoms)
        self.assertEqual(result["mapping"], dict(A="B", B="A", C="D", D="F"))
        self.assertAlmostEqual(np.linalg.det(fitted), 1)
        for receptor in result["receptors"]:
            self.assertLess(receptor["placement_CA_RMSD_angstrom"], 1e-12)
            self.assertEqual(receptor["native_contact_recovery"], 1)

    def test_lost_observed_atoms_reduce_recovery_without_lowering_native_denominator(self):
        refs, pred, native_atoms, pred_atoms = fixture()
        original, _, _ = score_prediction(refs, pred, native_atoms, pred_atoms)
        removed = {k: xyz for k, xyz in pred_atoms.items() if k[0] != "C"}
        result, _, _ = score_prediction(refs, pred, native_atoms, removed)
        self.assertEqual(result["receptors"][0]["native_contacts"], original["receptors"][0]["native_contacts"])
        self.assertEqual(result["receptors"][0]["native_contact_recovery"], 0)
        self.assertIsNone(result["receptors"][0]["native_contact_precision"])
        self.assertEqual(result["receptors"][1]["native_contact_recovery"], 1)

    def test_sample_order_is_natural(self):
        files = [Path(f"sample_{i}.pdb") for i in (9, 1, 10, 0, 2)]
        self.assertEqual([p.name for p in sorted(files, key=natural_key)],
                         ["sample_0.pdb", "sample_1.pdb", "sample_2.pdb", "sample_9.pdb", "sample_10.pdb"])


if __name__ == "__main__":
    unittest.main()
