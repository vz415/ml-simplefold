"""Structural analysis tests: numbering, unresolved residues, alternates, fits."""
import argparse
import tempfile
import unittest
from pathlib import Path

import gemmi
import numpy as np

from scripts.compare_structures import ChainCA, compare, kabsch, matched_indices, reference_ca


def write_reference(path):
    columns = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
               "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
               "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
               "B_iso_or_equiv", "auth_seq_id", "auth_asym_id", "pdbx_PDB_model_num"]
    lines = ["data_fixture", "_entity_poly.entity_id 1",
             "_entity_poly.pdbx_seq_one_letter_code_can PQSACDEF", "loop_"]
    lines += ["_atom_site." + column for column in columns]
    xyz = np.array([[0, 0, 0], [3, 1, 0], [5, 2, 1], [6, 5, 2], [7, 7, 4]], float)
    for i, (residue, coord) in enumerate(zip(("ALA", "CYS", "ASP", "GLU", "PHE"), xyz)):
        x, y, z = coord
        insertion = "A" if i == 1 else "?"
        # Deposited full positions 4..8, unrelated author numbering 100..104.
        lines.append(f"ATOM {i+1} C CA B {residue} L 1 {i+4} {insertion} {x} {y} {z} 0.7 20 {i+100} A 1")
    # Lower-occupancy alternate must not win by file order or alternate ID A.
    lines.append("ATOM 6 C CA A ALA L 1 4 ? 99 99 99 0.3 20 100 A 1")
    path.write_text("\n".join(lines) + "\n")
    return xyz


def write_prediction(path, observed):
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    coords = np.vstack(([[50, 40, 30], [45, 40, 30], [40, 40, 30]], observed))
    for i, (name, coord) in enumerate(zip(("PRO", "GLN", "SER", "ALA", "CYS", "ASP", "GLU", "PHE"), coords)):
        residue = gemmi.Residue()
        residue.name = name
        residue.seqid = gemmi.SeqId(i+1, " ")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(*coord)
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.write_pdb(str(path))


class ComparisonTests(unittest.TestCase):
    def test_reference_full_numbering_and_altloc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.cif"
            xyz = write_reference(path)
            reference = reference_ca(path, "A")
            self.assertEqual(reference.full_sequence, "PQSACDEF")
            self.assertEqual(reference.sequence, "ACDEF")
            self.assertEqual(reference.full_positions, [3, 4, 5, 6, 7])
            self.assertEqual(reference.author_residues, ["100", "101A", "102", "103", "104"])
            np.testing.assert_allclose(reference.coordinates, xyz)

    def test_sequence_mapping_skips_unresolved_prefix(self):
        reference = ChainCA("AAAA", np.zeros((4, 3)), [2, 3, 4, 5], [], "A", "AAAAAA")
        prediction = ChainCA("AAAAAA", np.zeros((6, 3)), list(range(6)), [], "A", "AAAAAA")
        np.testing.assert_array_equal(matched_indices(reference, prediction), [[0, 2], [1, 3], [2, 4], [3, 5]])

    def test_kabsch_rigid_transform_and_no_reflection(self):
        fixed = np.array([[0, 0, 0], [3, 1, 0], [1, 3, 2], [0, 1, 4]], float)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
        moving = fixed @ rotation + [10, -4, 3]
        fitted_rotation, _, distances = kabsch(moving, fixed)
        np.testing.assert_allclose(distances, 0, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(fitted_rotation), 1)
        reflected = fixed * [-1, 1, 1]
        _, _, reflected_distances = kabsch(reflected, fixed)
        self.assertGreater(float(np.sqrt(np.mean(reflected_distances**2))), 0.1)

    def test_comparison_outputs_and_missing_residue_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.cif"
            xyz = write_reference(reference)
            predictions = root / "predictions"
            predictions.mkdir()
            write_prediction(predictions / "sample_0.pdb", xyz + [10, 20, -5])
            write_prediction(predictions / "sample_1.pdb", xyz + [12, 22, -6])
            args = argparse.Namespace(reference=reference, chain="A", prediction_dir=predictions,
                                      output_dir=root / "analysis", fasta=None)
            report = compare(args)
            self.assertEqual(report["reference_unresolved_positions"], [1, 2, 3])
            for row in report["metrics"]:
                self.assertAlmostEqual(row["coverage_full_reference"], 5/8)
                self.assertEqual(row["coverage_observed_reference"], 1)
                self.assertAlmostEqual(row["ca_rmsd_sequence_fit_angstrom"], 0, places=6)
                self.assertEqual(row["ca_lddt_pair_weighted"], 1)
                self.assertAlmostEqual(row["gdt_ts_single_fit_full_percent"], 62.5)
            for filename in ("metrics.csv", "comparison.json", "ca_distances.png", "best_superposition.html",
                             "pairwise_rmsd.csv", "pair_distance_error_distribution.png", "best_pair_distance_error_matrix.png",
                             "ensemble_superposition.html"):
                self.assertTrue((args.output_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
