"""Native receptor docking tests using synthetic coordinates, not model weights."""
import copy
import tempfile
import unittest
from pathlib import Path

import gemmi
import numpy as np

from scripts.compare_structures import ChainCA, kabsch
from scripts.complex_interfaces import GROUPS, InterfaceReference, _contacts, _dihedral


def fixture():
    rng = np.random.default_rng(234)
    rotation = np.diag([-1., -1., 1.])
    references, predictions, reference_atoms, prediction_atoms = {}, {}, {}, {}
    sequences = ("ACDEFG", "AGTEFQGKQ", "AAFWFA")
    authors = ([str(i) for i in range(11, 17)], [str(i) for i in range(81, 90)],
               ["40", "41", "42", "60", "83", "84"])
    for index, ((_, refs, preds), sequence, numbers) in enumerate(zip(GROUPS, sequences, authors)):
        xyz = rng.normal(size=(len(sequence), 3)) * .55 + [5, index * 1.1, 0]
        for copy_index, (rc, pc) in enumerate(zip(refs, preds)):
            coordinates = xyz if copy_index == 0 else xyz @ rotation
            positions = list(range(len(sequence)))
            references[rc] = ChainCA(sequence, coordinates, positions, numbers, rc, sequence)
            predictions[pc] = ChainCA(sequence, coordinates.copy(), positions,
                                     [str(i + 1) for i in positions], pc, sequence)
            for position, center in enumerate(coordinates):
                for atom, displacement in (("N", [-.3, 0, .1]), ("CA", [0, 0, 0]),
                                           ("C", [.3, .1, 0]), ("O", [.4, .2, 0]),
                                           ("CB", [0, -.2, -.3])):
                    shift = np.array(displacement) @ rotation if copy_index else np.array(displacement)
                    reference_atoms[(rc, position, atom)] = center + shift
                    prediction_atoms[(pc, position, atom)] = center + shift
    return references, predictions, reference_atoms, prediction_atoms


def write_atoms(path, chains, atoms):
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    for name, data in sorted(chains.items()):
        chain = gemmi.Chain(name)
        for position, author in zip(data.full_positions, data.author_residues):
            residue = gemmi.Residue()
            residue.name = gemmi.expand_one_letter(data.full_sequence[position], gemmi.ResidueKind.AA)
            residue.seqid, residue.het_flag = gemmi.SeqId(author), "A"
            for (source_chain, source_position, atom_name), xyz in atoms.items():
                if (source_chain, source_position) != (name, position):
                    continue
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(atom_name[0])
                atom.pos, atom.occ = gemmi.Position(*xyz), 1
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    structure.write_pdb(str(path))


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.refs, self.pred, self.ref_atoms, self.pred_atoms = fixture()
        self.ref_pdb = self.root / "reference.pdb"
        self.pred_pdb = self.root / "prediction.pdb"
        write_atoms(self.ref_pdb, self.refs, self.ref_atoms)
        self.prepared = InterfaceReference(self.ref_pdb, self.refs)

    def score(self, chains=None, atoms=None):
        chains = self.pred if chains is None else chains
        atoms = self.pred_atoms if atoms is None else atoms
        write_atoms(self.pred_pdb, chains, atoms)
        return self.prepared.analyze(self.pred_pdb, chains)

    def test_native_perfect_contact_recovery_and_explicit_author_map(self):
        result = self.score()
        self.assertLess(result["metrics"]["BMP2_dimer_CA_RMSD_angstrom"], 1e-12)
        self.assertLess(result["metrics"]["reference_anchored_receptor_placement_CA_RMSD_angstrom"], 1e-12)
        for row in result["receptors"]:
            self.assertEqual(row["native_contact_recall"], 1)
            self.assertEqual(row["native_contact_precision"], 1)
            self.assertLess(row["interface_backbone_RMSD_angstrom"], 1e-10)
            for hotspot in row["hotspots"]:
                self.assertEqual(hotspot["status"], "ok")
                self.assertEqual(hotspot["contact_recall"], 1)
                self.assertEqual(hotspot["heavy_atom_coverage"], 1)
        wrist = result["receptors"][0]
        self.assertEqual(wrist["hotspots"][0]["reference_author_residue"], "85")
        self.assertEqual(wrist["hotspots"][0]["reference_full_sequence_position"], 5)
        self.assertEqual(wrist["hotspots"][0]["prediction_author_residue"], "5")

    def test_rigid_transform_and_all_copy_swaps_preserve_scores(self):
        baseline = self.score()
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        translation = np.array([19., -22., 7.])
        swapped, atoms, renaming = {}, {}, {}
        for _, _, names in GROUPS:
            renaming.update(zip(names, names[::-1]))
        for name, source in renaming.items():
            data = self.pred[source]
            swapped[name] = ChainCA(data.sequence, data.coordinates @ rotation + translation,
                                    data.full_positions, data.author_residues, name, data.full_sequence)
        for (chain, pos, atom), xyz in self.pred_atoms.items():
            atoms[(renaming[chain], pos, atom)] = xyz @ rotation + translation
        transformed = self.score(swapped, atoms)
        for key, value in baseline["metrics"].items():
            if value is not None:
                self.assertAlmostEqual(value, transformed["metrics"][key], places=5, msg=key)
        r, t = np.array(transformed["rotation"]), np.array(transformed["translation"])
        self.assertAlmostEqual(np.linalg.det(r), 1)
        for rc in ("A", "D"):
            aligned = swapped[transformed["mapping"][rc]].coordinates @ r + t
            self.assertLess(np.linalg.norm(aligned - self.refs[rc].coordinates), 1e-10)

    def test_receptor_displacement_loses_contacts_despite_unchanged_fold(self):
        pred, atoms = copy.deepcopy(self.pred), copy.deepcopy(self.pred_atoms)
        shift = np.array([0., 0., 30.])
        pred["C"].coordinates += shift
        for key in atoms:
            if key[0] == "C":
                atoms[key] += shift
        result = self.score(pred, atoms)
        row = next(row for row in result["receptors"] if row["prediction_chain"] == "C")
        self.assertAlmostEqual(row["ligand_frame_CA_RMSD_angstrom"], 30)
        self.assertAlmostEqual(row["centroid_displacement_angstrom"], 30)
        self.assertLess(row["orientation_error_degrees"], 1e-5)
        self.assertEqual(row["native_contact_recall"], 0)
        self.assertIsNone(row["native_contact_precision"])
        self.assertGreater(row["interface_backbone_RMSD_angstrom"], 8)
        _, _, errors = kabsch(pred["C"].coordinates, self.refs["B"].coordinates)
        self.assertLess(np.sqrt(np.mean(errors ** 2)), 1e-10)

    def test_symmetrically_wrong_docking_cannot_pass_placement(self):
        pred, atoms = copy.deepcopy(self.pred), copy.deepcopy(self.pred_atoms)
        for chain, shift in (("C", [0, 0, 30]), ("D", [0, 0, 30])):
            pred[chain].coordinates += shift
            for key in atoms:
                if key[0] == chain:
                    atoms[key] += shift
        result = self.score(pred, atoms)
        self.assertLess(result["metrics"]["reference_anchored_C2_CA_deviation_angstrom"], 1e-10)
        self.assertGreater(result["metrics"]["reference_anchored_receptor_placement_CA_RMSD_angstrom"], 10)
        self.assertEqual(result["metrics"]["BMPR1A_native_contact_recall"], 0)

    def test_invalid_or_missing_hotspot_identity_is_explicitly_unavailable(self):
        refs, atoms = copy.deepcopy(self.refs), copy.deepcopy(self.ref_atoms)
        for rc in ("B", "E"):
            refs[rc].author_residues[4] = "185"  # No guessing residue85 by polymer position.
        path = self.root / "missing.pdb"
        write_atoms(path, refs, atoms)
        prepared = InterfaceReference(path, refs)
        h = prepared.metadata["reference_receptors"][0]["hotspots"][0]
        self.assertEqual(h["status"], "unresolved_reference_residue")
        self.assertIsNone(h["contact_recall"])
        for rc in ("B", "E"):
            refs[rc].author_residues[4] = "85"
            refs[rc].sequence = refs[rc].full_sequence = refs[rc].sequence[:4] + "A" + refs[rc].sequence[5:]
        write_atoms(path, refs, atoms)
        prepared = InterfaceReference(path, refs)
        h = prepared.metadata["reference_receptors"][0]["hotspots"][0]
        self.assertEqual(h["status"], "reference_residue_identity_mismatch")
        self.assertIsNone(h["contact_recall"])

    def test_missing_atoms_lower_coverage_and_do_not_create_false_contacts(self):
        atoms = {key: xyz for key, xyz in self.pred_atoms.items() if not (key[0] == "C" and key[1] == 4)}
        result = self.score(atoms=atoms)
        row = next(row for row in result["receptors"] if row["prediction_chain"] == "C")
        hotspot = row["hotspots"][0]
        self.assertEqual(hotspot["status"], "missing_prediction_residue_atoms")
        self.assertEqual(hotspot["heavy_atom_coverage"], 0)
        self.assertEqual(hotspot["contact_recall"], 0)
        self.assertLess(row["native_contact_recall"], 1)
        self.assertEqual(row["native_contact_precision"], 1)
        self.assertLess(result["metrics"]["reference_heavy_atom_coverage"], 1)
        partial = dict(self.pred_atoms)
        del partial[("C", 4, "CB")]
        partial_row = next(row for row in self.score(atoms=partial)["receptors"] if row["prediction_chain"] == "C")
        self.assertEqual(partial_row["hotspots"][0]["status"], "partial_prediction_atoms")
        self.assertAlmostEqual(partial_row["hotspots"][0]["heavy_atom_coverage"], .8)

    def test_contact_cutoff_inclusive_and_unknown_atoms_masked(self):
        atoms = {("B", 0, "CB"): np.array([0., 0., 0.]),
                 ("A", 0, "CB"): np.array([4., 0., 0.])}
        self.assertEqual(_contacts(atoms, "B", ("A", "D")), {(0, "A", 0)})
        atoms[("A", 0, "CB")][0] += .00001
        self.assertFalse(_contacts(atoms, "B", ("A", "D")))
        extra = dict(self.pred_atoms)
        extra[("C", 4, "XX")] = self.pred_atoms[("A", 0, "CA")]
        baseline, expanded = self.score(), self.score(atoms=extra)
        self.assertEqual(baseline["metrics"]["BMPR1A_native_contact_precision"], expanded["metrics"]["BMPR1A_native_contact_precision"])

    def test_CA_only_input_does_not_report_backbone_or_helix_metrics(self):
        atoms = {key: xyz for key, xyz in self.pred_atoms.items() if key[2] == "CA"}
        result = self.score(atoms=atoms)
        self.assertIsNone(result["metrics"]["BMPR1A_alpha1_backbone_RMSD_angstrom"])
        self.assertIsNone(result["metrics"]["BMPR1A_alpha1_helix_dihedral_fraction"])
        self.assertIsNone(result["metrics"]["BMPR1A_interface_backbone_RMSD_angstrom"])

    def test_dihedral_convention_and_undefined_degenerate_geometry(self):
        a, b, c, d = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]], float)
        self.assertAlmostEqual(_dihedral(a, b, c, d), 90)
        self.assertIsNone(_dihedral(a, b, b, d))


if __name__ == "__main__":
    unittest.main()
