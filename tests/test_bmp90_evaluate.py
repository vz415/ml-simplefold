"""Reference-backed deterministic checks; no model inference or downloads."""
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bmp90_evaluate as ev
from bmp90_prepare_reference import atom_rows

PREP = ROOT / "artifacts/bmp90/preparation/bmp6"
REFERENCE = PREP / "9N4K_deposited_native_reference.pdb"


class AlignmentTests(unittest.TestCase):
    def test_proper_rigid_alignment(self):
        reference = np.array([[0, 0, 0], [1, 2, 0], [0, 1, 4], [4, 1, 1]], dtype=float)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        mobile = reference @ rotation + [13, -7, 4]
        r, t, rmsd = ev.fit(mobile, reference)
        self.assertAlmostEqual(rmsd, 0., places=12)
        self.assertAlmostEqual(np.linalg.det(r), 1., places=12)

    def test_missing_contacts_keep_denominator(self):
        result = ev.contact_recovery({("C", 1, "A", 2): 4., ("C", 3, "B", 4): 3.}, {})
        self.assertEqual(result["fraction_native_contacts"], 0.)
        self.assertIsNone(ev.contact_recovery({}, {})["fraction_native_contacts"])


@unittest.skipUnless(REFERENCE.exists(), "Needs pinned ignored 9N4K preparation bundle")
class ReferenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = json.loads((PREP / "deposited_native_no_tags_mapping.json").read_text())
        cls.chains = cls.mapping["chains"]
        cls.reference, _ = ev.read_atoms(REFERENCE)

    def test_extracted_atoms_are_exact_and_missing_tails_stay_missing(self):
        source_to_input = {c["reference_author_chain"]: c for c in self.chains}
        count = 0
        for a in atom_rows(PREP / "9N4K.cif"):
            if a["group_PDB"] != "ATOM":
                continue
            c = source_to_input[a["auth_asym_id"]]
            pos = int(a["auth_seq_id"]) - c["range_1based_inclusive"][0] + 1
            observed = self.reference[(c["input_chain"], pos, a["label_atom_id"])]
            self.assertEqual(observed.xyz, tuple(float(a[x]) for x in ("Cartn_x", "Cartn_y", "Cartn_z")))
            count += 1
        self.assertEqual(count, 4394)
        coverage = ev.validate_atoms(self.reference, self.chains, False)
        self.assertEqual(sum(x["observed_CA_residues"] for x in coverage.values()), 552)
        self.assertEqual(sum(len(x["missing_CA_input_indices"]) for x in coverage.values()), 164)
        with self.assertRaisesRegex(ValueError, "missing"):
            ev.validate_atoms(self.reference, self.chains, True)

    def test_self_and_rigid_transform_keep_assignment_and_contacts(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        moved = {k: dataclasses.replace(a, xyz=tuple(np.asarray(a.xyz) @ rotation+[41, -8, 17])) for k, a in self.reference.items()}
        best, alternatives = ev.coherent_assignment(self.reference, moved, self.chains, ["A", "B"])
        self.assertEqual(len(alternatives), 8)
        self.assertLess(best["objective_A_squared"], 1e-20)
        assigned = ev.rename_assignment(moved, best["reference_to_model_input_chain"])
        for rec in "CDEF":
            native = ev.native_contacts(self.reference, rec, ["A", "B"])
            predicted = ev.native_contacts(assigned, rec, ["A", "B"])
            self.assertEqual(ev.contact_recovery(native, predicted)["fraction_native_contacts"], 1.)

    def test_chain_permutation_is_global_and_bijective(self):
        swaps = dict(zip("ABCDEF", "BADCFE"))
        swapped = {(swaps[c], p, n): a for (c, p, n), a in self.reference.items()}
        best, _ = ev.coherent_assignment(self.reference, swapped, self.chains, ["A", "B"])
        self.assertEqual(best["reference_to_model_input_chain"], swaps)
        self.assertLess(best["objective_A_squared"], 1e-20)

    def test_duplicated_site_cannot_pass_both_receptor_copies(self):
        duplicated = dict(self.reference)
        for (c, p, n), a in self.reference.items():
            if c == "C":
                duplicated[("D", p, n)] = a
        best, _ = ev.coherent_assignment(self.reference, duplicated, self.chains, ["A", "B"])
        assigned = ev.rename_assignment(duplicated, best["reference_to_model_input_chain"])
        fractions = [ev.contact_recovery(ev.native_contacts(self.reference, r, ["A", "B"]),
                                         ev.native_contacts(assigned, r, ["A", "B"]))["fraction_native_contacts"] for r in "CD"]
        self.assertEqual(min(fractions), 0.)
        self.assertGreater(max(best["receptor_CA_rmsd_in_ligand_frame_A"].values()), 20.)

    def test_displaced_type_two_copy_is_visible(self):
        displaced = {k: dataclasses.replace(a, xyz=tuple(np.asarray(a.xyz)+[60, 0, 0])) if k[0] == "E" else a
                     for k, a in self.reference.items()}
        best, _ = ev.coherent_assignment(self.reference, displaced, self.chains, ["A", "B"])
        assigned = ev.rename_assignment(displaced, best["reference_to_model_input_chain"])
        fractions = [ev.contact_recovery(ev.native_contacts(self.reference, r, ["A", "B"]),
                                         ev.native_contacts(assigned, r, ["A", "B"]))["fraction_native_contacts"] for r in "EF"]
        self.assertEqual(min(fractions), 0.)

    def test_joint_holdout_agrees_with_independently_generated_ledger(self):
        path = ROOT / "artifacts/bmp90/specifications/bmp6_joint_v1.json"
        if not path.exists():
            self.skipTest("Joint recipe not present")
        excluded, _ = ev.guidance_info(json.loads(path.read_text()), self.chains)
        expected = json.loads(path.with_suffix(".heldout.json").read_text())
        contacts = {k for rec in "CDEF" for k in ev.native_contacts(self.reference, rec, ["A", "B"])
                    if k[:2] not in excluded and k[2:] not in excluded}
        ledger = {(r["left"]["chain"], r["left"]["position"], r["right"]["chain"], r["right"]["position"]) for r in expected["contacts"]}
        self.assertEqual(contacts, ledger)
        self.assertEqual(len(contacts), 167)

    def test_binary_export_preserves_full_numbering_and_observed_mask(self):
        lengths = {c["input_chain"]: c["length"] for c in self.chains}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "binary.pdb"
            ev.export_binary(path, self.reference, self.reference, "C", ["A", "B"], lengths)
            atoms, _ = ev.read_atoms(path)
        residues = {(c, p) for c, p, _ in atoms}
        self.assertEqual(len(residues), 104+104+79)
        self.assertEqual({p for c, p in residues if c == "X"}, set(range(36, 140)) | set(range(175, 279)))
        self.assertEqual({p for c, p in residues if c == "Y"}, set(range(11, 90)))


if __name__ == "__main__":
    unittest.main()
