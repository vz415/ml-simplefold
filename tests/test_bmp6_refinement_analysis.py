"""Refinement analysis keeps copy identity, chemistry and provenance fixed."""
from argparse import Namespace
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import analyze_bmp6_refinement as analysis
from scripts.structure_clashes import Atom

ROOT = Path(__file__).resolve().parents[1]


class FixedIdentityTests(unittest.TestCase):
    def setUp(self):
        shape = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]])
        self.chains = [{"input_chain": c, "gene": "ligand" if c in "AB" else "receptor",
                        "sequence": "AAAA"} for c in "ABCDEF"]
        self.atoms = {}
        for ci, c in enumerate("ABCDEF"):
            for p, point in enumerate(shape + [ci*10, ci*3, 0], 1):
                for ai, name in enumerate(analysis.BACKBONE):
                    key = (c, p, name)
                    self.atoms[key] = Atom(len(self.atoms)+1, c, p, "", "ALA", name,
                                           name[0], tuple(point+[0, 0, ai*.2]), 1., "", ci)

    def test_global_rigid_motion_removed_but_receptor_translation_retained(self):
        rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        moving = {k: replace(a, xyz=tuple(np.array(a.xyz)@rotation+[10, 20, 30])) for k, a in self.atoms.items()}
        zero = analysis.drift(self.atoms, moving, self.chains)
        self.assertLess(zero["backbone"]["assembly_fitted_RMSD_A"], 1e-10)
        shifted = {k: replace(a, xyz=tuple(np.array(a.xyz)+([3, 4, 0] if k[0] == "E" else [0, 0, 0])))
                   for k, a in self.atoms.items()}
        row = analysis.drift(self.atoms, shifted, self.chains)["backbone"]["per_input_chain"]["E"]
        self.assertAlmostEqual(row["ligand_frame_RMSD_A"], 5.)
        self.assertLess(row["independently_fitted_RMSD_A"], 1e-10)

    def test_assignment_cannot_exchange_genes_or_drop_a_copy(self):
        bad = dict(analysis.IDENTITY, A="C", C="A")
        with self.assertRaisesRegex(ValueError, "different genes"):
            analysis.validate_assignment(bad, self.chains)
        with self.assertRaisesRegex(ValueError, "bijection"):
            analysis.validate_assignment(dict(analysis.IDENTITY, E="F"), self.chains)


class DownloadedBMP6Tests(unittest.TestCase):
    def setUp(self):
        self.model = ROOT / "artifacts/remote-runs/58134298/fk/predictions_simplefold_3B/BMP6_ACVR1_ACVR2B_deposited_native_no_tags_sampled_1.pdb"
        evaluation = ROOT / "artifacts/remote-runs/58134298/evaluation/paid_analysis_58134307/fk.json"
        topology = ROOT / "artifacts/bmp90/preparation/bmp6_rosetta_20261014/topology_and_restraints.json"
        if not all(p.exists() for p in (self.model, evaluation, topology)):
            self.skipTest("Downloaded BMP6 evaluation and annotated topology unavailable")
        self.args = Namespace(original=self.model, refined=self.model, original_source=None,
            mapping=ROOT/"examples/bmp90/BMP6_ACVR1_ACVR2B_deposited_native_no_tags.mapping.json",
            reference=ROOT/"artifacts/bmp90/preparation/bmp6/9N4K_deposited_native_reference.pdb",
            spec=ROOT/"artifacts/bmp90/preparation/bmp6_revision_v3_20261014/bmp6_joint_n60_local_v3.json",
            parent_spec=ROOT/"artifacts/bmp90/specifications/bmp6_joint_v1.json",
            fasta=ROOT/"examples/bmp90/BMP6_ACVR1_ACVR2B_deposited_native_no_tags.fasta",
            original_evaluation=evaluation, sample_index=1, topology=topology,
            dockq="skip", dockq_python="unused")
        self.chains = json.loads(self.args.mapping.read_text())["chains"]
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)

    def test_noop_reproduces_contacts_and_has_zero_deltas_without_assignment_search(self):
        raw = self.model.read_bytes()
        with patch.object(analysis.ev, "coherent_assignment", side_effect=AssertionError("No reassignment allowed")):
            result = analysis.analyze(self.args)
        self.assertEqual(self.model.read_bytes(), raw)
        self.assertEqual(result["comparisons"]["fixed_guidance_reward_deltas"], {"current": 0., "parent": 0.})
        self.assertLess(result["comparisons"]["observed_core_drift_from_original"]["backbone"]["assembly_fitted_RMSD_A"], 1e-10)
        for stage in result["stages"].values():
            self.assertEqual(stage["fixed_assignment"], result["original_fixed_assignment"])
            self.assertEqual(len(stage["physical_checks"]["fixed_disulfide_distances"]), 27)
            for key in ("current", "parent"):
                self.assertEqual(stage["fixed_assignment_guidance"][key]["assignments_evaluated"], 1)
            for now, old in zip(stage["interfaces"], result["historical_interfaces"]):
                self.assertEqual(now["all_native_contacts"], old["all_native_contacts"])
        self.assertEqual(result["stages"]["original"]["physical_checks"], result["stages"]["refined"]["physical_checks"])
        self.assertEqual(result["stages"]["original"]["local_geometry_under_original_reference_assignment"],
                         result["stages"]["refined"]["local_geometry_under_original_reference_assignment"])
        json.dumps(result, allow_nan=False)

    def test_swapped_receptors_are_not_remapped_away(self):
        path = self.tmp/"swapped.pdb"
        rows = []
        for line in self.model.read_text().splitlines(keepends=True):
            if line.startswith("ATOM  ") and line[21] in "CD":
                line = line[:21]+{"C": "D", "D": "C"}[line[21]]+line[22:]
            rows.append(line)
        path.write_text("".join(rows))
        self.args.refined = path
        with patch.object(analysis.ev, "coherent_assignment", side_effect=AssertionError("No reassignment allowed")):
            result = analysis.analyze(self.args)
        self.assertEqual(result["stages"]["refined"]["fixed_assignment"], result["original_fixed_assignment"])
        for c in "CD":
            row = result["comparisons"]["drift_from_original"]["CA"]["per_input_chain"][c]
            self.assertGreater(row["ligand_frame_RMSD_A"], 10.)

    def test_missing_sidechain_atom_and_wrong_historical_hash_rejected(self):
        path = self.tmp/"missing.pdb"
        lines = self.model.read_text().splitlines(keepends=True)
        removed = next(i for i, l in enumerate(lines) if l.startswith("ATOM  ") and l[12:16].strip() == "CB")
        path.write_text("".join(lines[:removed]+lines[removed+1:]))
        with self.assertRaisesRegex(ValueError, "Incomplete/noncanonical heavy atoms"):
            analysis.strict_atoms(path, self.chains)
        self.args.sample_index = 2
        with self.assertRaisesRegex(ValueError, "Original source hash"):
            analysis.analyze(self.args)


if __name__ == "__main__":
    unittest.main()
