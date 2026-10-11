"""Refinement comparisons preserve copy identity and exact construct sequence."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import gemmi
import numpy as np

from scripts.analyze_rosetta_refinement import analyze, canonicalize, drift
from scripts.compare_structures import ChainCA
from scripts.run_rosetta_refinement import prepare_input


ROOT = Path(__file__).resolve().parents[1]


class CorrespondenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        structure = gemmi.Structure()
        model = gemmi.Model("1")
        for chain_name in "ABCDEF":
            chain = gemmi.Chain(chain_name)
            for index, letter in enumerate("ACDE"):
                residue = gemmi.Residue()
                residue.name = gemmi.expand_one_letter(letter, gemmi.ResidueKind.AA)
                residue.seqid = gemmi.SeqId(101 + index, " ")
                atom = gemmi.Atom()
                atom.name, atom.element = "CA", gemmi.Element("C")
                atom.pos = gemmi.Position(index, ord(chain_name), index % 2)
                atom.occ = 1.
                residue.add_atom(atom)
                chain.add_residue(residue)
            model.add_chain(chain)
        structure.add_model(model)
        self.structure = structure
        self.sequences = dict.fromkeys("ABCDEF", "ACDE")

    def test_known_chain_mapping_and_changed_residue_numbers(self):
        self.structure[0]["C"].name = "X"
        source, output = self.root / "input.pdb", self.root / "canonical.pdb"
        self.structure.write_pdb(str(source))
        raw = source.read_bytes()
        result = canonicalize(source, output, self.sequences, {"X": "C"})
        self.assertEqual(source.read_bytes(), raw)
        chain = gemmi.read_structure(str(output))[0]["C"]
        self.assertEqual([str(r.seqid) for r in chain], ["1", "2", "3", "4"])
        self.assertEqual(result["chain_residue_mapping"]["C"]["source_chain"], "X")
        self.assertEqual(result["chain_residue_mapping"]["C"]["residues"][0]["source_residue"], "101")
        with self.assertRaisesRegex(ValueError, "known chain map"):
            canonicalize(source, output, self.sequences)

    def test_mutation_and_missing_ca_are_not_silently_aligned(self):
        source, output = self.root / "input.pdb", self.root / "canonical.pdb"
        self.structure[0]["C"][0].name = "VAL"
        self.structure.write_pdb(str(source))
        with self.assertRaisesRegex(ValueError, "sequence differs"):
            canonicalize(source, output, self.sequences)
        self.structure[0]["C"][0].name = "ALA"
        # Keep an atom so the PDB preserves the residue with its CA absent.
        self.structure[0]["C"][0][0].name = "CB"
        self.structure.write_pdb(str(source))
        with self.assertRaisesRegex(ValueError, "missing occupied CA"):
            canonicalize(source, output, self.sequences)


class DriftTests(unittest.TestCase):
    def setUp(self):
        shape = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]])
        self.chains = {name: ChainCA("ACDE", shape + [index * 10, index * 3, 0],
                                     [0, 1, 2, 3], ["1", "2", "3", "4"], name, "ACDE")
                       for index, name in enumerate("ABCDEF")}

    def test_global_rigid_motion_is_removed(self):
        rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        moving = {name: replace(chain, coordinates=chain.coordinates @ rotation + [10, 20, 30])
                  for name, chain in self.chains.items()}
        result = drift(self.chains, moving)
        self.assertLess(result["assembly_CA_RMSD_angstrom"], 1e-10)
        for row in result["per_chain"].values():
            self.assertLess(row["CA_RMSD_in_BMP2_frame_angstrom"], 1e-10)

    def test_receptor_translation_preserves_fold_but_changes_docking(self):
        moving = dict(self.chains)
        moving["E"] = replace(moving["E"], coordinates=moving["E"].coordinates + [3., 4., 0.])
        result = drift(self.chains, moving)["per_chain"]
        self.assertAlmostEqual(result["E"]["CA_RMSD_in_BMP2_frame_angstrom"], 5.)
        self.assertLess(result["E"]["CA_RMSD_after_independent_chain_fit_angstrom"], 1e-10)
        self.assertLess(result["F"]["CA_RMSD_in_BMP2_frame_angstrom"], 1e-10)


class ActualPilotTests(unittest.TestCase):
    def test_unchanged_actual_inputs_have_zero_deltas_and_saved_reward(self):
        cif = ROOT / "artifacts/structures/2h62-inpaint/prepared/observed.cif"
        reference = cif.with_suffix(".pdb")
        for job, index in (("58030762", 4), ("58030763", 5)):
            folder = ROOT / "artifacts/remote-runs" / job
            original = folder / f"fk/predictions_simplefold_3B/2h62-inpaint-bmpr1a-trimmed_sampled_{index}.pdb"
            if not original.exists() or not cif.exists():
                self.skipTest("Downloaded pilot artifacts are unavailable")
            with self.subTest(job=job):
                raw = original.read_bytes()
                receipt = folder / "fk-experiment.json"
                settings = json.loads(receipt.read_text())["config"]["reward"]
                with tempfile.TemporaryDirectory() as temporary:
                    prepared, disulfides = prepare_input(original, cif, reference, settings, temporary)
                    original_lines = [line for line in original.read_text().splitlines() if line.startswith("ATOM  ")]
                    prepared_lines = [line for line in prepared.read_text().splitlines() if line.startswith("ATOM  ")]
                    self.assertEqual(original_lines, prepared_lines)
                    topology = json.loads((Path(temporary) / "topology.json").read_text())
                    self.assertEqual(topology["residue_count"], 592)
                    self.assertEqual(len(topology["disulfides"]), 27)
                    fixed_pairs = [list(map(int, line.split())) for line in disulfides.read_text().splitlines()]
                    self.assertEqual(fixed_pairs, [row["pose_indices"] for row in topology["disulfides"]])
                    structure = gemmi.read_structure(str(prepared))
                    parsed = [connection for connection in structure.connections
                              if connection.type == gemmi.ConnectionType.Disulf]
                    self.assertEqual(len(parsed), 27)
                    parsed_pairs = {tuple(sorted(((connection.partner1.chain_name, str(connection.partner1.res_id.seqid)),
                                                  (connection.partner2.chain_name, str(connection.partner2.res_id.seqid)))))
                                    for connection in parsed}
                    expected_pairs = {tuple(sorted(tuple(residue) for residue in row["pdb_residues"]))
                                      for row in topology["disulfides"]}
                    self.assertEqual(parsed_pairs, expected_pairs)
                    pose_residues = [residue for chain in structure[0] for residue in chain]
                    for pair, row in zip(fixed_pairs, topology["disulfides"]):
                        for pose_index, (chain, number) in zip(pair, row["pdb_residues"]):
                            residue = pose_residues[pose_index - 1]
                            self.assertEqual(residue.name, "CYS")
                            self.assertIn("SG", residue)
                            self.assertEqual(str(residue.seqid), number)
                            self.assertEqual(structure[0][chain][number][0].name, "CYS")
                    result = analyze(original, prepared, prepared, cif, reference, receipt)
                saved = json.loads((folder / "fk/fk-summary.json").read_text())["final_rewards"][index]
                self.assertAlmostEqual(result["stages"]["original"]["reward"]["reward"], saved, places=10)
                self.assertEqual(original.read_bytes(), raw)
                self.assertEqual(result["comparisons"]["original_to_refined"]["reward_delta"], 0.)
                self.assertEqual(result["comparisons"]["original_to_refined"]["geometry_deltas"]["clash_count"], 0)
                self.assertLess(result["stages"]["refined"]["drift_from_original"]["assembly_CA_RMSD_angstrom"], 1e-10)
                copies = result["stages"]["original"]["original_assignment_copy_metrics"]
                self.assertEqual(len(copies), 4)
                self.assertEqual({row["prediction_chain"] for row in copies}, set("CDEF"))
                # JSON receipt contains no NaN even for absent contacts.
                json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
