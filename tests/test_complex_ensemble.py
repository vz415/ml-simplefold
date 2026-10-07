"""Coordinate-only regressions for multimer symmetry and ensemble statistics."""
import argparse
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import gemmi
import numpy as np

from scripts.compare_structures import ChainCA
from scripts.analyze_complex_ensemble import (GROUPS, REF_ORDER, analyze, best_fit,
                                               distance_metrics, diversity_references, matched_coordinates, mappings,
                                               read_prediction, summary)


def fixture():
    rng = np.random.default_rng(923)
    refs, predicted = {}, {}
    sequences = ("ACDEFG", "HIKLMN", "PQRSTV")
    for group_index, (_, ref_names, pred_names) in enumerate(GROUPS):
        for copy_index, (ref_name, pred_name) in enumerate(zip(ref_names, pred_names)):
            full = rng.normal(size=(6, 3)) * 3 + [group_index * 12, copy_index * 17, 4]
            positions = [0, 1, 2, 3, 5] if copy_index == 0 else [1, 2, 3, 4, 5]
            sequence = sequences[group_index]
            refs[ref_name] = ChainCA("".join(sequence[p] for p in positions), full[positions],
                                    positions, [str(p + 1) for p in positions], ref_name, sequence)
            predicted[pred_name] = ChainCA(sequence, full, list(range(6)), [], pred_name, sequence)
    return refs, predicted


def write_chains(path, chains):
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    for name, data in sorted(chains.items()):
        chain = gemmi.Chain(name)
        for index, (letter, xyz) in enumerate(zip(data.sequence, data.coordinates)):
            residue = gemmi.Residue()
            residue.name = gemmi.expand_one_letter(letter, gemmi.ResidueKind.AA)
            residue.seqid = gemmi.SeqId(index + 1, " ")
            residue.het_flag = "A"
            atom = gemmi.Atom()
            atom.name, atom.element = "CA", gemmi.Element("C")
            atom.pos = gemmi.Position(*xyz)
            atom.occ = 1
            residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    structure.write_pdb(str(path))


class ComplexEnsembleTests(unittest.TestCase):
    def test_copy_swap_and_rigid_transform_recovered_with_missing_termini(self):
        refs, predicted = fixture()
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        swapped = {}
        for _, _, names in GROUPS:
            for name, source in zip(names, names[::-1]):
                data = predicted[source]
                xyz = data.coordinates @ rotation + [40, -19, 5]
                swapped[name] = ChainCA(data.sequence, xyz, data.full_positions, [], name, data.sequence)
        fixed = np.concatenate([refs[c].coordinates for c in REF_ORDER])
        rmsd, mapping, _, fitted, _ = best_fit(swapped, refs, fixed)
        self.assertLess(rmsd, 1e-12)
        self.assertAlmostEqual(np.linalg.det(fitted), 1)
        self.assertEqual(mapping, {"A": "B", "D": "A", "B": "D", "E": "C", "C": "F", "F": "E"})
        self.assertEqual(len(list(mappings())), 8)

    def test_contact_loss_and_lddt_are_sensitive_to_assembly_displacement(self):
        fixed = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0],
                          [0, 0, 4], [1, 0, 4], [2, 1, 4]], float)
        labels = np.array(list("AAABBB"))
        exact = distance_metrics(fixed, fixed + [12, 17, 5], labels)
        self.assertEqual(exact["assembly_CA_lddt"], 1)
        self.assertEqual(exact["CA_interchain_contact_recall"], 1)
        displaced = fixed.copy()
        displaced[3:] += [0, 0, 20]
        bad = distance_metrics(fixed, displaced, labels)
        self.assertEqual(bad["CA_interchain_contact_recall"], 0)
        self.assertIsNone(bad["CA_interchain_contact_precision"])
        self.assertAlmostEqual(bad["assembly_CA_lddt"], 6 / 15)

    def test_sample_sd_and_undefined_values(self):
        result = summary([1, 2, 3, None])
        self.assertEqual(result, dict(n=3, mean=2, sd=1, median=2, min=1, max=3))
        self.assertIsNone(summary([4])["sd"])
        self.assertEqual(summary([None])["n"], 0)

    def test_full_sequence_validation_rejects_wrong_component(self):
        refs, predictions = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.pdb"
            write_chains(path, predictions)
            parsed = read_prediction(path, refs)
            self.assertEqual(sum(len(c.coordinates) for c in parsed.values()), 36)
            predictions["C"] = predictions["A"]
            write_chains(path, predictions)
            with self.assertRaisesRegex(ValueError, "sequence differs"):
                read_prediction(path, refs)

    def test_ten_samples_produce_all_45_pairs_and_validate_sample_count(self):
        rng = np.random.default_rng(412)
        refs, predictions = {}, {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observed = dict(zip(REF_ORDER, [103, 85, 92, 104, 88, 93]))
            for group, length, letter in zip(GROUPS, (114, 131, 102), "ACD"):
                _, ref_names, pred_names = group
                for rc, pc in zip(ref_names, pred_names):
                    xyz = rng.normal(size=(length, 3)) * 8
                    sequence = letter * length
                    positions = list(range(length - observed[rc], length))
                    refs[rc] = ChainCA(letter * observed[rc], xyz[positions], positions,
                                       [str(p + 1) for p in positions], rc, sequence)
                    predictions[pc] = ChainCA(sequence, xyz, list(range(length)), [], pc, sequence)
            # PDB author numbering includes unresolved prefixes.
            structure = gemmi.Structure()
            model = gemmi.Model("1")
            for c in REF_ORDER:
                chain = gemmi.Chain(c)
                for position, xyz in zip(refs[c].full_positions, refs[c].coordinates):
                    residue = gemmi.Residue()
                    residue.name = gemmi.expand_one_letter(refs[c].full_sequence[position], gemmi.ResidueKind.AA)
                    residue.seqid = gemmi.SeqId(position + 1, " ")
                    residue.het_flag = "A"
                    atom = gemmi.Atom()
                    atom.name, atom.element, atom.occ = "CA", gemmi.Element("C"), 1
                    atom.pos = gemmi.Position(*xyz)
                    residue.add_atom(atom)
                    chain.add_residue(residue)
                model.add_chain(chain)
            structure.add_model(model)
            reference = root / "reference.pdb"
            structure.write_pdb(str(reference))
            cif = root / "reference.cif"
            cif.write_text("mocked mmCIF reader for full-length integration fixture")
            pred_dir = root / "predictions"
            pred_dir.mkdir()
            for i in range(10):
                write_chains(pred_dir / f"sample_{i}.pdb", predictions)
            args = argparse.Namespace(reference=cif, reference_pdb=reference,
                                      prediction_dir=[f"100M={pred_dir}"], output_dir=root / "out", expected_samples=10)
            with patch("scripts.analyze_complex_ensemble.reference_ca", side_effect=lambda path, c: refs[c]):
                report = analyze(args)
                args.expected_samples = 11
                with self.assertRaisesRegex(ValueError, "expected 11 sample PDBs, found 10"):
                    analyze(args)
            model = report["models"][0]
            self.assertEqual(model["diversity"]["pair_count"], 45)
            self.assertLess(model["diversity"]["summary"]["max"], 1e-12)
            self.assertLess(model["summary"]["global_CA_RMSD_angstrom"]["max"], 0.001)
            self.assertEqual(model["summary"]["assembly_CA_lddt"]["mean"], 1)
            self.assertEqual(report["matched_reference_CA"], 565)
            self.assertEqual(report["matched_diversity_CA"], 560)
            for sample in model["samples"]:
                saved = gemmi.read_structure(str(args.output_dir / sample["aligned_pdb"]))
                self.assertEqual([c.name for c in saved[0]], list(REF_ORDER))

    def test_diversity_common_mask_is_sample_order_and_copy_label_invariant(self):
        refs, first = fixture()
        common = diversity_references(refs)
        for _, names, _ in GROUPS:
            self.assertEqual(common[names[0]].full_positions, common[names[1]].full_positions)
            self.assertEqual(common[names[0]].full_positions, [1, 2, 3, 5])
        rng = np.random.default_rng(342)
        second = {c: ChainCA(data.sequence, data.coordinates + rng.normal(size=data.coordinates.shape),
                             data.full_positions, [], c, data.sequence) for c, data in first.items()}
        canonical = next(mappings())
        first_coords = matched_coordinates(first, common, canonical)
        second_coords = matched_coordinates(second, common, canonical)
        forward, *_ = best_fit(second, common, first_coords)
        backward, *_ = best_fit(first, common, second_coords)
        self.assertGreater(forward, 0.1)
        self.assertAlmostEqual(forward, backward, places=12)
        renamed_first, renamed_second = dict(first), dict(second)
        for _, _, names in GROUPS:
            renamed_first[names[0]], renamed_first[names[1]] = first[names[1]], first[names[0]]
            renamed_second[names[0]], renamed_second[names[1]] = second[names[1]], second[names[0]]
        renamed_coords = matched_coordinates(renamed_first, common, canonical)
        rename_first, *_ = best_fit(second, common, renamed_coords)
        rename_second, *_ = best_fit(renamed_second, common, first_coords)
        self.assertAlmostEqual(forward, rename_first, places=12)
        self.assertAlmostEqual(forward, rename_second, places=12)

    def test_diversity_is_copy_permutation_invariant(self):
        refs, predicted = fixture()
        fixed = np.concatenate([refs[c].coordinates for c in REF_ORDER])
        _, _, matched, _, _ = best_fit(predicted, refs, fixed)
        reordered = dict(predicted)
        for _, _, names in GROUPS:
            reordered[names[0]], reordered[names[1]] = predicted[names[1]], predicted[names[0]]
        rmsd, *_ = best_fit(reordered, refs, matched)
        self.assertLess(rmsd, 1e-12)
        reordered["A"].coordinates = reordered["A"].coordinates + [8, 2, -2]
        rmsd, *_ = best_fit(reordered, refs, matched)
        self.assertGreater(rmsd, 1)


if __name__ == "__main__":
    unittest.main()
