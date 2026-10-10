"""Completion preparation preserves observed atoms and excludes modeled anchors."""
import json
import tempfile
import unittest
from pathlib import Path

import gemmi
import numpy as np

from scripts.compare_structures import reference_ca
from scripts.prepare_2h62_inpaint import CHAIN_SOURCES, prepare


def write_fixture(path):
    """Full construct lengths, unresolved termini, numbering and alternate atoms."""
    document = gemmi.cif.Document()
    block = document.add_new_block("fixture")
    poly = block.init_loop("_entity_poly.", ["entity_id", "type", "pdbx_seq_one_letter_code_can", "pdbx_strand_id"])
    for entity, length, strands in (("1", 114, "A,B"), ("2", 129, "C"), ("3", 98, "D")):
        poly.add_row([entity, gemmi.cif.quote("polypeptide(L)"), "A" * length, strands])
    seq = block.init_loop("_entity_poly_seq.", ["entity_id", "num", "mon_id", "hetero"])
    for entity, length in (("1", 114), ("2", 129), ("3", 98)):
        for position in range(1, length + 1):
            seq.add_row([entity, str(position), "ALA", "n"])
    asym = block.init_loop("_struct_asym.", ["id", "entity_id"])
    for chain, entity in zip("ABCD", ("1", "1", "2", "3")):
        asym.add_row([chain, entity])
    tags = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id",
            "label_asym_id", "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "Cartn_x",
            "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv", "auth_seq_id", "auth_comp_id",
            "auth_asym_id", "auth_atom_id", "pdbx_PDB_model_num"]
    atoms = block.init_loop("_atom_site.", tags)
    first = np.array([[3., 1., 2.], [4., 2., 1.], [2., 4., 3.], [5., 3., 6.]])
    rotation, translation = np.diag([-1., -1., 1.]), np.array([20., 10., 0.])
    positions = {"A": [10, 11, 12, 14], "B": [10, 11, 12, 14], "C": [34, 35, 36, 38], "D": [6, 7, 8, 10]}
    for chain_index, (chain, entity, xyz) in enumerate(zip("ABCD", ("1", "1", "2", "3"),
                                                         (first, first @ rotation + translation, first + 20, first + 30))):
        for i, (position, coord) in enumerate(zip(positions[chain], xyz)):
            variants = [("CA", "C", "A", .8, coord), ("N", "N", ".", 1., coord + [.1, .2, .3])]
            if i == 0:
                variants += [("CA", "C", "B", .2, coord + [2, 0, 0]),
                             ("CB", "C", ".", 0., coord + [0, 0, 2]),
                             ("H", "H", ".", 1., coord + [0, 0, 3])]
            for name, element, altloc, occupancy, point in variants:
                atoms.add_row(["ATOM", str(atoms.length() + 1), element, name, altloc, "ALA", chain, entity,
                               str(position), "A" if i == 0 else "?", *[f"{v:.3f}" for v in point],
                               str(occupancy), "17.50", str(100 * (chain_index + 1) + i), "ALA", chain, name, "1"])
    document.write_file(str(path))


def atom_rows(path):
    table = gemmi.cif.read(str(path)).sole_block().find_mmcif_category("_atom_site.")
    tags = [tag.split(".", 1)[1] for tag in table.tags]
    return [dict(zip(tags, row)) for row in table]


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.cif"
        write_fixture(self.source)
        self.original_source_bytes = self.source.read_bytes()
        self.output = self.root / "prepared"
        self.fasta = self.root / "target.fasta"
        self.report = prepare(self.source, self.output, self.fasta, self.root / "provenance.json")

    def test_native_atom_tokens_and_source_are_preserved(self):
        source_rows = atom_rows(self.source)
        output_rows = atom_rows(self.output / "observed.cif")
        self.assertEqual(set(row["auth_asym_id"] for row in output_rows), set("ABCD"))
        self.assertEqual(len(source_rows), len(output_rows))
        for chain in "ABCD":
            native = CHAIN_SOURCES[chain]
            before = [row for row in source_rows if row["auth_asym_id"] == native]
            after = [row for row in output_rows if row["auth_asym_id"] == chain]
            for source, output in zip(before, after):
                for field in source.keys() - {"id", "auth_asym_id", "label_asym_id"}:
                    self.assertEqual(source[field], output[field], (chain, field))
        self.assertEqual(self.source.read_bytes(), self.original_source_bytes)

    def test_known_regions_exclude_copies_hydrogens_and_zero_occupancy(self):
        mask = json.loads((self.output / "known_regions.json").read_text())
        for chain in "ABCD":
            record = mask["chains"][chain]
            self.assertEqual(len(record["known_full_positions_zero_based"]), 4)
            self.assertFalse(record["known_residue_mask"][0])
            # Both occupied CA alternates are explicitly retained, not averaged.
            self.assertEqual(len(record["occupied_heavy_atoms"]), 9)
            self.assertEqual(set(atom["element"] for atom in record["occupied_heavy_atoms"]), {"C", "N"})
            self.assertTrue(all(atom["occupancy"] > 0 for atom in record["occupied_heavy_atoms"]))
        for chain in "EF":
            self.assertEqual(mask["chains"][chain]["occupied_heavy_atoms"], [])
            self.assertFalse(any(mask["chains"][chain]["known_residue_mask"]))

    def test_full_sequences_and_label_author_numbering_remain_distinct(self):
        records = reference_ca(self.output / "observed.cif", "A")
        self.assertEqual(len(records.full_sequence), 114)
        self.assertEqual(records.full_positions, [9, 10, 11, 13])
        self.assertEqual(records.author_residues, ["100A", "101", "102", "103"])
        self.assertEqual(self.report["total_residues"], 682)
        self.assertEqual(self.report["chains"], {"A": 114, "B": 129, "C": 98, "D": 114, "E": 129, "F": 98})
        lines = self.fasta.read_text().splitlines()
        self.assertEqual(lines[::2], [f">{chain}|protein" for chain in "ABCDEF"])
        self.assertEqual(sum(map(len, lines[1::2])), 682)
        for chain in "ABCDEF":
            completed = reference_ca(self.output / "symmetry_completion.cif", chain)
            self.assertEqual(len(completed.full_sequence), self.report["chains"][chain])

    def test_transform_places_only_added_copies(self):
        transform = self.report["transform"]
        rotation, translation = np.array(transform["R"]), np.array(transform["t_angstrom"])
        self.assertAlmostEqual(transform["copy_swap_CA_RMSD_angstrom"], 0, places=10)
        self.assertAlmostEqual(transform["rotation_degrees"], 180, places=5)
        completed = atom_rows(self.output / "symmetry_completion.cif")
        native = atom_rows(self.source)
        for output_chain, source_chain in (("E", "C"), ("F", "D")):
            before = [row for row in native if row["auth_asym_id"] == source_chain]
            after = [row for row in completed if row["auth_asym_id"] == output_chain]
            for source, copy in zip(before, after):
                xyz = np.array([float(source[axis]) for axis in ("Cartn_x", "Cartn_y", "Cartn_z")])
                actual = np.array([float(copy[axis]) for axis in ("Cartn_x", "Cartn_y", "Cartn_z")])
                np.testing.assert_allclose(actual, xyz @ rotation + translation, atol=1e-9)
                self.assertEqual(copy["label_seq_id"], source["label_seq_id"])
                self.assertEqual(copy["auth_seq_id"], source["auth_seq_id"])
        np.testing.assert_allclose(rotation @ rotation, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(translation @ rotation + translation, 0, atol=1e-10)
        self.assertIn("NOT EXPERIMENTAL GROUND TRUTH", self.report["reference_warning"])

    def test_output_hashes_and_pdb_exports(self):
        from scripts.prepare_2h62_inpaint import sha256
        for output in self.report["outputs"].values():
            self.assertEqual(sha256(Path(output["path"])), output["sha256"])
        self.assertEqual(len(gemmi.read_structure(str(self.output / "observed.pdb"))[0]), 4)
        self.assertEqual(len(gemmi.read_structure(str(self.output / "symmetry_completion.pdb"))[0]), 6)

    def test_downloaded_2h62_when_available(self):
        source = Path(__file__).resolve().parents[1] / "artifacts/structures/2h62-inpaint/2H62.cif"
        if not source.is_file():
            self.skipTest("Ignored experimental coordinate file not present; synthetic tests require no network")
        directory = self.root / "actual"
        report = prepare(source, directory, self.root / "actual.fasta", self.root / "actual.json")
        self.assertEqual(report["total_residues"], 682)
        self.assertAlmostEqual(report["transform"]["copy_swap_CA_RMSD_angstrom"], 1.6251805161, places=6)
        self.assertEqual(report["placement_diagnostics"]["total_modeled_interchain_heavy_atom_pairs_below_2_angstrom"], 51)
        observed = gemmi.read_structure(str(directory / "observed.cif"))
        completed = gemmi.read_structure(str(directory / "symmetry_completion.cif"))
        self.assertEqual(len(observed.connections), 16)
        self.assertEqual(len(completed.connections), 26)
        self.assertTrue(all(connection.partner1.chain_name == connection.partner2.chain_name
                            for connection in completed.connections))
        for chain, native in CHAIN_SOURCES.items():
            if chain not in "ABCD":
                continue
            before, after = reference_ca(source, native), reference_ca(directory / "observed.cif", chain)
            np.testing.assert_array_equal(before.coordinates, after.coordinates)
            self.assertEqual(before.full_positions, after.full_positions)
            self.assertEqual(before.author_residues, after.author_residues)


if __name__ == "__main__":
    unittest.main()
