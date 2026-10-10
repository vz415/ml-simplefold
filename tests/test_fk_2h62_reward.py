"""Partial-reference steering invariants; synthetic structures, no inference."""
import copy
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import gemmi
import numpy as np

from scripts.fk_2h62_reward import (DEFAULT_SETTINGS, PartialComplexReward,
                                    fixed_geometry, window_score)
from scripts.structure_clashes import Atom, SIDECHAINS


ROTATION = np.diag([-1., -1., 1.])


def write_structure(path, chain_data, sequences, cif=False):
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    for chain_name, atoms in chain_data.items():
        chain = gemmi.Chain(chain_name)
        for position, letter in enumerate(sequences[chain_name]):
            residue = gemmi.Residue()
            residue.name = gemmi.expand_one_letter(letter, gemmi.ResidueKind.AA)
            residue.seqid = gemmi.SeqId(position + 1, " ")
            residue.label_seq = position + 1
            residue.het_flag = "A"
            residue.entity_type = gemmi.EntityType.Polymer
            residue.subchain = chain_name
            for (p, name), xyz in atoms.items():
                if p != position:
                    continue
                atom = gemmi.Atom()
                atom.name, atom.element = name, gemmi.Element(name[0])
                atom.pos = gemmi.Position(*xyz)
                atom.occ = 1.
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    if not cif:
        structure.write_pdb(str(path))
        return
    for index, chain in enumerate(chain_data):
        entity = gemmi.Entity(str(index + 1))
        entity.entity_type = gemmi.EntityType.Polymer
        entity.polymer_type = gemmi.PolymerType.PeptideL
        entity.subchains = [chain]
        entity.full_sequence = [gemmi.expand_one_letter(letter, gemmi.ResidueKind.AA) for letter in sequences[chain]]
        structure.entities.append(entity)
    document = structure.make_mmcif_document()
    poly = document.sole_block().find_mmcif_category("_entity_poly.")
    seq_column = list(poly.tags).index("_entity_poly.pdbx_seq_one_letter_code")
    loop = poly.loop
    original = [row[seq_column] for row in poly]
    loop.add_columns(["_entity_poly.pdbx_seq_one_letter_code_can"], "?")
    for index, sequence in enumerate(original):
        loop[index, len(loop.tags) - 1] = sequence
    document.write_file(str(path))


def fixture(root):
    # Tests use deposited sequences from the tracked input, never an ignored
    # downloaded structure. Coordinates are synthetic and exactly C2-related.
    lines = (Path(__file__).resolve().parents[1] / "examples/2h62-inpaint.fasta").read_text().splitlines()
    sequences = {line[1]: lines[index + 1] for index, line in enumerate(lines) if line.startswith(">")}
    rng = np.random.default_rng(629)
    source = {}
    for chain, center in (("A", [20, 0, 0]), ("B", [-20, 3, 0]), ("C", [20, 3, 0])):
        seq = sequences[chain]
        atoms = {}
        for position, letter in enumerate(seq):
            residue = gemmi.expand_one_letter(letter, gemmi.ResidueKind.AA)
            names = {"N", "CA", "C", "O"} | {name for pair in SIDECHAINS[residue].split() for name in pair.split("-")}
            residue_center = np.array(center) + rng.normal(size=3) * 3
            for name in sorted(names):
                atoms[position, name] = residue_center + rng.normal(size=3) * .4
        source[chain] = atoms
    for position in (33, 34, 87, 89, 99):
        for key in [key for key in source["A"] if key[0] == position]:
            source["A"][key] = np.array([20., 0., 0.]) + rng.normal(size=3) * .2
    for name, xyz in {"CA": [20., 0., 0.], "N": [20., 0., .2],
                      "O": [19., .3, .2], "C": [19., -.4, .1]}.items():
        source["A"][50, name] = np.array(xyz)
    for position, center in ((59, [20, 3, 0]), (41, [20, -3, 0])):
        for key in [key for key in source["C"] if key[0] == position]:
            source["C"][key] = np.array(center, float) + rng.normal(size=3) * .2
    for name, xyz in {"OE1": [-20., 2.8, 0.], "NE2": [-19., 3., 0.], "CD": [-20., 4., 0.]}.items():
        source["B"][85, name] = np.array(xyz)
    source["D"] = {key: xyz @ ROTATION for key, xyz in source["A"].items()}
    write_structure(root / "observed.cif", source, sequences, cif=True)
    write_structure(root / "observed.pdb", source, sequences)
    predicted = {"A": source["A"], "B": source["D"], "C": source["B"],
                 "D": {key: xyz @ ROTATION for key, xyz in source["B"].items()},
                 "E": source["C"], "F": {key: xyz @ ROTATION for key, xyz in source["C"].items()}}
    prediction_sequences = dict(A=sequences["A"], B=sequences["A"], C=sequences["B"],
                                D=sequences["B"], E=sequences["C"], F=sequences["C"])
    return predicted, prediction_sequences


class PartialRewardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.predicted, self.sequences = fixture(self.root)
        settings = copy.deepcopy(DEFAULT_SETTINGS)
        settings["penalties"] = {key: dict(spec, weight=0.) for key, spec in settings["penalties"].items()}
        self.reward = PartialComplexReward(self.root / "observed.cif", self.root / "observed.pdb", settings)

    def score(self, coordinates=None):
        path = self.root / "prediction.pdb"
        write_structure(path, self.predicted if coordinates is None else coordinates, self.sequences)
        return self.reward.score(path)

    def test_native_motifs_get_full_credit_in_both_distinct_sites(self):
        result = self.score()
        self.assertAlmostEqual(result["reward"], 5., places=5)
        for feature in ("W60_packing", "Y42_engagement", "Q86_geometry"):
            self.assertAlmostEqual(result["features"][feature], 1., places=5)
        self.assertEqual(result["coverage"]["experimental_heavy_atom_fraction"], 1.)
        self.assertEqual(len(result["assignment_candidates"]), 8)

    def test_rigid_transform_and_identical_copy_permutations_preserve_reward(self):
        reference = self.score()
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        for swap_group in ((("A", "B"),), (("C", "D"),), (("E", "F"),),
                           (("A", "B"), ("C", "D"), ("E", "F"))):
            with self.subTest(swaps=swap_group):
                transformed = {c: {key: xyz @ rotation + [5, -7, 11] for key, xyz in atoms.items()}
                               for c, atoms in self.predicted.items()}
                for swaps in swap_group:
                    transformed[swaps[0]], transformed[swaps[1]] = transformed[swaps[1]], transformed[swaps[0]]
                result = self.score(transformed)
                self.assertAlmostEqual(reference["reward"], result["reward"], places=6)
                for feature in reference["features"]:
                    self.assertAlmostEqual(reference["features"][feature], result["features"][feature], places=6)

    def test_two_receptors_on_one_site_cannot_score_as_two_successes(self):
        duplicate = copy.deepcopy(self.predicted)
        duplicate["F"] = copy.deepcopy(duplicate["E"])
        duplicate["D"] = copy.deepcopy(duplicate["C"])
        result = self.score(duplicate)
        self.assertLess(result["features"]["W60_packing"], .01)
        self.assertLess(result["features"]["Y42_engagement"], .01)
        self.assertLess(result["features"]["Q86_geometry"], .01)
        self.assertEqual(len(set(result["assignment"]["BMPR1A_site_to_prediction"])), 2)
        self.assertEqual(len(set(result["assignment"]["ACVR2B_site_to_prediction"])), 2)

    def test_missing_hotspot_atoms_lower_score_and_keep_native_denominator(self):
        reference = self.score()
        missing = copy.deepcopy(self.predicted)
        for key in [key for key in missing["E"] if key[0] == 59 and key[1] != "CA"]:
            del missing["E"][key]
        result = self.score(missing)
        self.assertEqual(result["features"]["W60_packing"], 0.)
        self.assertEqual([row["native_contacts"] for row in reference["interfaces"]],
                         [row["native_contacts"] for row in result["interfaces"]])
        self.assertLess(result["coverage"]["experimental_heavy_atom_fraction"], 1.)

    def test_q86_requires_complementary_partners_and_orientation(self):
        modified = copy.deepcopy(self.predicted)
        # Keep both donor/acceptor distances intact but change their approach.
        modified["C"][85, "CD"] += [0, 0, 10]
        result = self.score(modified)
        self.assertLess(result["features"]["Q86_geometry"], .3)

    def test_native_long_contact_has_full_credit_without_shorter_is_better(self):
        self.assertEqual(window_score(4.305, 4.305, .75), 1.)
        self.assertLess(window_score(.5, 4.305, .75), .001)


def atom(serial, chain, number, residue, name, xyz, element=None):
    return Atom(serial, chain, number, "", residue, name, element or name[0], tuple(xyz), 1., "", ord(chain))


class FixedTopologyTests(unittest.TestCase):
    def test_compressed_disulfide_is_bond_strain_not_inferred_nonbonded_clash(self):
        disulfides = [(("A", 77), ("B", 77))]
        normal = [atom(1, "A", 78, "CYS", "SG", [0, 0, 0]),
                  atom(2, "B", 78, "CYS", "SG", [2.05, 0, 0])]
        compressed = [normal[0], replace(normal[1], xyz=(1.4, 0., 0.))]
        native = fixed_geometry(normal, disulfides, {})
        strained = fixed_geometry(compressed, disulfides, {})
        self.assertEqual(native["clash_count"], strained["clash_count"])
        self.assertEqual(strained["clash_count"], 0)
        self.assertEqual(strained["evaluated_disulfide_count"], 1)
        self.assertGreater(strained["covalent_bond_strain"], native["covalent_bond_strain"])

    def test_unknown_sulfur_pair_never_changes_topology_with_distance(self):
        atoms = [atom(1, "A", 10, "CYS", "SG", [0, 0, 0]),
                 atom(2, "B", 20, "CYS", "SG", [2.05, 0, 0])]
        result = fixed_geometry(atoms, [], {})
        self.assertEqual(result["evaluated_disulfide_count"], 0)
        self.assertEqual(result["bond_count"], 0)
        self.assertEqual(result["clash_count"], 1)

    def test_peptide_and_carbonyl_have_distinct_chemical_length_targets(self):
        atoms = [atom(1, "A", 1, "ALA", "C", [0, 0, 0]),
                 atom(2, "A", 1, "ALA", "O", [1.231, 0, 0]),
                 atom(3, "A", 2, "ALA", "N", [0, 1.329, 0])]
        result = fixed_geometry(atoms, [], {})
        targets = {row["kind"]: row["target_angstrom"] for row in result["worst_bonds"]}
        self.assertEqual(targets["ALA:C:O"], 1.231)
        self.assertEqual(targets["peptide:C:N"], 1.329)
        self.assertEqual(result["covalent_bond_strain"], 0.)


if __name__ == "__main__":
    unittest.main()
