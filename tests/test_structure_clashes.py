"""Coordinate-only clash regressions; fixtures never load a folding model."""
import tempfile
import unittest
from pathlib import Path
from scripts.structure_clashes import analyze_clashes


def atom(serial, name, residue, chain, number, xyz, element=None, occupancy=1):
    element = element or name[0]
    return (f'ATOM  {serial:5d} {name:>4s} {residue:>3s} {chain}{number:4d}    '
            f'{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occupancy:6.2f}{20.:6.2f}          {element:>2s}\n')


class TestStructureClashes(unittest.TestCase):
    def analyze(self, lines, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.pdb'
            path.write_text(''.join(lines))
            return analyze_clashes(path, **kwargs)

    def test_template_bonds_and_bond_angles_excluded(self):
        result = self.analyze([
            atom(1, 'N', 'ALA', 'A', 1, (0, 0, 0)),
            atom(2, 'CA', 'ALA', 'A', 1, (1.4, 0, 0)),
            atom(3, 'CB', 'ALA', 'A', 1, (1.4, 1.5, 0)),
        ])
        self.assertEqual(result['clash_count'], 0)

    def test_separate_chain_overlap_is_not_bonded(self):
        result = self.analyze([
            atom(1, 'CA', 'ALA', 'A', 1, (0, 0, 0)),
            atom(2, 'CA', 'ALA', 'B', 1, (1., 0, 0)),
        ])
        self.assertEqual(result['interchain_clash_count'], 1)
        self.assertEqual(result['intrachain_clash_count'], 0)
        self.assertEqual(result['CA_nonadjacent_clash_count'], 1)
        self.assertAlmostEqual(result['max_overlap_angstrom'], 2.4)

    def test_peptide_one_two_and_one_three_exclusions(self):
        result = self.analyze([
            atom(1, 'CA', 'GLY', 'A', 1, (0, 0, 0)),
            atom(2, 'C', 'GLY', 'A', 1, (1.5, 0, 0)),
            atom(3, 'N', 'GLY', 'A', 2, (2.8, 0, 0)),
            atom(4, 'CA', 'GLY', 'A', 2, (4.2, 0, 0)),
        ])
        self.assertEqual(result['clash_count'], 0)

    def test_nonbonded_atoms_in_same_residue_still_count(self):
        result = self.analyze([
            atom(1, 'O', 'LYS', 'A', 1, (0, 0, 0)),
            atom(2, 'NZ', 'LYS', 'A', 1, (.5, 0, 0)),
        ])
        self.assertEqual(result['intrachain_clash_count'], 1)

    def test_covalent_one_four_excluded_by_default(self):
        # Chemically plausible cis peptide: O–C–N–CA has a short 1–4 contact.
        # Bond lengths are 1.23, 1.34, 1.46 Å; angles approximately 119/123°.
        lines = [
            atom(1, 'O', 'ALA', 'A', 1, (0, 0, 0)),
            atom(2, 'C', 'ALA', 'A', 1, (1.23, 0, 0)),
            atom(3, 'N', 'PRO', 'A', 2, (1.878, 1.175, 0)),
            atom(4, 'CA', 'PRO', 'A', 2, (1.173, 2.454, 0)),
        ]
        self.assertEqual(self.analyze(lines)['clash_count'], 0)
        retained = self.analyze(lines, include_1_4=True)
        self.assertEqual(retained['clash_count'], 1)
        self.assertEqual(retained['worst_clashes'][0]['first'], 'A:1:ALA:O')

    def test_ter_or_number_gap_prevents_false_peptide_bond(self):
        for second, separator in [(2, 'TER\n'), (3, '')]:
            with self.subTest(second=second, separator=separator):
                result = self.analyze([
                    atom(1, 'C', 'GLY', 'A', 1, (0, 0, 0)), separator,
                    atom(2, 'N', 'GLY', 'A', second, (1.3, 0, 0)),
                ])
                self.assertEqual(result['clash_count'], 1)

    def test_inferred_disulfide_only_plausible_distance(self):
        for distance, expected in [(2.05, 0), (1.0, 1), (2.95, 1)]:
            with self.subTest(distance=distance):
                result = self.analyze([
                    atom(1, 'SG', 'CYS', 'A', 1, (0, 0, 0), 'S'),
                    atom(2, 'SG', 'CYS', 'B', 1, (distance, 0, 0), 'S'),
                ])
                self.assertEqual(result['clash_count'], expected)
                self.assertEqual(len(result['assumptions']['inferred_disulfides']), 1 - expected)

    def test_ambiguous_sulfur_cluster_not_inferred_as_bonded(self):
        result = self.analyze([
            atom(1, 'SG', 'CYS', 'A', 1, (0, 0, 0), 'S'),
            atom(2, 'SG', 'CYS', 'B', 1, (2.05, 0, 0), 'S'),
            atom(3, 'SG', 'CYS', 'C', 1, (1.025, 1.775, 0), 'S'),
        ])
        self.assertEqual(result['clash_count'], 3)
        self.assertEqual(result['assumptions']['inferred_disulfides'], [])

    def test_explicit_bond_after_model_records(self):
        result = self.analyze([
            'MODEL        1\n',
            atom(1, 'SG', 'CYS', 'A', 1, (0, 0, 0), 'S'),
            atom(2, 'SG', 'CYS', 'B', 1, (2.95, 0, 0), 'S'),
            'ENDMDL\n', 'CONECT    1    2\n',
        ])
        self.assertEqual(result['clash_count'], 0)

    def test_zero_occupancy_excluded(self):
        result = self.analyze([
            atom(1, 'CA', 'ALA', 'A', 1, (0, 0, 0)),
            atom(2, 'CA', 'ALA', 'B', 1, (0, 0, 0), occupancy=0),
        ])
        self.assertEqual(result['clash_count'], 0)
        self.assertEqual(result['heavy_atom_count'], 1)
        self.assertEqual(result['omitted_atoms']['zero_occupancy'], 1)


if __name__ == '__main__':
    unittest.main()
