"""Geometry evidence masks and partial-reference reporting; no inference."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bmp90_evaluate as ev
import test_fk_mapped_geometry_reward as fixture


class GeometryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.MappedGeometryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        # Three noncollinear CA positions per chain for the reference fit.
        for c in f.sequences:
            f.sequences[c] += "A"
            x,y,z = f.atoms[c,2,"CA"]
            f.atoms[c,3,"CA"] = (x+2,y,z+1)
        self.chains = [{"input_chain":c,"sequence":s,"gene":("ligand" if c in "AB" else "typeI" if c in "CD" else "typeII")}
                       for c,s in f.sequences.items()]
        for row in f.spec["chains"]:
            row["length"] = 3
            row["sequence_sha256"] = hashlib.sha256(f.sequences[row["input_chain"]].encode()).hexdigest()
        f.write_atoms()
        self.reference,_ = ev.read_atoms(f.pdb)

    def evaluate(self, reference=None):
        f=self.fixture
        return ev.evaluate_model(f.pdb, self.reference if reference is None else reference,
                                 self.chains, dict(zip("ABCDEF","ABCDEF")), ["A","B"],f.spec)

    def test_all_atom_endpoints_excluded_across_equivalent_copies(self):
        spec=copy.deepcopy(self.fixture.spec)
        # Remove every CA endpoint except unrelated position3, isolating the
        # geometry exclusion logic without requiring a scoring-valid pair list.
        spec['pairs']=[{'id':'ca','left':{'chain':'C','position':3,'residue':'A','atom':'CA'},
                       'right':{'chain':'A','position':3,'residue':'A','atom':'CA'}}]
        spec['integrity_pairs']=[]
        excluded,_=ev.guidance_info(spec,self.chains)
        for c,p in [('A',2),('B',2),('E',1),('E',2),('F',1),('F',2)]:
            self.assertIn((c,p),excluded)
        self.assertNotIn(('E',3),excluded)

    def test_end_to_end_geometry_and_uncovered_copy_null(self):
        full=self.evaluate()
        self.assertEqual(full['local_geometry_under_reference_assignment'][0]['maximum_copy_error'],0)
        partial={k:a for k,a in self.reference.items() if k[0] not in 'DF'}
        result=self.evaluate(partial)
        module=result['local_geometry_under_reference_assignment'][0]
        self.assertEqual(module['copies'][0]['geometry_error'],0)
        self.assertIsNone(module['copies'][1]['geometry_error'])
        self.assertIsNone(module['factor'])
        self.assertEqual(module['copies'][1]['status'],'unassessed_no_reference_assignment')
        self.assertEqual(result['reference_uncovered_chains'],['D','F'])
        self.assertEqual(result['native_contact_recovery_summary'],{'mean':None,'weakest':None})
        for interface in result['interfaces']:
            if interface['reference_receptor_chain'] in 'DF':
                self.assertIsNone(interface['all_native_contacts'])

    def test_missing_reference_atom_cannot_be_assessed_from_model_atom(self):
        reference=dict(self.reference);del reference['A',2,'OG']
        module=self.evaluate(reference)['local_geometry_under_reference_assignment'][0]
        copy=next(c for c in module['copies'] if c['receptor_chain']=='E')
        self.assertIsNone(copy['geometry_error'])
        self.assertEqual(copy['status'],'unassessed_missing_reference_atoms')
        self.assertTrue(all(m['value'] is None for m in copy['measurements']))

    def test_partial_reference_as_model_skips_absent_receptor_copy_assignments(self):
        f = self.fixture
        atoms = {k: xyz for k, xyz in f.atoms.items() if k[0] not in 'DF'}
        f.write_atoms(atoms)
        reference, _ = ev.read_atoms(f.pdb)
        result = ev.evaluate_model(f.pdb, reference, self.chains,
                                   dict(zip('ABCDEF', 'ABCDEF')), ['A', 'B'], f.spec,
                                   allow_partial=True)
        self.assertEqual(len(result['assignment_alternatives']), 2)
        for candidate in result['assignment_alternatives']:
            assignment = candidate['reference_to_model_input_chain']
            self.assertEqual((assignment['C'], assignment['E']), ('C', 'E'))
        self.assertLess(result['coherent_assignment']['objective_A_squared'], 1e-20)
        self.assertEqual(result['reference_uncovered_chains'], ['D', 'F'])
        self.assertEqual(result['native_contact_recovery_summary'], {'mean': None, 'weakest': None})
        module = result['local_geometry_under_reference_assignment'][0]
        self.assertEqual(module['copies'][0]['geometry_error'], 0)
        self.assertIsNone(module['copies'][1]['geometry_error'])
        with self.assertRaisesRegex(ValueError, 'missing .* expected CA'):
            ev.evaluate_model(f.pdb, reference, self.chains,
                              dict(zip('ABCDEF', 'ABCDEF')), ['A', 'B'], f.spec)

    def test_no_covered_copy_match_fails_after_considering_all_assignments(self):
        partial = {k: a for k, a in self.reference.items() if k[0] not in 'DF'}
        no_type_two = {k: a for k, a in partial.items() if k[0] != 'E'}
        with self.assertRaisesRegex(ValueError, 'No valid coherent assignment'):
            ev.coherent_assignment(partial, no_type_two, self.chains, ['A', 'B'])
        too_short = {k: a for k, a in partial.items() if k[0] != 'E' or k[1] < 3}
        with self.assertRaisesRegex(ValueError, 'No valid coherent assignment'):
            ev.coherent_assignment(partial, too_short, self.chains, ['A', 'B'])

    def test_explicit_homologous_label_and_preserved_species_label(self):
        value=ev.reference_comparison_scope({'reference_surrogate_kind':'homologous_ligand'},True,[])
        self.assertEqual(value[:2],(True,'homologous_surrogate_agreement'))
        self.assertIn('not exact target-ligand accuracy',value[2])
        self.assertEqual(ev.reference_comparison_scope({'reference_kind':'species surrogate'},True,[])[1],
                         'species_surrogate_agreement')
        self.assertEqual(ev.reference_comparison_scope({},True,[])[1],'partial_reference_recovery')
        self.assertEqual(ev.reference_comparison_scope({},False,[])[1],'reference_recovery')


if __name__=='__main__':
    unittest.main()
