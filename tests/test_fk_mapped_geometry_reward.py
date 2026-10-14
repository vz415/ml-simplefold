"""Optional heavy-atom guidance invariants; deterministic, stdlib only."""
import copy
import hashlib
import itertools
import json
import math
import unittest

import test_fk_mapped_distance_reward as fixture
from scripts.fk_mapped_distance_reward import (
    MappedDistanceReward, geometry_measurement, local_geometry_results,
    read_local_geometry_atoms)


class MappedGeometryTests(unittest.TestCase):
    reward = fixture.MappedDistanceRewardTests.reward
    joint_spec = fixture.MappedDistanceRewardTests.joint_spec

    def setUp(self):
        fixture.MappedDistanceRewardTests.setUp(self)
        fixture.THREE.update(W="TRP", L="LEU")
        self.sequences.update(E="WL", F="WL")
        self.fasta.write_text("".join(f">{c}|protein\n{s}\n" for c, s in self.sequences.items()))
        self.spec["fasta_sha256"] = hashlib.sha256(self.fasta.read_bytes()).hexdigest()
        for chain in self.spec["chains"]:
            chain["sequence_sha256"] = hashlib.sha256(self.sequences[chain["input_chain"]].encode()).hexdigest()
        self.base = self.joint_spec()
        self.spec = copy.deepcopy(self.base)
        self.spec["schema_version"] = 3
        self.atoms = {(c, p, "CA"): xyz for (c, p), xyz in self.coords.items()}
        copies = []
        for rec, lig, offset in (("E", "A", 0), ("F", "B", 40)):
            self.atoms[rec, 1, "C"] = (offset+1, 4, 2)
            self.atoms[rec, 2, "N"] = (offset+1, 6, 1)
            self.atoms[lig, 2, "OG"] = (offset, 6, 1)
            self.atoms[lig, 2, "CB"] = (offset, 5, 1)
            def endpoint(c, p, a):
                return dict(chain=c, position=p, residue=self.sequences[c][p-1], atom=a)
            n, og = endpoint(rec, 2, "N"), endpoint(lig, 2, "OG")
            def measurement(points, field):
                return {"atoms": points, field: geometry_measurement([
                    self.atoms[p["chain"], p["position"], p["atom"]] for p in points])}
            copies.append({"id": rec, "receptor_chain": rec, "interface_group": rec,
                           "evidence_labels": ["synthetic_fixture"],
                           "distance": measurement([n, og], "target_angstrom"),
                           "angles": [measurement(points, "target_degrees") for points in (
                               [endpoint(rec, 1, "C"), n, og],
                               [endpoint(rec, 2, "CA"), n, og],
                               [endpoint(lig, 2, "CB"), og, n])]})
        self.spec["local_geometry_modules"] = [{"id": "local", "kind": "distance_three_angles",
            "strength": .5, "factor_floor": .5, "missing_geometry_error": 10.,
            "distance_tolerance_angstrom": .75, "distance_scale_angstrom": 3.,
            "angle_tolerance_degrees": 20., "angle_scale_degrees": 30.,
            "evidence_labels": ["synthetic_fixture"], "copies": copies}]

    def write_atoms(self, atoms=None, extras=(), reverse=False):
        records = [fixture.atom_line(i, c, p, self.sequences[c][p-1], xyz, name)
                   for i, ((c, p, name), xyz) in enumerate((self.atoms if atoms is None else atoms).items(), 1)]
        records.extend(extras)
        if reverse:
            records.reverse()
        self.pdb.write_text("".join(records)+"END\n")
        return self.pdb

    def test_optimum_has_five_atoms_four_measurements_and_two_copies(self):
        result = self.reward().score(self.write_atoms())
        self.assertEqual(result["reward"], 1.)
        module = result["local_geometry"][0]
        self.assertEqual(module["maximum_copy_error"], 0.)
        for c in module["copies"]:
            self.assertEqual((c["required_atoms"], c["present_atoms"]), (5, 5))
            self.assertEqual(len(c["measurements"]), 4)
        json.dumps(result, allow_nan=False)

    def test_distance_and_angle_have_effect_without_ca_change(self):
        for key, xyz in ((('E', 2, 'N'), (20, 6, 1)), (('E', 1, 'C'), (-1, 6, 1)),
                         (('A', 2, 'CB'), (2, 6, 1))):
            with self.subTest(key=key):
                atoms = dict(self.atoms); atoms[key] = xyz
                result = self.reward().score(self.write_atoms(atoms))
                self.assertEqual(result["ca_reward_under_selected_assignment"], 1.)
                self.assertLess(result["reward"], 1.)
                errors = [c["geometry_error"] for c in result["local_geometry"][0]["copies"]]
                self.assertEqual(min(errors), 0.)
                self.assertGreater(max(errors), 0.)
                self.assertEqual(result["local_geometry"][0]["maximum_copy_error"], max(errors))
                if key[2] != "N":
                    self.assertEqual(result["local_geometry"][0]["copies"][0]["measurements"][0]["error"], 0.)

    def test_zero_strength_exactly_matches_v2_even_missing_or_wrong_heavy_atoms(self):
        off = copy.deepcopy(self.spec); off["local_geometry_modules"][0]["strength"] = 0
        base_reward, off_reward = self.reward(self.base), self.reward(off)
        cases = [self.atoms, {k: v for k, v in self.atoms.items() if k[2] == "CA"},
                 {k: (v[0]+4, v[1], v[2]) if k[0] == "F" else v for k,v in self.atoms.items()}]
        for atoms in cases:
            p = self.write_atoms(atoms)
            base, actual = base_reward.score(p), off_reward.score(p)
            for key in base:
                if key != "limited_scope":
                    self.assertEqual(actual[key], base[key], key)
            self.assertEqual(actual["local_geometry_factor"], 1.)

    def test_missing_zero_occupancy_and_degenerate_geometry_stay_finite(self):
        keys = [('E',1,'C'), ('E',2,'N'), ('E',2,'CA'), ('A',2,'CB'), ('A',2,'OG')]
        for key, mode in itertools.product(keys, ("missing", "zero_occupancy")):
            with self.subTest(key=key, mode=mode):
                atoms = dict(self.atoms); xyz = atoms.pop(key)
                c,p,name = key
                extra = [fixture.atom_line(999,c,p,self.sequences[c][p-1],xyz,name,occupancy=0)] if mode == "zero_occupancy" else []
                result = self.reward().score(self.write_atoms(atoms,extra))
                module = result["local_geometry"][0]
                self.assertGreaterEqual(module["maximum_copy_error"], 10)
                self.assertTrue(any(c["present_atoms"] == 4 for c in module["copies"]))
                json.dumps(result, allow_nan=False)
        atoms = dict(self.atoms); atoms['E',1,'C'] = atoms['E',2,'N']
        result = self.reward().score(self.write_atoms(atoms))
        self.assertEqual(result["local_geometry"][0]["maximum_copy_error"], 10)
        self.assertEqual(result["local_geometry"][0]["copies"][0]["measurements"][1]["status"], "degenerate_geometry")

    def test_global_transform_and_all_copy_swaps(self):
        reward = self.reward()
        for swap in itertools.product((False,True), repeat=3):
            mapping = dict(zip("ABCDEF", "ABCDEF"))
            for enabled, (a,b) in zip(swap, [('A','B'),('C','D'),('E','F')]):
                if enabled:
                    mapping[a],mapping[b] = b,a
            atoms = {(mapping[c],p,n): (-y+7,x-8,z+2) for (c,p,n),(x,y,z) in self.atoms.items()}
            result = reward.score(self.write_atoms(atoms))
            self.assertAlmostEqual(result["reward"], 1.)
            self.assertEqual(len(set(result["assignment"]["role_to_input_chain"].values())),6)

    def test_geometry_cannot_remap_independently_of_ca(self):
        # Only geometry-bearing heavy atoms are swapped. CA optimum and local
        # geometry optimum now require incompatible receptor permutations.
        atoms = dict(self.atoms)
        for name,pos in [('C',1),('N',2)]:
            atoms['E',pos,name],atoms['F',pos,name] = atoms['F',pos,name],atoms['E',pos,name]
        result = self.reward().score(self.write_atoms(atoms))
        self.assertLess(result["reward"], .9)
        self.assertGreater(result["local_geometry"][0]["maximum_copy_error"], 1.)
        assignment = result["assignment"]["role_to_input_chain"]
        for c in result["local_geometry"][0]["copies"]:
            for m in c["measurements"]:
                for source, assigned in zip(m["atoms"],m["assigned_atom_keys"]):
                    self.assertEqual(assigned[0],assignment[source['chain']])

    def test_conformer_selection_never_mixes_alternate_atoms(self):
        atoms = dict(self.atoms); del atoms['E',2,'N']; del atoms['E',2,'CA']
        # Legacy CA selects A; N exists only in incompatible B and must be absent.
        extras = [fixture.atom_line(997,'E',2,'L',self.atoms['E',2,'CA'],'CA',altloc='A',occupancy=.7),
                  fixture.atom_line(998,'E',2,'L',self.atoms['E',2,'CA'],'CA',altloc='B',occupancy=.3),
                  fixture.atom_line(999,'E',2,'L',self.atoms['E',2,'N'],'N',altloc='B',occupancy=.9)]
        results = [self.reward().score(self.write_atoms(atoms,extras,reverse=r)) for r in (False,True)]
        self.assertEqual(results[0],results[1])
        self.assertGreaterEqual(results[0]['local_geometry'][0]['maximum_copy_error'],10)
        choice = next(c for c in results[0]['local_geometry_conformers'] if c['input_chain']=='E' and c['position']==2)
        self.assertEqual(choice['selected_altloc'],'A')

    def test_shared_blank_ca_is_identical_in_ca_objective_and_geometry(self):
        extra=[fixture.atom_line(999,'E',2,'L',(60,60,60),'CA',altloc='A',occupancy=.5)]
        result=self.reward().score(self.write_atoms(extras=extra))
        self.assertEqual(result['reward'],1.)
        atoms,_=read_local_geometry_atoms(self.pdb,self.sequences,dict(zip('ABCDEF','ABCDEF')),
                                           self.spec['local_geometry_modules'])
        self.assertEqual(atoms['E',2,'CA'],self.atoms['E',2,'CA'])

    def test_invalid_geometry_specs_fail_closed(self):
        bad = []
        for field in ('strength','factor_floor','missing_geometry_error','distance_scale_angstrom','angle_scale_degrees',
                      'angle_tolerance_degrees','distance_tolerance_angstrom'):
            for value in (float('nan'),float('inf'),-1,True):
                s = copy.deepcopy(self.spec); s['local_geometry_modules'][0][field] = value; bad.append(s)
        for field,value in [('strength',0),('factor_floor',0)]:
            s=copy.deepcopy(self.spec);s['local_geometry_modules'][0][field]=value; self.reward(s)
        for path,value in [(['copies',0,'angles'],[]),(['copies',0,'distance','target_angstrom'],float('nan')),
                           (['copies',0,'angles',0,'target_degrees'],180),(['copies',0,'interface_group'],'C'),
                           (['copies',0,'distance','atoms',0,'atom'],'OG'),
                           (['copies',0,'distance','atoms',0,'residue'],'A'),
                           (['copies',0,'distance','atoms',0,'position'],0)]:
            s=copy.deepcopy(self.spec); dest=s['local_geometry_modules'][0]
            for key in path[:-1]: dest=dest[key]
            dest[path[-1]]=value;bad.append(s)
        s=copy.deepcopy(self.spec);s['local_geometry_modules'][0]['copies'].pop();bad.append(s)
        s=copy.deepcopy(self.spec);s['local_geometry_modules'][0]['copies'][0]['angles'][1]=copy.deepcopy(s['local_geometry_modules'][0]['copies'][0]['angles'][0]);bad.append(s)
        for s in bad:
            with self.subTest(spec=s),self.assertRaises(ValueError): self.reward(s)
        s=copy.deepcopy(self.spec);s['schema_version']=2
        with self.assertRaises(ValueError):self.reward(s)

    def test_nonfinite_heavy_atoms_and_extreme_geometry_do_not_return_nan(self):
        for value in (float('nan'),float('inf')):
            atoms=dict(self.atoms);atoms['E',2,'N']=(value,0,0)
            with self.assertRaises(ValueError): self.reward().score(self.write_atoms(atoms))
        self.assertTrue(math.isfinite(geometry_measurement([(1e300,0,0),(0,0,0),(0,1e300,0)])))
        with self.assertRaises(ValueError): geometry_measurement([(1e308,0,0),(-1e308,0,0)])


if __name__ == '__main__':
    unittest.main()
