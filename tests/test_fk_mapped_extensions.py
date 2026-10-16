"""Optional robust interfaces and fixed-topology sterics; no sampling."""
import copy
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.fk_mapped_distance_reward import (MappedDistanceReward, _HeavyAtomSafeguard,
                                              _HEAVY_RADII, _SIDECHAIN_BONDS)
from tests import test_fk_mapped_distance_reward as fixture


def guard_config(disulfides=None, elements=None):
    return {"kind": "interchain_overlap", "elements": elements or ["C", "N", "O"],
            "overlap_threshold_angstrom": 1., "overlap_scale_angstrom": .5,
            "pair_scale": 20., "strength": 1., "factor_floor": .8,
            "missing_heavy_atom_policy": "factor_floor", "native_disulfides": disulfides or [],
            "evidence_labels": ["synthetic_test"]}


def atom_line(serial, c, pos, aa, name, xyz, alt=" ", occupancy=1.):
    reverse = {"A": "ALA", "S": "SER", "N": "ASN", "T": "THR", "G": "GLY", "K": "LYS", "C": "CYS"}
    return (f"ATOM  {serial:5d} {name:^4s}{alt}{reverse[aa]:>3s} {c}{pos:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occupancy:6.2f}{0.:6.2f}          {name[0]:>2s}  \n")


class InterfaceExtensionTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.MappedDistanceRewardTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def extended(self):
        spec = self.f.joint_spec()
        for p in spec["pairs"]:
            if p["left"]["chain"] in "CD":
                p["distance_feature"] = "cauchy"
        for group in spec["interface_groups"][:2]:
            group["required_subgroups"] = [
                {"id": "face", "pair_ids": group["pair_ids"][:2]},
                {"id": "bridge", "pair_ids": group["pair_ids"][2:]}]
        return spec

    def test_cauchy_long_tail_and_half_maximum(self):
        spec = self.f.joint_spec()
        spec["equivalent_chain_groups"] = []
        pair = spec["pairs"][0]
        pair["distance_feature"] = "cauchy"
        pair["distance_angstrom"] += pair["sigma_angstrom"]
        r = self.f.reward(spec).score(self.f.write_pdb())
        self.assertAlmostEqual(r["pairs"][0]["feature"], .5)
        pair["distance_angstrom"] += 20*pair["sigma_angstrom"]
        r = self.f.reward(spec).score(self.f.write_pdb())
        self.assertAlmostEqual(r["pairs"][0]["feature"], 1/(1+21**2))
        self.assertGreater(r["pairs"][0]["feature"], math.exp(-.5*21**2))

    def test_missing_bridge_cannot_be_hidden_and_subgroups_share_assignment(self):
        spec = self.extended()
        spec["equivalent_chain_groups"] = []
        coords = dict(self.f.coords)
        del coords["C", 1]
        result = self.f.reward(spec).score(self.f.write_pdb(coords))
        interface = result["interfaces"][0]
        self.assertEqual(interface["score"], 0.)
        self.assertGreater(interface["unpartitioned_weighted_mean"], 0.)
        self.assertEqual(interface["subgroup_scores"][1]["score"], 0.)
        self.assertEqual(interface["subgroup_scores"][1]["total_pairs"], 1)
        spec = self.extended()
        base = self.f.reward(spec).score(self.f.write_pdb())
        for swap in itertools.product((False, True), repeat=3):
            mapping = dict(zip("ABCDEF", "ABCDEF"))
            for flag, group in zip(swap, ("AB", "CD", "EF")):
                if flag:
                    mapping.update(zip(group, reversed(group)))
            coords = {(mapping[c], p): (-y+3, x+4, z-9) for (c, p), (x, y, z) in self.f.coords.items()}
            changed = self.f.reward(spec).score(self.f.write_pdb(coords))
            self.assertEqual(changed["reward"], base["reward"])
            self.assertEqual(len(set(changed["assignment"]["role_to_input_chain"].values())), 6)

    def test_subgroup_partition_and_feature_typos_rejected(self):
        for change in ("missing", "duplicate", "unknown", "empty"):
            spec = self.extended()
            sub = spec["interface_groups"][0]["required_subgroups"]
            if change == "missing": sub[0]["pair_ids"].pop()
            if change == "duplicate": sub[1]["pair_ids"] = sub[0]["pair_ids"][:]
            if change == "unknown": sub[1]["pair_ids"] = ["unknown"]
            if change == "empty": sub.clear()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.f.reward(spec)
        spec = self.extended()
        spec["pairs"][0]["distance_feature"] = "cauchi"
        with self.assertRaisesRegex(ValueError, "distance_feature"):
            self.f.reward(spec)

    def with_internal_spans(self):
        spec = self.f.joint_spec()
        for group in spec["interface_groups"][:2]:
            receptor = group["receptor_chain"]
            span = {"id": f"{receptor}_internal_span", "scope": "intra_receptor",
                    "left": {"chain": receptor, "position": 1, "residue": "N", "atom": "CA"},
                    "right": {"chain": receptor, "position": 2, "residue": "T", "atom": "CA"},
                    "distance_angstrom": 6., "sigma_angstrom": .5, "weight": 1.,
                    "evidence_labels": ["synthetic_shape_span"]}
            spec["pairs"].append(span)
            group["required_subgroups"] = [{"id": "docking", "pair_ids": list(group["pair_ids"])},
                                            {"id": "shape", "pair_ids": [span["id"]]}]
            group["pair_ids"].append(span["id"])
        return spec

    def test_internal_shape_is_an_explicit_required_copy_cap(self):
        spec = self.with_internal_spans()
        result = self.f.reward(spec).score(self.f.write_pdb())
        self.assertEqual(result["reward"], 1.)
        spans = [pair for pair in result["pairs"] if "scope" in pair]
        self.assertEqual({p["id"] for p in spans}, {"C_internal_span", "D_internal_span"})
        self.assertTrue(all(p["scope"] == "intra_receptor" for p in spans))
        coords = dict(self.f.coords)
        coords["C", 2] = (0, 8, 4)
        changed = self.f.reward(spec).score(self.f.write_pdb(coords))
        interface = next(g for g in changed["interfaces"] if g["id"] == "C")
        self.assertAlmostEqual(interface["score"], math.exp(-8.))
        self.assertGreater(interface["subgroup_scores"][0]["score"], .8)
        self.assertGreater(interface["unpartitioned_weighted_mean"], interface["score"])

    def test_internal_missing_atom_keeps_denominator_and_zero_copy_cap(self):
        spec = self.with_internal_spans()
        spec["equivalent_chain_groups"] = []
        coords = dict(self.f.coords)
        del coords["C", 2]
        result = self.f.reward(spec).score(self.f.write_pdb(coords))
        group = next(g for g in result["interfaces"] if g["id"] == "C")
        self.assertEqual(group["score"], 0.)
        self.assertEqual(group["subgroup_scores"][1]["total_pairs"], 1)
        self.assertEqual(group["subgroup_scores"][1]["present_pairs"], 0)
        self.assertGreater(group["subgroup_scores"][0]["score"], 0.)

    def test_internal_spans_follow_same_eight_bijections_and_rigid_motion(self):
        spec = self.with_internal_spans()
        base_coords = dict(self.f.coords)
        base_coords["C", 2] = (0, 8, 4)
        reward = self.f.reward(spec)
        base = reward.score(self.f.write_pdb(base_coords))
        for swap in itertools.product((False, True), repeat=3):
            mapping = dict(zip("ABCDEF", "ABCDEF"))
            for flag, group in zip(swap, ("AB", "CD", "EF")):
                if flag:
                    mapping.update(zip(group, reversed(group)))
            coords = {(mapping[c], p): (-y+3, x+4, z-9) for (c, p), (x, y, z) in base_coords.items()}
            result = reward.score(self.f.write_pdb(coords))
            self.assertAlmostEqual(result["reward"], base["reward"])
            # A simultaneous ligand/receptor swap can exchange otherwise tied
            # logical copies; the invariant multiset and coherent distances matter.
            self.assertEqual(sorted(result["features"].values()), sorted(base["features"].values()))
            assignment = result["assignment"]["role_to_input_chain"]
            self.assertEqual(len(set(assignment.values())), 6)
            rows = {p["id"]: p for p in result["pairs"]}
            for pair in spec["pairs"]:
                endpoints = [(assignment[pair[end]["chain"]], pair[end]["position"]) for end in ("left", "right")]
                self.assertAlmostEqual(rows[pair["id"]]["distance_angstrom"], math.dist(*(coords[key] for key in endpoints)))

    def test_internal_scope_validation_rejects_wrong_chain_or_cross_mislabel(self):
        for change in ("omit", "cross", "other_receptor", "between_receptors"):
            spec = self.with_internal_spans()
            pair = next(p for p in spec["pairs"] if p["id"] == "C_internal_span")
            if change == "omit":
                del pair["scope"]
            if change == "cross":
                pair["right"].update(chain="A", residue="S")
            if change == "other_receptor":
                pair["left"]["chain"] = pair["right"]["chain"] = "D"
            if change == "between_receptors":
                pair["right"]["chain"] = "D"
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "receptor"):
                self.f.reward(spec)

    def test_invalid_scope_and_scope_outside_interface_groups_are_rejected(self):
        for value in (None, "", "intra", "cross_interface", 1, {}, []):
            spec = self.with_internal_spans()
            spec["pairs"][-1]["scope"] = value
            with self.subTest(scope=value), self.assertRaisesRegex(ValueError, "Pair scope"):
                self.f.reward(spec)
        spec = self.f.joint_spec()
        spec["integrity_pairs"][0]["scope"] = "intra_receptor"
        with self.assertRaisesRegex(ValueError, "interface-group pair"):
            self.f.reward(spec)
        spec = copy.deepcopy(self.f.spec)
        spec["pairs"][0]["scope"] = "intra_receptor"
        with self.assertRaisesRegex(ValueError, "version-2/3"):
            self.f.reward(spec)

    def complete_atoms(self):
        atoms = {}
        for (c, p), xyz in self.f.coords.items():
            aa = self.f.sequences[c][p-1]
            names = {"N", "CA", "C", "O"} | set(_SIDECHAIN_BONDS[aa].replace("-", " ").split())
            for i, name in enumerate(sorted(names)):
                atoms[c, p, name] = xyz if name == "CA" else (xyz[0]+i*.4, xyz[1]+.2, xyz[2]+.3)
        return atoms

    def write_heavy(self, atoms):
        self.f.pdb.write_text("".join(atom_line(i, c, p, self.f.sequences[c][p-1], name, xyz)
            for i, ((c, p, name), xyz) in enumerate(sorted(atoms.items()), 1))+"END\n")

    def test_missing_heavy_is_worst_factor_and_zero_strength_is_exactly_neutral(self):
        spec = self.f.joint_spec()
        spec["heavy_atom_safeguard"] = guard_config()
        atoms = self.complete_atoms()
        self.write_heavy(atoms)
        whole = self.f.reward(spec).score(self.f.pdb)
        self.assertEqual(whole["heavy_atom_safeguard"]["status"], "assessed_geometric_overlaps")
        del atoms["C", 1, "CB"]
        self.write_heavy(atoms)
        missing = self.f.reward(spec).score(self.f.pdb)
        self.assertEqual(missing["heavy_atom_safeguard"]["factor"], .8)
        self.assertLessEqual(missing["reward"], whole["reward"])
        spec["heavy_atom_safeguard"]["strength"] = 0.
        neutral = self.f.reward(spec).score(self.f.pdb)
        old = self.f.reward(self.f.joint_spec()).score(self.f.pdb)
        self.assertEqual(neutral["reward"], old["reward"])
        self.assertEqual(neutral["assignment"], old["assignment"])

    def test_nonfinite_heavy_and_invalid_guard_settings_fail_closed(self):
        spec = self.f.joint_spec()
        spec["heavy_atom_safeguard"] = guard_config()
        atoms = self.complete_atoms()
        for value in (float("nan"), float("inf")):
            atoms["C", 1, "CB"] = (value, 0, 0)
            self.write_heavy(atoms)
            with self.assertRaises(ValueError):
                self.f.reward(spec).score(self.f.pdb)
        for key, value in (("pair_scale", 0), ("strength", float("nan")), ("factor_floor", 1.1),
                           ("missing_heavy_atom_policy", "ignore"), ("native_disulfides", None)):
            invalid = copy.deepcopy(spec)
            invalid["heavy_atom_safeguard"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.f.reward(invalid)


class FixedTopologyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)/"model.pdb"
        endpoint = lambda c: {"chain": c, "position": 1, "residue": "C", "atom": "SG"}
        self.config = guard_config([{"left": endpoint("A"), "right": endpoint("B"),
                                    "evidence_labels": ["annotated_fixture"]}], ["C", "N", "O", "S"])
        self.guard = _HeavyAtomSafeguard(self.config, {"A": "C", "B": "C"}, {"A": "A", "B": "B"},
                                       [{"A": "A", "B": "B"}, {"A": "B", "B": "A"}])

    def write(self, atoms, conect=""):
        self.path.write_text("".join(atom_line(i, c, p, "C", name, xyz)
            for i, ((c, p, name), xyz) in enumerate(sorted(atoms.items()), 1))+conect+"END\n")

    def test_native_cross_chain_1_2_1_3_1_4_exclusions_and_grid_match_bruteforce(self):
        for a, b in (("SG", "SG"), ("CB", "SG"), ("CA", "SG"), ("CB", "CB")):
            self.assertIn(frozenset((("A", 1, a), ("B", 1, b))), self.guard.excluded)
        self.assertNotIn(frozenset((("A", 1, "N"), ("B", 1, "SG"))), self.guard.excluded)
        atoms = {k: (i*.23, (i%3)*.3, (i%2)*.4) for i, k in enumerate(sorted(self.guard.required))}
        self.write(atoms)
        score = self.guard.score(self.path)
        expected = []
        for (a, xyz), (b, other) in itertools.combinations(atoms.items(), 2):
            if a[0] == b[0] or frozenset((a, b)) in self.guard.excluded:
                continue
            excess = max(_HEAVY_RADII[a[2][0]]+_HEAVY_RADII[b[2][0]]-math.dist(xyz, other)-1., 0.)
            if excess:
                expected.append(-math.expm1(-.5*(excess/.5)**2))
        self.assertEqual(score["overlap_pair_count"], len(expected))
        self.assertAlmostEqual(score["severity_sum"], math.fsum(expected), places=12)
        self.assertGreater(score["excluded_topology_candidate_pairs"], 0)
        # Arbitrary PDB bonds never remove otherwise counted nonbonded pairs.
        self.write(atoms, "CONECT    1    7\n")
        self.assertEqual(score, self.guard.score(self.path))
        # The exclusion set stays fixed even with an implausibly long S–S bond.
        original_exclusions = copy.deepcopy(self.guard.excluded)
        atoms["B", 1, "SG"] = (100., 100., 100.)
        self.write(atoms)
        self.guard.score(self.path)
        self.assertEqual(self.guard.excluded, original_exclusions)

    def test_topology_missing_duplicate_or_noncysteine_endpoints_rejected(self):
        for change in ("missing", "duplicate", "wrong"):
            config = copy.deepcopy(self.config)
            if change == "missing": config["native_disulfides"] = []
            if change == "duplicate": config["native_disulfides"] *= 2
            if change == "wrong": config["native_disulfides"][0]["left"]["position"] = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                _HeavyAtomSafeguard(config, {"A": "C", "B": "C"}, {"A": "A", "B": "B"}, [{"A": "A", "B": "B"}])


if __name__ == "__main__":
    unittest.main()
