"""Mapped reward invariants on synthetic PDBs; no sampling or dependencies."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.fk_mapped_distance_reward import MappedDistanceReward


THREE = {"A": "ALA", "S": "SER", "N": "ASN", "T": "THR", "G": "GLY", "K": "LYS"}


def atom_line(serial, chain, position, residue, xyz, atom="CA", altloc=" ", occupancy=1):
    x, y, z = xyz
    return (f"ATOM  {serial:5d} {atom:^4s}{altloc}{THREE[residue]:>3s} {chain}{position:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{0:6.2f}          C  \n")


class MappedDistanceRewardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fasta = self.root / "input.fasta"
        self.spec_path = self.root / "spec.json"
        self.pdb = self.root / "input.pdb"
        self.sequences = {"A": "AS", "B": "AS", "C": "NT", "D": "NT", "E": "GK", "F": "GK"}
        self.fasta.write_text("".join(f">{c}|protein\n{s}\n" for c, s in self.sequences.items()))
        self.coords = {}
        for left, right, base in (("A", "C", 0), ("B", "D", 40)):
            self.coords[left, 1] = (base, 0, 0)
            self.coords[left, 2] = (base, 6, 0)
            self.coords[right, 1] = (base, 0, 4)
            self.coords[right, 2] = (base, 6, 4)
        for chain, base in (("E", 10), ("F", 50)):
            self.coords[chain, 1] = (base, 0, 0)
            self.coords[chain, 2] = (base, 6, 0)
        self.spec = {
            "schema_version": 1, "active": True, "mapping_status": "verified",
            "target_id": "synthetic", "scope": "synthetic distance reconstruction test",
            "fasta_sha256": hashlib.sha256(self.fasta.read_bytes()).hexdigest(),
            "evidence_labels": ["synthetic_fixture"],
            "chains": [{"input_chain": c, "saved_chain": c, "length": len(s),
                        "sequence_sha256": hashlib.sha256(s.encode()).hexdigest()}
                       for c, s in self.sequences.items()],
            "equivalent_chain_groups": [["A", "B"], ["C", "D"], ["E", "F"]],
            "pairs": []}
        for receptor, ligand in (("C", "A"), ("D", "B")):
            for r, l in ((1, 1), (2, 2), (1, 2)):
                self.spec["pairs"].append({
                    "id": f"{receptor}{r}_{ligand}{l}",
                    "left": {"chain": receptor, "position": r,
                             "residue": self.sequences[receptor][r - 1], "atom": "CA"},
                    "right": {"chain": ligand, "position": l,
                              "residue": self.sequences[ligand][l - 1], "atom": "CA"},
                    "distance_angstrom": math.dist(self.coords[receptor, r], self.coords[ligand, l]),
                    "sigma_angstrom": .5, "weight": 1., "evidence_labels": ["synthetic_fixture"]})

    def reward(self, spec=None):
        self.spec_path.write_text(json.dumps(self.spec if spec is None else spec))
        return MappedDistanceReward(self.fasta, self.spec_path)

    def write_pdb(self, coords=None, saved=None, modeled=False):
        coords = self.coords if coords is None else coords
        saved = saved or {c: c for c in self.sequences}
        lines = ["MODEL        1\n"] if modeled else []
        for serial, ((chain, pos), xyz) in enumerate(coords.items(), 1):
            lines.append(atom_line(serial, saved[chain], pos, self.sequences[chain][pos - 1], xyz))
        if modeled:
            lines.append("ENDMDL\n")
        lines.append("END\n")
        self.pdb.write_text("".join(lines))
        return self.pdb

    def test_reference_optimum_and_finite_json(self):
        reward = self.reward()
        result = reward.score(self.write_pdb(modeled=True))
        self.assertEqual(result["reward"], 1.)
        self.assertEqual(result["coverage"]["present_pairs"], 6)
        self.assertEqual(result["assignments_evaluated"], 8)
        self.assertEqual(reward.metadata["positive_maximum"], 1.)
        json.dumps(result, allow_nan=False)
        json.dumps(reward.metadata, allow_nan=False)

    def test_displaced_interface_reduces_score(self):
        coords = {key: (x, y, z + 30) if key[0] == "D" else (x, y, z)
                  for key, (x, y, z) in self.coords.items()}
        result = self.reward().score(self.write_pdb(coords))
        self.assertAlmostEqual(result["reward"], .5)

    def test_copy_permutation_and_rigid_motion_use_one_bijection(self):
        swaps = {"A": "B", "B": "A", "C": "D", "D": "C", "E": "F", "F": "E"}
        for changed in ({"A", "B"}, {"C", "D"}, {"E", "F"}, set(swaps)):
            with self.subTest(changed=changed):
                coords = {(swaps[c] if c in changed else c, p): (-y + 3, x + 4, z - 9)
                          for (c, p), (x, y, z) in self.coords.items()}
                result = self.reward().score(self.write_pdb(coords))
                self.assertEqual(result["reward"], 1.)
                assignment = result["assignment"]["role_to_input_chain"]
                self.assertEqual(len(set(assignment.values())), 6)

    def test_duplicated_receptor_cannot_take_both_sites(self):
        coords = dict(self.coords)
        for position in (1, 2):
            coords["D", position] = coords["C", position]
        result = self.reward().score(self.write_pdb(coords))
        self.assertAlmostEqual(result["reward"], .5)
        assignment = result["assignment"]["role_to_input_chain"]
        self.assertNotEqual(assignment["C"], assignment["D"])
        self.assertEqual(result["diagnostics"]["interchain_CA_pairs_below_2_angstrom"], 2)

    def test_assignment_cannot_change_independently_for_each_pair(self):
        # Each receptor has one residue at each opposing site. Per-pair greedy
        # matching gives both short contacts full credit; no coherent map can.
        coords = dict(self.coords)
        coords["C", 2], coords["D", 2] = coords["D", 2], coords["C", 2]
        result = self.reward().score(self.write_pdb(coords))
        self.assertLess(result["reward"], .7)

    def test_missing_atoms_keep_original_denominator_and_report_coverage(self):
        coords = dict(self.coords)
        del coords["C", 2]
        result = self.reward().score(self.write_pdb(coords))
        self.assertAlmostEqual(result["reward"], 5 / 6)
        self.assertEqual(result["coverage"]["total_pairs"], 6)
        self.assertEqual(result["coverage"]["present_pairs"], 5)
        self.assertAlmostEqual(result["coverage"]["weighted_fraction"], 5 / 6)
        missing = [r for r in result["pairs"] if not r["present"]]
        self.assertEqual(missing[0]["feature"], 0.)
        self.assertIsNone(missing[0]["distance_angstrom"])
        self.assertEqual(missing[0]["missing_atoms"][0]["position"], 2)

    def test_all_atoms_missing_returns_zero_not_nan(self):
        self.pdb.write_text("END\n")
        result = self.reward().score(self.pdb)
        self.assertEqual(result["reward"], 0.)
        self.assertEqual(result["coverage"]["present_pairs"], 0)
        json.dumps(result, allow_nan=False)

    def test_zero_occupancy_ca_is_missing_and_metadata_keeps_frozen_spec(self):
        reward = self.reward()
        self.write_pdb()
        lines = self.pdb.read_text().splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.startswith("ATOM") and line[21] == "C" and int(line[22:26]) == 2:
                lines[i] = line[:54] + "  0.00" + line[60:]
        self.pdb.write_text("".join(lines))
        result = reward.score(self.pdb)
        self.assertAlmostEqual(result["reward"], 5 / 6)
        self.assertEqual(result["coverage"]["present_pairs"], 5)
        self.assertEqual(result["observed_CA_atoms"], 11)
        self.spec_path.write_text("{}")
        self.assertEqual(reward.metadata["specification"], self.spec)

    def test_nonuniform_weights_keep_fixed_denominator(self):
        self.spec["pairs"][1]["weight"] = 5.
        self.spec["equivalent_chain_groups"] = []
        coords = dict(self.coords)
        del coords["C", 2]
        result = self.reward().score(self.write_pdb(coords))
        self.assertAlmostEqual(result["reward"], .5)

    def test_explicit_saved_chain_mapping(self):
        saved = dict(zip(self.sequences, "QWERTY"))
        for chain in self.spec["chains"]:
            chain["saved_chain"] = saved[chain["input_chain"]]
        result = self.reward().score(self.write_pdb(saved=saved))
        self.assertEqual(result["reward"], 1.)
        self.assertEqual(result["assignment"]["role_to_saved_chain"]["A"], "Q")

    def test_wrong_fasta_or_sequence_hash_or_residue_mapping_fails(self):
        bad_specs = []
        for path, value in ((["fasta_sha256"], "0" * 64),
                            (["chains", 0, "sequence_sha256"], "0" * 64),
                            (["chains", 0, "length"], 3),
                            (["pairs", 0, "left", "residue"], "A"),
                            (["pairs", 0, "left", "position"], 0),
                            (["pairs", 0, "left", "position"], 1.0)):
            spec = copy.deepcopy(self.spec)
            dest = spec
            for key in path[:-1]:
                dest = dest[key]
            dest[path[-1]] = value
            bad_specs.append(spec)
        for spec in bad_specs:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                self.reward(spec)

    def test_inactive_unresolved_or_unsupported_atom_specs_fail(self):
        for field, value in (("active", False), ("mapping_status", "pending"),
                             ("schema_version", 2), ("schema_version", True)):
            spec = copy.deepcopy(self.spec)
            spec[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.reward(spec)
        self.spec["pairs"][0]["left"]["atom"] = "CB"
        with self.assertRaisesRegex(ValueError, "Only CA"):
            self.reward()

    def test_invalid_distances_widths_and_weights_fail(self):
        for field in ("distance_angstrom", "sigma_angstrom", "weight"):
            for value in (0, -1, float("nan"), float("inf"), True, "2"):
                spec = copy.deepcopy(self.spec)
                spec["pairs"][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.reward(spec)

    def test_total_weight_overflow_rejected(self):
        for pair in self.spec["pairs"]:
            pair["weight"] = 1e308
        with self.assertRaisesRegex(ValueError, "Total pair weight"):
            self.reward()

    def test_malformed_spec_container_rejected(self):
        for bad in ([], None, "invalid"):
            self.spec_path.write_text(json.dumps(bad))
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MappedDistanceReward(self.fasta, self.spec_path)

    def test_tiny_width_far_geometry_remains_finite(self):
        for pair in self.spec["pairs"]:
            pair["sigma_angstrom"] = 1e-300
        coords = {key: (x, y, z + 1) if key[0] in "CD" else (x, y, z)
                  for key, (x, y, z) in self.coords.items()}
        result = self.reward().score(self.write_pdb(coords))
        self.assertEqual(result["reward"], 0.)
        json.dumps(result, allow_nan=False)

    def test_invalid_equivalent_groups_fail(self):
        for groups in ([["A", "C"]], [["A", "B"], ["B", "A"]], [["A", "A"]]):
            spec = copy.deepcopy(self.spec)
            spec["equivalent_chain_groups"] = groups
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                self.reward(spec)

    def test_wrong_pdb_identity_and_numbering_fail(self):
        reward = self.reward()
        self.write_pdb()
        self.pdb.write_text(self.pdb.read_text().replace("ALA A   1", "SER A   1"))
        with self.assertRaisesRegex(ValueError, "identity/number"):
            reward.score(self.pdb)
        self.pdb.write_text(atom_line(1, "A", 3, "A", (0, 0, 0)))
        with self.assertRaisesRegex(ValueError, "identity/number"):
            reward.score(self.pdb)

    def test_nonfinite_coordinates_and_multiple_models_fail(self):
        reward = self.reward()
        for bad in (float("nan"), float("inf")):
            self.pdb.write_text(atom_line(1, "A", 1, "A", (bad, 0, 0)))
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "finite"):
                reward.score(self.pdb)
        self.write_pdb(modeled=True)
        self.pdb.write_text(self.pdb.read_text() + "MODEL        2\nENDMDL\n")
        with self.assertRaisesRegex(ValueError, "one model"):
            reward.score(self.pdb)

    def test_altloc_selection_deterministic_by_occupancy_then_blank_then_A(self):
        reward = self.reward()
        self.write_pdb()
        base = self.pdb.read_text().splitlines(keepends=True)
        # Occupancy wins over blank; equal occupancy prefers blank over A.
        variants = [atom_line(500, "C", 1, "N", (0, 0, 30), altloc="B", occupancy=.3),
                    atom_line(501, "C", 1, "N", (0, 0, 20), altloc="A", occupancy=.8),
                    atom_line(502, "C", 1, "N", self.coords["C", 1], occupancy=.8)]
        base = [line for line in base if not (line.startswith("ATOM") and line[21] == "C" and int(line[22:26]) == 1)]
        for order in (variants, variants[::-1]):
            self.pdb.write_text("".join(base + order))
            self.assertEqual(reward.score(self.pdb)["reward"], 1.)

    def joint_spec(self):
        spec = copy.deepcopy(self.spec)
        spec["schema_version"] = 2
        for receptor, ligand in (("E", "A"), ("F", "B")):
            for r, l in ((1, 1), (2, 2), (1, 2)):
                spec["pairs"].append({"id": f"{receptor}{r}_{ligand}{l}",
                    "left": {"chain": receptor, "position": r,
                             "residue": self.sequences[receptor][r - 1], "atom": "CA"},
                    "right": {"chain": ligand, "position": l,
                              "residue": self.sequences[ligand][l - 1], "atom": "CA"},
                    "distance_angstrom": math.dist(self.coords[receptor, r], self.coords[ligand, l]),
                    "sigma_angstrom": 2., "weight": 1., "evidence_labels": ["synthetic_fixture"]})
        spec["interface_groups"] = [
            {"id": receptor, "receptor_chain": receptor, "type": "type_I" if receptor in "CD" else "type_II",
             "weight": 1., "pair_ids": [p["id"] for p in spec["pairs"] if p["left"]["chain"] == receptor]}
            for receptor in "CDEF"]
        spec["aggregation"] = {"minimum_fraction": .5}
        spec["integrity_pairs"] = [{"id": "ligand_dimer", "left": {"chain": "A", "position": 1, "residue": "A", "atom": "CA"},
            "right": {"chain": "B", "position": 1, "residue": "A", "atom": "CA"},
            "distance_angstrom": 40., "sigma_angstrom": 3., "weight": 1., "evidence_labels": ["synthetic_fixture"]}]
        spec["safeguards"] = {"ca_clash_distance_angstrom": 3., "ca_clash_weight": 1.,
            "backbone_min_angstrom": 2.8, "backbone_max_angstrom": 7., "backbone_sigma_angstrom": .5,
            "backbone_weight": .2, "missing_core_weight": 2., "maximum_penalty_exponent": 6.,
            "integrity_factor_floor": .5,
            "required_ca_ranges": [{"chain": c, "start": 1, "end": 2} for c in self.sequences]}
        return spec

    def test_joint_reference_optimum_has_four_copy_scores(self):
        result = self.reward(self.joint_spec()).score(self.write_pdb())
        self.assertEqual(result["reward"], 1.)
        self.assertEqual(len(result["interfaces"]), 4)
        self.assertEqual(result["minimum_interface_score"], 1.)
        self.assertEqual(result["physical_safeguards"]["factor"], 1.)
        self.assertEqual(result["integrity_factor"], 1.)
        json.dumps(result, allow_nan=False)

    def test_joint_weak_copy_is_not_hidden_by_three_good_copies(self):
        coords = {key: (x + 40, y, z) if key[0] == "F" else (x, y, z)
                  for key, (x, y, z) in self.coords.items()}
        result = self.reward(self.joint_spec()).score(self.write_pdb(coords))
        self.assertAlmostEqual(result["mean_interface_score"], .75)
        self.assertLess(result["minimum_interface_score"], .01)
        self.assertAlmostEqual(result["reward"], .375)
        self.assertIn("F", result["weakest_interface_ids"])

    def test_joint_duplicate_site_and_core_overlap_are_penalized(self):
        coords = dict(self.coords)
        for pos in (1, 2):
            coords["F", pos] = coords["E", pos]
        result = self.reward(self.joint_spec()).score(self.write_pdb(coords))
        self.assertLess(result["physical_safeguards"]["factor"], .15)
        self.assertLess(result["reward"], result["interface_score"])
        self.assertEqual(result["diagnostics"]["guard_overlap_pairs"], 2)

    def test_joint_backbone_stretch_and_missing_core_are_explicit(self):
        reward = self.reward(self.joint_spec())
        coords = dict(self.coords)
        coords["E", 2] = (10, 12, 0)
        stretched = reward.score(self.write_pdb(coords))
        self.assertLess(stretched["physical_safeguards"]["factor"], .01)
        del coords["E", 2]
        missing = reward.score(self.write_pdb(coords))
        self.assertEqual(missing["physical_safeguards"]["required_CA_present"], 11)
        self.assertEqual(missing["physical_safeguards"]["missing_required_CA"], [{"chain": "E", "position": 2}])
        self.assertGreater(missing["physical_safeguards"]["uncapped_penalty_terms"]["missing_core"], 0.)

    def test_joint_dimer_integrity_has_gentle_floor(self):
        coords = {key: (x + 30, y, z) if key[0] in "BDF" else (x, y, z)
                  for key, (x, y, z) in self.coords.items()}
        result = self.reward(self.joint_spec()).score(self.write_pdb(coords))
        self.assertEqual(result["interface_score"], 1.)
        self.assertAlmostEqual(result["integrity_factor"], .5)
        self.assertAlmostEqual(result["reward"], .5)

    def test_joint_copy_permutation_preserves_whole_assignment_and_guards(self):
        swaps = dict(zip("ABCDEF", "BADCFE"))
        coords = {(swaps[c], p): xyz for (c, p), xyz in self.coords.items()}
        result = self.reward(self.joint_spec()).score(self.write_pdb(coords))
        self.assertEqual(result["reward"], 1.)
        self.assertEqual(len(set(result["assignment"]["role_to_input_chain"].values())), 6)

    def test_joint_group_and_safeguard_validation(self):
        spec = self.joint_spec()
        spec["interface_groups"][1]["pair_ids"] = spec["interface_groups"][0]["pair_ids"]
        with self.assertRaises(ValueError):
            self.reward(spec)
        spec = self.joint_spec()
        spec["safeguards"]["backbone_min_angstrom"] = 8.
        with self.assertRaisesRegex(ValueError, "bounds"):
            self.reward(spec)
        spec = self.joint_spec()
        spec["safeguards"]["ca_clash_weight"] = float("nan")
        with self.assertRaises(ValueError):
            self.reward(spec)


if __name__ == "__main__":
    unittest.main()
