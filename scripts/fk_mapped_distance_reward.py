"""Validated, bounded CA-distance reward; standard library only.

Schema (all fields required; ``target_id`` and ``scope`` are descriptive)::

    {
      "schema_version": 1, "active": true, "mapping_status": "verified",
      "target_id": "BMP6_ACVR1_ACVR2B", "scope": "protein-only reconstruction",
      "fasta_sha256": "<SHA256 of exact FASTA bytes>",
      "evidence_labels": ["9N4K_partial_reconstruction"],
      "chains": [{"input_chain": "A", "saved_chain": "A", "length": 104,
                  "sequence_sha256": "<SHA256 of uppercase sequence>"}, ...],
      "equivalent_chain_groups": [["A", "B"], ["C", "D"], ["E", "F"]],
      "pairs": [{"id": "wrist_C", "left": {
          "chain": "C", "position": 11, "residue": "S", "atom": "CA"},
        "right": {"chain": "A", "position": 41, "residue": "N", "atom": "CA"},
        "distance_angstrom": 8.044, "sigma_angstrom": 2.0, "weight": 1.0,
        "evidence_labels": ["9N4K_measured_CA_geometry"]}]
    }

Pair chains are FASTA identities; positions and PDB residue numbers are 1-based.
The explicit saved-chain map accommodates parser regrouping. Equivalent groups
must contain identical sequences. All pair distances share ONE bijection, chosen
by exhaustive maximization of the weighted mean Gaussian score. Missing atoms
score zero with the original denominator. This is a sparse reconstruction
objective, not an energy, affinity, signaling, or native-accuracy estimate.

Version 2 adds ``interface_groups`` (id, receptor_chain, type, pair_ids, weight),
``aggregation`` (minimum_fraction in [0,1]), optional ``integrity_pairs`` using
the same pair schema, and explicit ``safeguards``. Every interface pair belongs
to exactly one group. The score combines mean and weakest interface, then applies
bounded CA-overlap/backbone/coverage penalties and a gentle integrity multiplier.
Integrity targets are additional guidance, never independent validation.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path


AMINO_ACIDS = dict(zip(
    ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
     "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"),
    "ARNDCQEGHILKMFPSTWYV"))
MAX_ASSIGNMENTS = 4096


def _positive(value, label):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _labels(value):
    if not isinstance(value, list) or not value or any(
            not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError("evidence_labels must be a nonempty list of strings")
    return list(value)


def _fasta(path):
    sequences = {}
    current = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            tokens = line[1:].split("|", 1)[0].split()
            current = tokens[0] if tokens else ""
            if not current or current in sequences:
                raise ValueError("FASTA chain IDs must be nonempty and unique")
            sequences[current] = ""
        else:
            if current is None or any(x not in AMINO_ACIDS.values() for x in line):
                raise ValueError("FASTA requires uppercase standard amino-acid sequences")
            sequences[current] += line
    if not sequences or any(not seq for seq in sequences.values()):
        raise ValueError("FASTA contains no sequence or an empty chain")
    return sequences


class MappedDistanceReward:
    """Freeze a verified specification and score single-model standard PDBs."""

    def __init__(self, fasta_path, spec_json_path):
        fasta_bytes = Path(fasta_path).read_bytes()
        spec_bytes = Path(spec_json_path).read_bytes()
        spec = json.loads(spec_bytes)
        if (not isinstance(spec, dict) or type(spec.get("schema_version")) is not int
                or spec.get("schema_version") not in (1, 2) or spec.get("active") is not True
                or spec.get("mapping_status") != "verified"):
            raise ValueError("Reward spec must be version 1 or 2, active, and mapping verified")
        self._version = spec["schema_version"]
        if any(not isinstance(spec.get(k), str) or not spec[k].strip()
               for k in ("target_id", "scope")):
            raise ValueError("target_id and scope must be nonempty descriptive strings")
        fasta_hash = hashlib.sha256(fasta_bytes).hexdigest()
        if spec.get("fasta_sha256") != fasta_hash:
            raise ValueError("FASTA SHA256 does not match verified reward spec")
        self._sequences = _fasta(fasta_path)
        self._saved = {}
        chains = spec.get("chains")
        if not isinstance(chains, list):
            raise ValueError("chains must specify every FASTA chain")
        for chain in chains:
            if not isinstance(chain, dict):
                raise ValueError("Each chain specification must be an object")
            name, saved = chain.get("input_chain"), chain.get("saved_chain")
            if (not isinstance(name, str) or name not in self._sequences or name in self._saved
                    or not isinstance(saved, str) or len(saved) != 1
                    or not saved.isascii() or not saved.isalnum()
                    or saved in self._saved.values()):
                raise ValueError("Invalid or duplicate input/saved chain mapping")
            seq = self._sequences[name]
            if (type(chain.get("length")) is not int or chain["length"] != len(seq)
                    or chain.get("sequence_sha256") != hashlib.sha256(seq.encode()).hexdigest()):
                raise ValueError(f"Sequence length/hash mismatch for chain {name}")
            self._saved[name] = saved
        if set(self._saved) != set(self._sequences):
            raise ValueError("Spec chain mapping must cover exactly the FASTA chains")
        self._input = {saved: name for name, saved in self._saved.items()}
        evidence = _labels(spec.get("evidence_labels"))

        groups = spec.get("equivalent_chain_groups")
        if not isinstance(groups, list):
            raise ValueError("equivalent_chain_groups must be a list")
        used, assignment_count = set(), 1
        for group in groups:
            if (not isinstance(group, list) or len(group) < 2
                    or any(not isinstance(c, str) or c not in self._sequences for c in group)
                    or len(set(group)) != len(group) or used.intersection(group)):
                raise ValueError("Equivalent groups must be disjoint known chain lists")
            if len({self._sequences[c] for c in group}) != 1:
                raise ValueError("Equivalent chains must have identical verified sequences")
            used.update(group)
            assignment_count *= math.factorial(len(group))
        if assignment_count > MAX_ASSIGNMENTS:
            raise ValueError(f"Too many coherent assignments (maximum {MAX_ASSIGNMENTS})")
        self._assignments = []
        for permutations in itertools.product(*(itertools.permutations(g) for g in groups)):
            assignment = {chain: chain for chain in self._sequences}
            for group, perm in zip(groups, permutations):
                assignment.update(zip(group, perm))
            self._assignments.append(assignment)

        self._pairs = []
        ids = set()
        pairs = spec.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("At least one distance pair is required")
        self._interface_count = len(pairs)
        integrity_pairs = spec.get("integrity_pairs", []) if self._version == 2 else []
        if not isinstance(integrity_pairs, list):
            raise ValueError("integrity_pairs must be a list")
        pairs = pairs + integrity_pairs
        for pair in pairs:
            if not isinstance(pair, dict):
                raise ValueError("Each distance pair must be an object")
            name = pair.get("id")
            if not isinstance(name, str) or not name.strip() or name in ids:
                raise ValueError("Pair IDs must be nonempty and unique")
            ids.add(name)
            endpoints = []
            for field in ("left", "right"):
                point = pair.get(field, {})
                if not isinstance(point, dict):
                    raise ValueError("Each pair endpoint must be an object")
                chain, pos = point.get("chain"), point.get("position")
                if (not isinstance(chain, str) or chain not in self._sequences or type(pos) is not int
                        or not 1 <= pos <= len(self._sequences[chain])):
                    raise ValueError(f"Invalid 1-based position in {name}/{field}")
                if point.get("residue") != self._sequences[chain][pos - 1]:
                    raise ValueError(f"Residue identity mismatch in {name}/{field}")
                if point.get("atom") != "CA":
                    raise ValueError("Only CA distance atoms are supported")
                endpoints.append((chain, pos))
            if endpoints[0] == endpoints[1]:
                raise ValueError("Distance pair endpoints must be different atoms")
            self._pairs.append({"id": name, "endpoints": tuple(endpoints),
                                "distance": _positive(pair.get("distance_angstrom"), "distance"),
                                "sigma": _positive(pair.get("sigma_angstrom"), "sigma"),
                                "weight": _positive(pair.get("weight"), "weight"),
                                "evidence_labels": _labels(pair.get("evidence_labels"))})
        try:
            self._total_weight = math.fsum(p["weight"] for p in self._pairs)
        except OverflowError as exc:
            raise ValueError("Total pair weight must be finite") from exc
        if self._version == 2:
            self._configure_joint(spec)
        self.metadata = {
            "kind": "mapped_ca_distance", "schema_version": self._version,
            "target_id": spec.get("target_id"), "scope": spec.get("scope"),
            "fasta_sha256": fasta_hash, "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
            "specification": spec,
            "evidence_labels": evidence, "pair_count": len(self._pairs),
            "total_weight": self._total_weight, "positive_maximum": 1.0,
            "assignment_count": len(self._assignments),
            "saved_chain_to_input_chain": dict(self._input),
            "score_definition": "sum(weight * exp(-0.5*((d-reference)/sigma)^2))/sum(weight)",
            "missing_atoms": "zero feature with full fixed denominator",
            "overlap_diagnostic_threshold_angstrom": 2.0,
            "limited_scope": "Sparse protein CA geometry only; not physical energy, binding affinity, signaling, or independent structural accuracy."}
        if self._version == 2:
            self.metadata.update({
                "score_definition": "((1-minimum_fraction)*weighted_mean_interface + minimum_fraction*weakest_interface) * safeguards * integrity_factor",
                "interface_pair_count": self._interface_count,
                "integrity_pair_count": len(integrity_pairs),
                "integrity_definition": "floor + (1-floor)*minimum_integrity_pair_Gaussian",
                "safeguard_threshold_status": "Engineering settings, not calibrated acceptance thresholds"})

    def _configure_joint(self, spec):
        groups = spec.get("interface_groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("Version 2 requires interface_groups")
        expected = {p["id"]: p for p in self._pairs[:self._interface_count]}
        seen, group_ids, self._interfaces = set(), set(), []
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError("Interface group must be an object")
            name, receptor = group.get("id"), group.get("receptor_chain")
            members = group.get("pair_ids")
            if (not isinstance(name, str) or not name or name in group_ids
                    or receptor not in self._sequences or group.get("type") not in ("type_I", "type_II")
                    or not isinstance(members, list) or not members
                    or any(not isinstance(m, str) or m not in expected for m in members)
                    or len(set(members)) != len(members) or seen.intersection(members)):
                raise ValueError("Invalid interface group or repeated/unknown pair")
            if any(sum(c == receptor for c, _ in expected[m]["endpoints"]) != 1 for m in members):
                raise ValueError("Each interface pair must touch its receptor exactly once")
            seen.update(members)
            group_ids.add(name)
            self._interfaces.append(dict(id=name, receptor_chain=receptor, type=group["type"],
                                         pair_ids=list(members), weight=_positive(group.get("weight"), "interface weight")))
        if seen != set(expected):
            raise ValueError("Every interface pair must belong to exactly one group")
        fraction = spec.get("aggregation", {}).get("minimum_fraction")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
            raise ValueError("minimum_fraction must be finite in [0,1]")
        self._minimum_fraction = float(fraction)
        guard = spec.get("safeguards")
        if not isinstance(guard, dict):
            raise ValueError("Version 2 requires explicit safeguards")
        keys = ("ca_clash_distance_angstrom", "ca_clash_weight", "backbone_min_angstrom",
                "backbone_max_angstrom", "backbone_sigma_angstrom", "backbone_weight",
                "missing_core_weight", "maximum_penalty_exponent")
        self._guard = {k: _positive(guard.get(k), k) for k in keys}
        if self._guard["backbone_min_angstrom"] >= self._guard["backbone_max_angstrom"]:
            raise ValueError("Backbone bounds must be ordered")
        floor = guard.get("integrity_factor_floor")
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not 0 <= floor <= 1:
            raise ValueError("integrity_factor_floor must be finite in [0,1]")
        self._guard["integrity_factor_floor"] = floor
        self._required = set()
        ranges = guard.get("required_ca_ranges")
        if not isinstance(ranges, list) or not ranges:
            raise ValueError("Explicit required_ca_ranges are needed for missing-core checks")
        for segment in ranges:
            if not isinstance(segment, dict):
                raise ValueError("Required CA range must be an object")
            chain, start, stop = segment.get("chain"), segment.get("start"), segment.get("end")
            if (chain not in self._sequences or type(start) is not int or type(stop) is not int
                    or not 1 <= start <= stop <= len(self._sequences[chain])):
                raise ValueError("Invalid required CA range")
            for position in range(start, stop + 1):
                if (chain, position) in self._required:
                    raise ValueError("Required CA ranges cannot overlap")
                self._required.add((chain, position))

    def _read_ca(self, path):
        lines = Path(path).read_text().splitlines()
        model_lines = [i for i, x in enumerate(lines) if x[:6].strip() == "MODEL"]
        ends = [i for i, x in enumerate(lines) if x[:6].strip() == "ENDMDL"]
        if (len(model_lines) > 1 or len(ends) != len(model_lines)
                or (model_lines and ends[0] <= model_lines[0])):
            raise ValueError("PDB must contain one model with valid MODEL/ENDMDL records")
        selected, identities = {}, {}
        atom_count = 0
        for lineno, line in enumerate(lines):
            if line[:6].strip() != "ATOM":
                continue
            if model_lines and not model_lines[0] < lineno < ends[0]:
                raise ValueError("ATOM record outside the single PDB model")
            try:
                saved, position = line[21], int(line[22:26])
                xyz = tuple(float(line[i:i + 8]) for i in (30, 38, 46))
                occupancy = float(line[54:60]) if line[54:60].strip() else 1.0
                resname, atom, altloc = line[17:20].strip(), line[12:16].strip(), line[16]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Malformed ATOM record at line {lineno + 1}") from exc
            if not all(math.isfinite(x) for x in (*xyz, occupancy)) or not 0 <= occupancy <= 1:
                raise ValueError("PDB coordinates/occupancies must be finite; occupancy must be in [0,1]")
            if line[26:27].strip():
                raise ValueError("Insertion codes are unsupported; require 1-based input residue numbering")
            if saved not in self._input:
                raise ValueError(f"Unexpected saved PDB chain {saved!r}")
            chain = self._input[saved]
            if (not 1 <= position <= len(self._sequences[chain])
                    or AMINO_ACIDS.get(resname) != self._sequences[chain][position - 1]):
                raise ValueError(f"PDB residue identity/number mismatch at {saved}:{position}")
            key = chain, position
            if key in identities and identities[key] != resname:
                raise ValueError("Inconsistent alternate residue identity")
            identities[key] = resname
            atom_count += 1
            if atom != "CA" or occupancy == 0:
                continue
            # Largest occupancy, then blank, then A, then lexical altloc/coordinates.
            # Coordinates break exact metadata ties independently of record order.
            priority = (-occupancy, 0 if altloc == " " else 1 if altloc == "A" else 2,
                        altloc, xyz)
            if key not in selected or priority < selected[key][0]:
                selected[key] = priority, xyz
        return {key: row[1] for key, row in selected.items()}, atom_count

    @staticmethod
    def _overlaps(atoms, guard_distance=None):
        count, closest, soft_sum, guard_count = 0, None, 0., 0
        for (left, xyz), (right, other) in itertools.combinations(atoms.items(), 2):
            if left[0] == right[0]:
                continue
            d = math.dist(xyz, other)
            if not math.isfinite(d):
                raise ValueError("Nonfinite distance from PDB coordinates")
            closest = d if closest is None else min(closest, d)
            count += d < 2.0
            if guard_distance is not None and d < guard_distance:
                guard_count += 1
                soft_sum += ((guard_distance - d) / guard_distance) ** 2
        return {"interchain_CA_pairs_below_2_angstrom": count,
                "minimum_interchain_CA_distance_angstrom": closest,
                "guard_overlap_pairs": guard_count if guard_distance is not None else None,
                "guard_overlap_squared_sum": soft_sum if guard_distance is not None else None,
                "interpretation": "Conservative CA overlap diagnostic only; not an all-atom clash score or energy."}

    def _physical_guard(self, atoms, overlaps):
        g = self._guard
        backbone_rows, worst_error = [], 0.
        for chain, sequence in self._sequences.items():
            observed, failures, chain_worst = 0, 0, 0.
            for position in range(1, len(sequence)):
                if (chain, position) not in atoms or (chain, position + 1) not in atoms:
                    continue
                d = math.dist(atoms[chain, position], atoms[chain, position + 1])
                if not math.isfinite(d):
                    raise ValueError("Nonfinite backbone distance")
                error = max(g["backbone_min_angstrom"] - d, d - g["backbone_max_angstrom"], 0.)
                observed += 1
                failures += error > 0
                chain_worst = max(chain_worst, error)
            worst_error = max(worst_error, chain_worst)
            backbone_rows.append({"input_chain": chain, "observed_bonds": observed,
                                  "expected_bonds": len(sequence) - 1, "out_of_range_bonds": failures,
                                  "maximum_bound_excess_angstrom": chain_worst if observed else None})
        missing = self._required.difference(atoms)
        terms = {"ca_clash": g["ca_clash_weight"] * overlaps["guard_overlap_squared_sum"],
                 "backbone": g["backbone_weight"] * (worst_error / g["backbone_sigma_angstrom"]) ** 2,
                 "missing_core": g["missing_core_weight"] * len(missing) / len(self._required)}
        exponent = min(g["maximum_penalty_exponent"], sum(terms.values()))
        return {"factor": math.exp(-exponent), "penalty_exponent": exponent,
                "uncapped_penalty_terms": terms, "backbone": backbone_rows,
                "required_CA_present": len(self._required) - len(missing),
                "required_CA_total": len(self._required),
                "missing_required_CA": [{"chain": c, "position": p} for c, p in sorted(missing)],
                "unobserved_tail_status": "Absent coordinates outside required ranges are unassessed, not a geometry pass.",
                "threshold_status": "uncalibrated engineering safeguards"}

    def _joint_value(self, rows, physical_guard):
        by_id = {r["id"]: r for r in rows}
        copies = []
        for group in self._interfaces:
            pairs = [by_id[p] for p in group["pair_ids"]]
            denominator = math.fsum(p["weight"] for p in pairs)
            value = math.fsum(p["weight"] * p["feature"] for p in pairs) / denominator
            copies.append({**group, "score": value,
                           "present_pairs": sum(p["present"] for p in pairs), "total_pairs": len(pairs)})
        mean = math.fsum(c["weight"] * c["score"] for c in copies) / math.fsum(c["weight"] for c in copies)
        minimum = min(c["score"] for c in copies)
        interface_score = (1 - self._minimum_fraction) * mean + self._minimum_fraction * minimum
        integrity_rows = rows[self._interface_count:]
        integrity = min((p["feature"] for p in integrity_rows), default=1.)
        floor = self._guard["integrity_factor_floor"]
        factor = floor + (1 - floor) * integrity
        return {"reward": interface_score * physical_guard["factor"] * factor,
                "interfaces": copies, "mean_interface_score": mean,
                "minimum_interface_score": minimum, "interface_score": interface_score,
                "weakest_interface_ids": [c["id"] for c in copies if c["score"] == minimum],
                "integrity_minimum": integrity, "integrity_factor": factor,
                "physical_safeguards": physical_guard}

    def score(self, pdb_path):
        atoms, atom_count = self._read_ca(pdb_path)
        overlaps = self._overlaps(atoms, self._guard["ca_clash_distance_angstrom"] if self._version == 2 else None)
        physical_guard = self._physical_guard(atoms, overlaps) if self._version == 2 else None
        best, best_key = None, None
        for assignment in self._assignments:
            rows = []
            for pair in self._pairs:
                keys = [(assignment[chain], pos) for chain, pos in pair["endpoints"]]
                present = all(key in atoms for key in keys)
                distance, value = None, 0.0
                if present:
                    distance = math.dist(*(atoms[key] for key in keys))
                    if not math.isfinite(distance):
                        raise ValueError("Nonfinite pair distance")
                    z = abs(distance - pair["distance"]) / pair["sigma"]
                    value = math.exp(-0.5 * z * z)
                rows.append({"id": pair["id"], "feature": value,
                             "distance_angstrom": distance, "present": present,
                             "weight": pair["weight"],
                             "missing_atoms": [{"input_chain": chain, "saved_chain": self._saved[chain],
                                                "position": pos, "atom": "CA"}
                                               for chain, pos in keys if (chain, pos) not in atoms]})
            joint = self._joint_value(rows, physical_guard) if self._version == 2 else {}
            reward = joint.get("reward", math.fsum(p["weight"] * p["feature"] for p in rows) / self._total_weight)
            present_weight = math.fsum(p["weight"] for p in rows if p["present"])
            covered = sum(p["present"] for p in rows)
            key = reward, present_weight, covered
            if best_key is None or key > best_key:
                best_key = key
                best = {"reward": min(1.0, max(0.0, reward)),
                        "features": {p["id"]: p["feature"] for p in rows}, "pairs": rows,
                        "assignment": {"role_to_input_chain": dict(assignment),
                                       "role_to_saved_chain": {c: self._saved[a] for c, a in assignment.items()}},
                        "coverage": {"present_pairs": covered, "total_pairs": len(rows),
                                     "pair_fraction": covered / len(rows),
                                     "weighted_fraction": present_weight / self._total_weight}}
                best.update(joint)
        best.update({"assignments_evaluated": len(self._assignments),
                     "observed_CA_atoms": len(atoms), "observed_ATOM_records": atom_count,
                     "diagnostics": overlaps,
                     "limited_scope": self.metadata["limited_scope"]})
        return best
