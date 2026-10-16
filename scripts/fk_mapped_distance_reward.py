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

Version 3 preserves the CA objective and adds ``local_geometry_modules``. Each
``distance_three_angles`` module contains ``copies`` with one ``distance``
(two ``atoms``, ``target_angstrom``) and three ``angles`` (three ordered ``atoms``,
``target_degrees``; the middle atom is the vertex). Endpoints use the same chain,
position, residue and atom fields. Each copy names its receptor/interface group.
Module settings are distance/angle tolerance and scale, missing_geometry_error,
strength, factor_floor and evidence_labels. The worst copy's flat-tolerance Huber
error gives ``floor + (1-floor)*exp(-strength*error)``. All modules and CA terms
share one assignment. Strength zero exactly retains the version-2 scalar/map.

Optional version-2/3 extensions preserve omitted-field behavior exactly:
``distance_feature: "cauchy"`` on a pair uses 1/(1+((d-target)/sigma)^2)
instead of the default Gaussian. ``required_subgroups`` on an interface is a
disjoint, complete partition of its pair_ids into {id, pair_ids} objects; the
interface is the minimum of their independently normalized weighted means.
Every subgroup uses the same assembly assignment and fixed missing-atom
denominators. These are engineering objectives, not additional contact evidence.
An interface-group pair may explicitly declare ``scope: "intra_receptor"``:
both endpoints must then belong to that group's receptor. Omitted scope keeps
the default requirement of exactly one receptor endpoint. This version-2/3
extension uses the same distance features and subgroup aggregation; internal
spans describe receptor shape, not intermolecular contacts.

``heavy_atom_safeguard`` optionally multiplies the final reward by a bounded
interchain-overlap factor. It requires explicit symmetric native disulfides,
standard residue/peptide topology, finite coordinates, and complete standard
heavy atoms. No coordinate-inferred bonds or PDB CONECT exclusions are used.
The spatial grid is evaluated once per model, outside the assignment loop.
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
GEOMETRY_ATOMS = {"N", "CA", "C", "CB", "OG"}


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


def _nonnegative(value, label):
    if value == 0 and not isinstance(value, bool):
        return 0.0
    return _positive(value, label)


def validate_local_geometry_modules(modules, sequences, interfaces):
    """Validate the deliberately narrow, mapped local-geometry schema."""
    if not isinstance(modules, list) or not modules:
        raise ValueError("Version 3 requires nonempty local_geometry_modules")
    groups = {g["id"]: g for g in interfaces}
    ids = set()
    for module in modules:
        if not isinstance(module, dict):
            raise ValueError("Geometry module must be an object")
        name = module.get("id")
        if (not isinstance(name, str) or not name or name in ids
                or module.get("kind") != "distance_three_angles"):
            raise ValueError("Invalid or repeated geometry module id/kind")
        ids.add(name)
        _labels(module.get("evidence_labels"))
        for key in ("distance_scale_angstrom", "angle_scale_degrees", "missing_geometry_error"):
            _positive(module.get(key), key)
        for key in ("distance_tolerance_angstrom", "angle_tolerance_degrees", "strength", "factor_floor"):
            _nonnegative(module.get(key), key)
        if module["factor_floor"] > 1 or module["angle_tolerance_degrees"] > 180:
            raise ValueError("Geometry factor floor/angle tolerance out of range")
        copies = module.get("copies")
        if not isinstance(copies, list) or not copies:
            raise ValueError("Geometry module requires copies")
        copy_ids, receptor_ids = set(), set()
        for copy in copies:
            if not isinstance(copy, dict):
                raise ValueError("Geometry copy must be an object")
            cid, receptor, group_id = copy.get("id"), copy.get("receptor_chain"), copy.get("interface_group")
            if (not isinstance(cid, str) or not cid or cid in copy_ids
                    or not isinstance(receptor, str) or receptor not in sequences
                    or receptor in receptor_ids or not isinstance(group_id, str) or group_id not in groups
                    or groups[group_id]["receptor_chain"] != receptor):
                raise ValueError("Geometry copy must identify a unique receptor/interface")
            copy_ids.add(cid)
            receptor_ids.add(receptor)
            _labels(copy.get("evidence_labels"))
            angles = copy.get("angles")
            if not isinstance(angles, list) or len(angles) != 3:
                raise ValueError("Geometry copy requires exactly three angles")
            distance = copy.get("distance")
            angle_keys = set()
            for measurement, size, field in [(distance, 2, "target_angstrom")] + [
                    (a, 3, "target_degrees") for a in angles]:
                if not isinstance(measurement, dict):
                    raise ValueError("Geometry measurement must be an object")
                target = _positive(measurement.get(field), field)
                if size == 3 and target >= 180:
                    raise ValueError("Angle target must be strictly between 0 and 180 degrees")
                points = measurement.get("atoms")
                if not isinstance(points, list) or len(points) != size:
                    raise ValueError("Invalid number of geometry atoms")
                keys = []
                for point in points:
                    if not isinstance(point, dict):
                        raise ValueError("Geometry atom must be a mapped endpoint")
                    chain, pos, aa, atom = (point.get(k) for k in ("chain", "position", "residue", "atom"))
                    if (not isinstance(chain, str) or chain not in sequences or type(pos) is not int
                            or not 1 <= pos <= len(sequences[chain]) or aa != sequences[chain][pos-1]):
                        raise ValueError("Geometry endpoint residue/position mismatch")
                    if (not isinstance(atom, str) or atom not in GEOMETRY_ATOMS
                            or (atom == "OG" and aa != "S") or (atom == "CB" and aa == "G")):
                        raise ValueError("Unsupported geometry atom for residue")
                    keys.append((chain, pos, atom))
                if len(set(keys)) != size or not any(k[0] == receptor for k in keys):
                    raise ValueError("Geometry measurement requires distinct atoms and its receptor")
                if size == 3:
                    angle_key = (keys[1], tuple(sorted((keys[0], keys[2]))))
                    if angle_key in angle_keys:
                        raise ValueError("Geometry approach angles must be distinct")
                    angle_keys.add(angle_key)
            distance_chains = {p["chain"] for p in distance["atoms"]}
            if len(distance_chains) != 2 or receptor not in distance_chains:
                raise ValueError("Geometry distance must connect its receptor to one other chain")
            if any(p["chain"] not in distance_chains for a in angles for p in a["atoms"]):
                raise ValueError("Geometry angles must use the same receptor/partner chains")
        receptor_sequences = {sequences[c] for c in receptor_ids}
        if len(receptor_sequences) != 1 or receptor_ids != {
                c for c, sequence in sequences.items() if sequence in receptor_sequences}:
            raise ValueError("Geometry module must cover every copy of one receptor sequence")
    return modules


def geometry_measurement(coordinates):
    """Finite distance or ordered angle; None for degenerate angle arms."""
    if len(coordinates) == 2:
        value = math.dist(*coordinates)
        if not math.isfinite(value):
            raise ValueError("Nonfinite local geometry distance")
        return value
    left, vertex, right = coordinates
    u = tuple(a-b for a, b in zip(left, vertex))
    v = tuple(a-b for a, b in zip(right, vertex))
    a, b = math.hypot(*u), math.hypot(*v)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Nonfinite local geometry angle arm")
    if a <= 1e-12 or b <= 1e-12:
        return None
    dot = math.fsum((x/a)*(y/b) for x, y in zip(u, v))
    return math.degrees(math.acos(max(-1., min(1., dot))))


def _huber_excess(value, target, tolerance, scale):
    z = max(abs(value-target)-tolerance, 0.)/scale
    if not math.isfinite(z):
        raise ValueError("Nonfinite normalized local geometry deviation")
    return 0.5*z*z if z <= 1 else z-0.5


def local_geometry_results(modules, atoms, assignment, supported_chains=None):
    """Measurements under one supplied assignment; never remap locally."""
    results = []
    for module in modules:
        copies = []
        for copy in module["copies"]:
            measurements, all_keys = [], set()
            definitions = [(copy["distance"], "distance", "target_angstrom", "angstrom")] + [
                (a, "angle", "target_degrees", "degrees") for a in copy["angles"]]
            for definition, kind, target_field, units in definitions:
                roles = [(p["chain"], p["position"], p["atom"]) for p in definition["atoms"]]
                keys = [(assignment[c], p, a) for c, p, a in roles]
                all_keys.update(keys)
                missing = [k for k in keys if k not in atoms]
                supported = supported_chains is None or all(c in supported_chains for c, _, _ in roles)
                value = geometry_measurement([atoms[k] for k in keys]) if not missing and supported else None
                status = ("unassessed_no_reference_assignment" if not supported else
                          "missing_atoms" if missing else "degenerate_geometry" if value is None else "assessed")
                scale_key = "distance_scale_angstrom" if kind == "distance" else "angle_scale_degrees"
                tolerance_key = "distance_tolerance_angstrom" if kind == "distance" else "angle_tolerance_degrees"
                error = (_huber_excess(value, definition[target_field], module[tolerance_key], module[scale_key])
                         if value is not None else None)
                measurements.append({"kind": kind, "atoms": definition["atoms"], "assigned_atom_keys": keys,
                                     "value": value, "target": definition[target_field], "units": units,
                                     "error": error, "status": status, "missing_atom_keys": missing})
            complete = all(m["error"] is not None for m in measurements)
            error = (measurements[0]["error"] or 0.) + math.fsum(m["error"] or 0. for m in measurements[1:])/3
            if not complete:
                error = max(error, module["missing_geometry_error"])
            if not math.isfinite(error):
                raise ValueError("Nonfinite local geometry error")
            copies.append({"id": copy["id"], "receptor_chain": copy["receptor_chain"],
                           "interface_group": copy["interface_group"], "evidence_labels": copy["evidence_labels"],
                           "geometry_error": error, "complete": complete,
                           "status": "assessed" if complete else "missing_degenerate_or_unassessed_geometry",
                           "measurements": measurements, "required_atoms": len(all_keys),
                           "present_atoms": sum(k in atoms for k in all_keys)})
        worst = max(c["geometry_error"] for c in copies)
        factor = (1.0 if module["strength"] == 0 else module["factor_floor"] +
                  (1-module["factor_floor"])*math.exp(-module["strength"]*worst))
        results.append({"id": module["id"], "copies": copies, "maximum_copy_error": worst,
                        "factor": factor, "strength": module["strength"], "factor_floor": module["factor_floor"],
                        "evidence_labels": module["evidence_labels"]})
    return results


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


def read_local_geometry_atoms(path, sequences, saved_to_input, modules):
    """Read mapped records; retain one conformer per geometry residue.

    CA selection remains byte-for-byte legacy behavior for on/off equality.
    Heavy-atom local geometry follows that CA altloc when it is nonblank.
    """
    required = set()
    for module in modules:
        for copy in module["copies"]:
            for definition in [copy["distance"]] + copy["angles"]:
                for point in definition["atoms"]:
                    # Every sequence-equivalent copy can be used by a bijection.
                    for c, seq in sequences.items():
                        if seq == sequences[point["chain"]]:
                            required.add((c, point["position"], point["atom"]))
    residues = {k[:2] for k in required}
    records = {}
    lines = Path(path).read_text().splitlines()
    models = [i for i, line in enumerate(lines) if line[:6].strip() == "MODEL"]
    ends = [i for i, line in enumerate(lines) if line[:6].strip() == "ENDMDL"]
    if len(models) > 1 or len(ends) != len(models) or (models and ends[0] <= models[0]):
        raise ValueError("Local geometry requires a single PDB model")
    for lineno, line in enumerate(lines):
        if line[:6].strip() != "ATOM":
            continue
        if models and not models[0] < lineno < ends[0]:
            raise ValueError("ATOM outside the single PDB model")
        try:
            chain, pos = saved_to_input[line[21]], int(line[22:26])
            atom, alt = line[12:16].strip(), line[16]
            occupancy = float(line[54:60]) if line[54:60].strip() else 1.
            xyz = tuple(float(line[i:i+8]) for i in (30, 38, 46))
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError("Malformed mapped geometry ATOM record") from exc
        if not all(math.isfinite(x) for x in (*xyz, occupancy)) or not 0 <= occupancy <= 1:
            raise ValueError("Geometry coordinates and occupancies must be finite and valid")
        if (line[26:27].strip() or not 1 <= pos <= len(sequences[chain])
                or AMINO_ACIDS.get(line[17:20].strip()) != sequences[chain][pos-1]):
            raise ValueError("Geometry ATOM residue identity/number mismatch")
        if (chain, pos) not in residues or ((chain, pos, atom) not in required and atom != "CA"):
            continue
        if occupancy == 0:
            continue
        priority = (-occupancy, 0 if alt == " " else 1 if alt == "A" else 2, alt, xyz)
        key = (chain, pos, atom, alt)
        if key not in records or priority < records[key][0]:
            records[key] = priority, xyz, occupancy
    atoms, choices = {}, []
    for chain, pos in sorted(residues):
        local = {(name, alt): value for (c, p, name, alt), value in records.items() if (c, p) == (chain, pos)}
        ca = [(value[0], alt) for (name, alt), value in local.items() if name == "CA"]
        ca_alt = min(ca)[1] if ca else " "
        alts = sorted({alt for _, alt in local if alt != " "})
        if ca_alt != " ":
            selected = ca_alt
        elif alts:
            def alt_priority(alt):
                occupied = [v[2] for (name, a), v in local.items() if a == alt]
                return -math.fsum(occupied)/len(occupied), 0 if alt == "A" else 1, alt
            selected = min(alts, key=alt_priority)
        else:
            selected = " "
        choices.append({"input_chain": chain, "position": pos, "selected_altloc": selected,
                        "legacy_CA_altloc": ca_alt})
        for c, p, name in required:
            if (c, p) != (chain, pos):
                continue
            # The local CA must be exactly the CA used by the parent objective,
            # even when a shared blank CA coexists with lower-occupancy alt CAs.
            record = (local.get((name, ca_alt)) if name == "CA" else
                      local.get((name, selected), local.get((name, " "))))
            if record is not None:
                atoms[c, p, name] = record[1]
    return atoms, choices


# Standard amino-acid heavy-atom connectivity. Kept dependency-free so the
# optional safeguard does not introduce NumPy/SciPy into checkpoint scoring.
_SIDECHAIN_BONDS = {
    "G": "", "A": "CA-CB", "R": "CA-CB CB-CG CG-CD CD-NE NE-CZ CZ-NH1 CZ-NH2",
    "N": "CA-CB CB-CG CG-OD1 CG-ND2", "D": "CA-CB CB-CG CG-OD1 CG-OD2",
    "C": "CA-CB CB-SG", "Q": "CA-CB CB-CG CG-CD CD-OE1 CD-NE2",
    "E": "CA-CB CB-CG CG-CD CD-OE1 CD-OE2",
    "H": "CA-CB CB-CG CG-ND1 ND1-CE1 CE1-NE2 NE2-CD2 CD2-CG",
    "I": "CA-CB CB-CG1 CB-CG2 CG1-CD1", "L": "CA-CB CB-CG CG-CD1 CG-CD2",
    "K": "CA-CB CB-CG CG-CD CD-CE CE-NZ", "M": "CA-CB CB-CG CG-SD SD-CE",
    "F": "CA-CB CB-CG CG-CD1 CG-CD2 CD1-CE1 CD2-CE2 CE1-CZ CE2-CZ",
    "P": "CA-CB CB-CG CG-CD CD-N", "S": "CA-CB CB-OG",
    "T": "CA-CB CB-OG1 CB-CG2",
    "W": "CA-CB CB-CG CG-CD1 CG-CD2 CD1-NE1 NE1-CE2 CE2-CD2 CD2-CE3 CE3-CZ3 CZ3-CH2 CH2-CZ2 CZ2-CE2",
    "Y": "CA-CB CB-CG CG-CD1 CG-CD2 CD1-CE1 CD2-CE2 CE1-CZ CE2-CZ CZ-OH",
    "V": "CA-CB CB-CG1 CB-CG2"}
_HEAVY_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}


class _HeavyAtomSafeguard:
    """A fixed-topology, assignment-invariant, bounded overlap multiplier."""

    def __init__(self, config, sequences, saved_to_input, assignments):
        if not isinstance(config, dict) or config.get("kind") != "interchain_overlap":
            raise ValueError("heavy_atom_safeguard must have kind interchain_overlap")
        elements = config.get("elements")
        if (not isinstance(elements, list) or any(not isinstance(e, str) for e in elements)
                or len(set(elements)) != len(elements) or set(elements) not in ({"C", "N", "O"}, {"C", "N", "O", "S"})):
            raise ValueError("Heavy safeguard elements must be C/N/O or C/N/O/S")
        self.elements = set(elements)
        self.threshold = _positive(config.get("overlap_threshold_angstrom"), "overlap_threshold_angstrom")
        self.scale = _positive(config.get("overlap_scale_angstrom"), "overlap_scale_angstrom")
        self.pair_scale = _positive(config.get("pair_scale"), "pair_scale")
        self.strength = _nonnegative(config.get("strength"), "strength")
        self.floor = _nonnegative(config.get("factor_floor"), "factor_floor")
        self.cutoff = 2*max(_HEAVY_RADII[e] for e in elements)-self.threshold
        if self.floor > 1 or self.cutoff <= 0:
            raise ValueError("Heavy safeguard floor/overlap threshold out of range")
        if config.get("missing_heavy_atom_policy") != "factor_floor":
            raise ValueError("Heavy safeguard requires missing_heavy_atom_policy=factor_floor")
        _labels(config.get("evidence_labels"))
        self.sequences, self.input = sequences, saved_to_input
        self.required, self.allowed, graph = set(), set(), {}

        def bond(a, b):
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)

        for chain, sequence in sequences.items():
            for pos, aa in enumerate(sequence, 1):
                for pair in ("N-CA CA-C C-O "+_SIDECHAIN_BONDS[aa]).split():
                    a, b = pair.split("-")
                    left, right = (chain, pos, a), (chain, pos, b)
                    self.required.update((left, right))
                    bond(left, right)
                if pos > 1:
                    bond((chain, pos-1, "C"), (chain, pos, "N"))
            terminal = (chain, len(sequence), "OXT")
            self.allowed.add(terminal)
            bond((chain, len(sequence), "C"), terminal)
        self.allowed.update(self.required)
        disulfides = config.get("native_disulfides")
        if not isinstance(disulfides, list):
            raise ValueError("Explicit native_disulfides list is required")
        bonds, used = set(), set()
        for row in disulfides:
            if not isinstance(row, dict):
                raise ValueError("Native disulfide must be an object")
            _labels(row.get("evidence_labels"))
            keys = []
            for label in ("left", "right"):
                point = row.get(label)
                if not isinstance(point, dict):
                    raise ValueError("Disulfide requires two mapped SG endpoints")
                c, pos = point.get("chain"), point.get("position")
                if (not isinstance(c, str) or c not in sequences or type(pos) is not int
                        or not 1 <= pos <= len(sequences[c]) or sequences[c][pos-1] != "C"
                        or point.get("residue") != "C" or point.get("atom") != "SG"):
                    raise ValueError("Disulfide endpoint must be a mapped CYS SG")
                keys.append((c, pos, "SG"))
            if keys[0] == keys[1] or any(k in used for k in keys):
                raise ValueError("Disulfide endpoints must be distinct and singly paired")
            used.update(keys)
            bonds.add(frozenset(keys))
            bond(*keys)
        cysteines = {k for k in self.required if k[2] == "SG"}
        if used != cysteines:
            raise ValueError("Declared native disulfides must cover every CYS exactly once")
        for assignment in assignments:
            mapped = {frozenset((assignment[c], p, a) for c, p, a in pair) for pair in bonds}
            if mapped != bonds:
                raise ValueError("Native topology must be invariant under equivalent-copy assignments")
        self.disulfide_count = len(bonds)
        self.excluded = set()
        # Only interchain exclusions are needed. The complete standard graph
        # supplies the paths through an annotated interchain disulfide.
        for key in graph:
            visited, frontier = {key}, {key}
            for _ in range(3):
                frontier = {n for k in frontier for n in graph[k]}-visited
                visited.update(frontier)
            self.excluded.update(frozenset((key, other)) for other in visited if key[0] != other[0])

    def _read(self, path):
        """Use legacy CA selection and one compatible heavy-atom conformer."""
        residues = {}
        for line in Path(path).read_text().splitlines():
            if line[:6].strip() != "ATOM":
                continue
            # _read_ca has already validated every ATOM's identity, numbering,
            # occupancy, finite coordinates and single-model boundaries.
            c, pos = self.input[line[21]], int(line[22:26])
            name, alt = line[12:16].strip(), line[16]
            element = line[76:78].strip().upper() or name.lstrip("0123456789")[0]
            if element in ("H", "D"):
                if name.lstrip("0123456789")[0] not in ("H", "D"):
                    raise ValueError("Heavy atom mislabeled as hydrogen")
                continue
            key = c, pos, name
            if key not in self.allowed or element != name[0] or element not in _HEAVY_RADII:
                raise ValueError(f"Unsupported heavy atom identity/element: {key}/{element}")
            occupancy = float(line[54:60]) if line[54:60].strip() else 1.
            if occupancy == 0:
                continue
            xyz = tuple(float(line[i:i+8]) for i in (30, 38, 46))
            priority = (-occupancy, 0 if alt == " " else 1 if alt == "A" else 2, alt, xyz)
            records = residues.setdefault((c, pos), {})
            if (name, alt) not in records or priority < records[name, alt][0]:
                records[name, alt] = priority, xyz, occupancy
        atoms = {}
        for (c, pos), records in residues.items():
            ca = [(r[0], alt) for (name, alt), r in records.items() if name == "CA"]
            ca_alt = min(ca)[1] if ca else " "
            alts = sorted({alt for _, alt in records if alt != " "})
            if ca_alt != " ":
                selected = ca_alt
            elif alts:
                def priority(alt):
                    values = [r[2] for (name, a), r in records.items() if a == alt]
                    return -math.fsum(values)/len(values), alt != "A", alt
                selected = min(alts, key=priority)
            else:
                selected = " "
            for name in {name for name, _ in records}:
                row = (records.get((name, ca_alt)) if name == "CA" else
                       records.get((name, selected), records.get((name, " "))))
                if row is not None:
                    atoms[c, pos, name] = row[1]
        return atoms

    def score(self, path):
        atoms = self._read(path)
        missing = self.required-atoms.keys()
        grid, rows, candidate_count, excluded_count = {}, [], 0, 0
        offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
        for key in sorted(atoms):
            if key[2][0] not in self.elements:
                continue
            xyz = atoms[key]
            cell = tuple(math.floor(x/self.cutoff) for x in xyz)
            for offset in offsets:
                neighbor = tuple(a+b for a, b in zip(cell, offset))
                for other in grid.get(neighbor, ()):
                    if key[0] == other[0]:
                        continue
                    candidate_count += 1
                    if frozenset((key, other)) in self.excluded:
                        excluded_count += 1
                        continue
                    distance = math.dist(xyz, atoms[other])
                    if not math.isfinite(distance):
                        raise ValueError("Nonfinite heavy-atom distance")
                    overlap = _HEAVY_RADII[key[2][0]]+_HEAVY_RADII[other[2][0]]-distance
                    excess = max(overlap-self.threshold, 0.)
                    if excess:
                        z = excess/self.scale
                        severity = -math.expm1(-.5*z*z)
                        rows.append({"atoms": [key, other], "distance_angstrom": distance,
                                     "overlap_angstrom": overlap, "severity": severity})
            grid.setdefault(cell, []).append(key)
        total = math.fsum(r["severity"] for r in rows)
        normalized = total/self.pair_scale
        if not math.isfinite(normalized):
            raise ValueError("Nonfinite normalized heavy-atom overlap severity")
        factor = (1. if self.strength == 0 else self.floor if missing else
                  self.floor+(1-self.floor)*math.exp(-self.strength*normalized))
        return {"factor": factor, "overlap_pair_count": len(rows), "severity_sum": total,
                "normalized_overlap": normalized, "strength": self.strength, "factor_floor": self.floor,
                "pair_scale": self.pair_scale, "overlap_threshold_angstrom": self.threshold,
                "overlap_scale_angstrom": self.scale, "elements": sorted(self.elements),
                "required_heavy_atoms": len(self.required), "observed_required_heavy_atoms": len(self.required)-len(missing),
                "missing_heavy_atoms": [list(k) for k in sorted(missing)],
                "status": "unassessed_missing_heavy_atoms_factor_floor" if missing else "assessed_geometric_overlaps",
                "candidate_interchain_pairs": candidate_count, "excluded_topology_candidate_pairs": excluded_count,
                "native_disulfide_count": self.disulfide_count,
                "worst_overlaps": sorted(rows, key=lambda r: (-r["overlap_angstrom"], r["atoms"]))[:20],
                "method": "Interchain geometric overlaps; fixed annotated and standard covalent graph excludes 1–2/1–3/1–4; no distance-inferred bonds or CONECT",
                "interpretation": "Bounded engineering safeguard, not calibrated physical acceptance, energy or MolProbity clashscore"}


class MappedDistanceReward:
    """Freeze a verified specification and score single-model standard PDBs."""

    def __init__(self, fasta_path, spec_json_path):
        fasta_bytes = Path(fasta_path).read_bytes()
        spec_bytes = Path(spec_json_path).read_bytes()
        spec = json.loads(spec_bytes)
        if (not isinstance(spec, dict) or type(spec.get("schema_version")) is not int
                or spec.get("schema_version") not in (1, 2, 3) or spec.get("active") is not True
                or spec.get("mapping_status") != "verified"):
            raise ValueError("Reward spec must be version 1, 2 or 3, active, and mapping verified")
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
        integrity_pairs = spec.get("integrity_pairs", []) if self._version >= 2 else []
        if not isinstance(integrity_pairs, list):
            raise ValueError("integrity_pairs must be a list")
        pairs = pairs + integrity_pairs
        for pair_index, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                raise ValueError("Each distance pair must be an object")
            name = pair.get("id")
            if not isinstance(name, str) or not name.strip() or name in ids:
                raise ValueError("Pair IDs must be nonempty and unique")
            ids.add(name)
            if "scope" in pair and (pair["scope"] != "intra_receptor"
                                    or self._version < 2 or pair_index >= self._interface_count):
                raise ValueError("Pair scope must be intra_receptor on a version-2/3 interface-group pair")
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
            feature = pair.get("distance_feature", "gaussian")
            if feature not in ("gaussian", "cauchy") or (feature != "gaussian" and self._version < 2):
                raise ValueError("distance_feature must be gaussian or version-2/3 cauchy")
            self._pairs.append({"id": name, "endpoints": tuple(endpoints),
                                "distance": _positive(pair.get("distance_angstrom"), "distance"),
                                "sigma": _positive(pair.get("sigma_angstrom"), "sigma"),
                                "weight": _positive(pair.get("weight"), "weight"),
                                "distance_feature": feature,
                                **({"scope": pair["scope"]} if "scope" in pair else {}),
                                "evidence_labels": _labels(pair.get("evidence_labels"))})
        try:
            self._total_weight = math.fsum(p["weight"] for p in self._pairs)
        except OverflowError as exc:
            raise ValueError("Total pair weight must be finite") from exc
        if self._version >= 2:
            self._configure_joint(spec)
        self._geometry_modules = []
        if self._version == 3:
            self._geometry_modules = validate_local_geometry_modules(
                spec.get("local_geometry_modules"), self._sequences, self._interfaces)
        elif "local_geometry_modules" in spec:
            raise ValueError("Local geometry modules require schema version 3")
        self._heavy_guard = None
        if "heavy_atom_safeguard" in spec:
            if self._version < 2:
                raise ValueError("heavy_atom_safeguard requires schema version 2 or 3")
            self._heavy_guard = _HeavyAtomSafeguard(spec["heavy_atom_safeguard"], self._sequences,
                                                  self._input, self._assignments)
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
        if self._version >= 2:
            self.metadata.update({
                "score_definition": "((1-minimum_fraction)*weighted_mean_interface + minimum_fraction*weakest_interface) * safeguards * integrity_factor",
                "interface_pair_count": self._interface_count,
                "integrity_pair_count": len(integrity_pairs),
                "integrity_definition": "floor + (1-floor)*minimum_integrity_pair_Gaussian",
                "safeguard_threshold_status": "Engineering settings, not calibrated acceptance thresholds"})
        if self._version == 3:
            self.metadata.update({
                "kind": "mapped_ca_distance_with_local_geometry",
                "score_definition": self.metadata["score_definition"] + " * product(local_geometry_module_factors)",
                "local_geometry_definition": "Per module: floor+(1-floor)*exp(-strength*maximum_copy_error); copy error = distance flat-tolerance Huber + mean of three angle flat-tolerance Hubers. Multiply module factors into the CA reward under the same assignment.",
                "local_geometry_missing_policy": "Any absent or degenerate measurement gives at least missing_geometry_error; three-angle denominator remains fixed.",
                "local_geometry_altloc_policy": "Use the legacy selected CA conformer if nonblank; otherwise choose one residue conformer by mean required-atom occupancy, then A/lexical. Blank atoms are shared; incompatible alternate conformers are never mixed within local geometry.",
                "limited_scope": "Sparse CA and local heavy-atom geometry priors only; not all-atom packing, explicit-hydrogen chemistry, physical energy, affinity, or independent structural accuracy."})
        if any(p["distance_feature"] == "cauchy" for p in self._pairs):
            self.metadata["distance_feature_definitions"] = {
                "gaussian": "exp(-0.5*((distance-target)/sigma)^2)",
                "cauchy": "1/(1+((distance-target)/sigma)^2); sigma is the half-maximum distance error"}
        if self._version >= 2 and any("required_subgroups" in g for g in self._interfaces):
            self.metadata["required_subgroup_definition"] = "Each interface is the minimum of its complete, disjoint subgroups' fixed-denominator weighted means under one assignment"
        if any(p.get("scope") == "intra_receptor" for p in self._pairs):
            self.metadata["intra_receptor_pair_definition"] = "Explicit intra_receptor pairs require both endpoints in their interface group's receptor and share its coherent assignment; internal shape spans are not intermolecular contacts"
        if self._heavy_guard is not None:
            self.metadata.update({
                "score_definition": self.metadata["score_definition"]+" * heavy_atom_safeguard_factor",
                "heavy_atom_safeguard_definition": "floor+(1-floor)*exp(-strength*sum(1-exp(-0.5*(max(overlap-threshold,0)/scale)^2))/pair_scale); missing required heavy atoms give floor; strength zero gives exactly one",
                "heavy_atom_topology": "Standard atom-name and peptide bonds plus explicit symmetric native disulfides; no coordinate inference or CONECT; exclude 1–2/1–3/1–4",
                "limited_scope": self.metadata["limited_scope"]+" Includes a bounded interchain heavy-atom overlap safeguard, not calibrated physical validation."})

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
            for member in members:
                pair = expected[member]
                receptor_endpoints = sum(c == receptor for c, _ in pair["endpoints"])
                if pair.get("scope") == "intra_receptor":
                    if receptor_endpoints != 2:
                        raise ValueError("An intra_receptor pair must have both endpoints in its group's receptor")
                elif receptor_endpoints != 1:
                    raise ValueError("Each interface pair must touch its receptor exactly once")
            seen.update(members)
            group_ids.add(name)
            configured = dict(id=name, receptor_chain=receptor, type=group["type"],
                              pair_ids=list(members), weight=_positive(group.get("weight"), "interface weight"))
            if "required_subgroups" in group:
                subgroups = group["required_subgroups"]
                if not isinstance(subgroups, list) or len(subgroups) < 2:
                    raise ValueError("required_subgroups must contain at least two disjoint groups")
                subgroup_ids, covered, configured_subgroups = set(), set(), []
                for subgroup in subgroups:
                    if not isinstance(subgroup, dict):
                        raise ValueError("Required subgroup must be an object")
                    sid, ids = subgroup.get("id"), subgroup.get("pair_ids")
                    if (not isinstance(sid, str) or not sid or sid in subgroup_ids
                            or not isinstance(ids, list) or not ids
                            or any(not isinstance(p, str) or p not in members for p in ids)
                            or len(set(ids)) != len(ids) or covered.intersection(ids)):
                        raise ValueError("Invalid, duplicate, overlapping or unknown required subgroup")
                    subgroup_ids.add(sid)
                    covered.update(ids)
                    configured_subgroups.append({"id": sid, "pair_ids": list(ids)})
                if covered != set(members):
                    raise ValueError("required_subgroups must partition every interface pair exactly once")
                configured["required_subgroups"] = configured_subgroups
            self._interfaces.append(configured)
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
            extra = {}
            if "required_subgroups" in group:
                subgroups = []
                for subgroup in group["required_subgroups"]:
                    selected = [by_id[p] for p in subgroup["pair_ids"]]
                    score = math.fsum(p["weight"]*p["feature"] for p in selected)/math.fsum(p["weight"] for p in selected)
                    subgroups.append({**subgroup, "score": score, "present_pairs": sum(p["present"] for p in selected),
                                      "total_pairs": len(selected)})
                extra = {"unpartitioned_weighted_mean": value, "subgroup_scores": subgroups}
                value = min(s["score"] for s in subgroups)
            copies.append({**group, "score": value, **extra,
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
        geometry_atoms, conformers = read_local_geometry_atoms(pdb_path, self._sequences, self._input, self._geometry_modules) if self._version == 3 else ({}, [])
        overlaps = self._overlaps(atoms, self._guard["ca_clash_distance_angstrom"] if self._version >= 2 else None)
        physical_guard = self._physical_guard(atoms, overlaps) if self._version >= 2 else None
        heavy_guard = self._heavy_guard.score(pdb_path) if self._heavy_guard is not None else None
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
                    value = ((1/math.hypot(1., z))**2 if pair["distance_feature"] == "cauchy"
                             else math.exp(-0.5 * z * z))
                rows.append({"id": pair["id"], "feature": value,
                             "distance_angstrom": distance, "present": present,
                             "weight": pair["weight"],
                             **({"scope": pair["scope"]} if "scope" in pair else {}),
                             "missing_atoms": [{"input_chain": chain, "saved_chain": self._saved[chain],
                                                "position": pos, "atom": "CA"}
                                               for chain, pos in keys if (chain, pos) not in atoms]})
            joint = self._joint_value(rows, physical_guard) if self._version >= 2 else {}
            reward = joint.get("reward", math.fsum(p["weight"] * p["feature"] for p in rows) / self._total_weight)
            if self._version == 3:
                geometry = local_geometry_results(self._geometry_modules, geometry_atoms, assignment)
                factor = math.prod(m["factor"] for m in geometry)
                joint.update({"ca_reward_under_selected_assignment": reward,
                              "local_geometry": geometry, "local_geometry_factor": factor,
                              "local_geometry_conformers": conformers, "reward": reward*factor})
                reward = joint["reward"]
            if heavy_guard is not None:
                joint.update({"reward_before_heavy_atom_safeguard": reward,
                              "heavy_atom_safeguard": heavy_guard, "reward": reward*heavy_guard["factor"]})
                reward = joint["reward"]
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
