"""Soft hotspot steering from the four experimental chains of 2H62.

The second receptor sites transfer local motifs by exchanging ligand labels;
they do not target the clashing six-chain symmetry-copy model. The reward is an
experimental ranking objective, not affinity, confidence, or physical energy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import itertools
import json
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree

try:
    from .analyze_complex_ensemble import read_prediction
    from .compare_structures import kabsch, reference_ca
    from .complex_interfaces import _contacts
    from .structure_clashes import RADII, SIDECHAINS, _read
except ImportError:
    from analyze_complex_ensemble import read_prediction
    from compare_structures import kabsch, reference_ca
    from complex_interfaces import _contacts
    from structure_clashes import RADII, SIDECHAINS, _read

DEFAULT_SETTINGS = {
    "positive_weights": {"W60_packing": 2., "Y42_engagement": 1., "Q86_geometry": 2.},
    "penalties": {
        "BMP2_dimer_CA_RMSD_angstrom": {"weight": .5, "scale": 2., "cap": 4.},
        "observed_receptor_placement_CA_RMSD_angstrom": {"weight": .5, "scale": 10., "cap": 4.},
        "clashes_per_1000_heavy_atoms": {"weight": .5, "scale": 50., "cap": 4.},
        "covalent_bond_strain": {"weight": .5, "scale": 1., "cap": 4.},
    },
    "distance_tolerance_angstrom": .75, "orientation_tolerance_degrees": 20.,
    "bond_tolerance_angstrom": .1, "clash_overlap_threshold_angstrom": .4,
}
COMPONENTS = {"A": "BMP2", "B": "BMP2", "C": "BMPR1A", "D": "BMPR1A", "E": "ACVR2B", "F": "ACVR2B"}
W60_RING = ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2")
Y42_RING = ("CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH")


def window_score(value, target, tolerance):
    """Native-centered flat window with Gaussian decay on either side."""
    if value is None:
        return 0.
    deviation = max(abs(value - target) - tolerance, 0.) / tolerance
    return float(np.exp(-.5 * deviation * deviation))


def angle(first, vertex, third):
    a, b = first - vertex, third - vertex
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-12:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denominator, -1., 1.))))


def atom_records(path, chains):
    """Coherent occupied alternate conformer, keyed by full sequence position."""
    atoms, _, _, omitted = _read(path)
    authors = {c: dict(zip(record.author_residues, record.full_positions)) for c, record in chains.items()}
    records = []
    for atom in atoms:
        author = str(atom.number) + atom.insertion
        if atom.chain in authors and author in authors[atom.chain]:
            records.append(replace(atom, number=authors[atom.chain][author] + 1,
                                   insertion="", segment=ord(atom.chain)))
    return records, {(a.chain, a.number - 1, a.name): np.array(a.xyz) for a in records}, omitted


def chemical_bonds(atoms, disulfides):
    """Atom-name protein topology plus fixed annotated disulfides; no SS inference."""
    residues = defaultdict(dict)
    for index, atom in enumerate(atoms):
        residues[(atom.chain, atom.number)][atom.name] = index
    pairs = {}
    for (chain, number), names in residues.items():
        residue = atoms[next(iter(names.values()))].residue
        for pair in ("N-CA CA-C C-O C-OXT " + SIDECHAINS.get(residue, "")).split():
            first, second = pair.split("-")
            if first in names and second in names:
                pairs[tuple(sorted((names[first], names[second])))] = (residue, *sorted((first, second)))
        following = residues.get((chain, number + 1), {})
        if "C" in names and "N" in following:
            pairs[tuple(sorted((names["C"], following["N"])))] = ("peptide", "C", "N")
    sulfur = {(a.chain, a.number - 1): i for i, a in enumerate(atoms) if a.residue == "CYS" and a.name == "SG"}
    for first, second in disulfides:
        if first in sulfur and second in sulfur:
            pairs[tuple(sorted((sulfur[first], sulfur[second])))] = ("disulfide", "SG", "SG")
    return pairs


def standard_bond_target(kind, atoms, pair):
    """Fallback chemical lengths in Å, only when a bond type is unobserved."""
    if kind[0] == "disulfide":
        return 2.05
    if kind[0] == "peptide":
        return 1.329
    names = set(kind[1:])
    if names == {"N", "CA"}: return 1.458
    if names == {"CA", "C"}: return 1.525
    if names == {"C", "O"}: return 1.231
    if names == {"C", "OXT"}: return 1.25
    elements = {atoms[i].element for i in pair}
    if "S" in elements or "SE" in elements: return 1.81
    if "O" in elements:
        return 1.25 if any(name.startswith(("OD", "OE")) for name in names) else 1.36 if kind[0] == "TYR" else 1.43
    aromatic = kind[0] in ("PHE", "TYR", "TRP", "HIS") and not names.intersection(("CA", "CB"))
    if "N" in elements:
        amide = kind[0] in ("ASN", "GLN", "ARG") and any(name in ("ND2", "NE2", "CZ") for name in names)
        return 1.34 if aromatic or amide else 1.47
    return 1.395 if aromatic else 1.53


def fixed_geometry(atoms, disulfides, targets, tolerance=.1, overlap_threshold=.4):
    """Evaluate covalent lengths and nonbonded overlaps with the same fixed graph."""
    bonds = chemical_bonds(atoms, disulfides)
    xyz = np.asarray([a.xyz for a in atoms]).reshape((-1, 3))
    neighbors = [set() for _ in atoms]
    strain = []
    for (i, j), kind in bonds.items():
        neighbors[i].add(j); neighbors[j].add(i)
        target = 2.05 if kind[0] == "disulfide" else targets.get(kind, standard_bond_target(kind, atoms, (i, j)))
        distance = float(np.linalg.norm(xyz[i] - xyz[j]))
        excess = max(abs(distance - target) - tolerance, 0.) / tolerance
        strain.append({"atoms": [atoms[i].label, atoms[j].label], "kind": ":".join(kind),
                       "distance_angstrom": distance, "target_angstrom": target,
                       "normalized_excess": excess,
                       "target_source": "fixed native disulfide chemistry" if kind[0] == "disulfide"
                       else "empirical 2H62 bond-type average" if kind in targets else "standard chemical fallback"})
    excluded = set()
    for i in range(len(atoms)):
        visited, frontier = {i}, {i}
        for _ in range(3):
            frontier = {k for j in frontier for k in neighbors[j]} - visited
            visited.update(frontier)
        excluded.update(tuple(sorted((i, j))) for j in visited if i != j)
    clashes = []
    for i, j in cKDTree(xyz).query_pairs(2 * max(RADII.values()) - overlap_threshold):
        if (i, j) in excluded:
            continue
        distance = float(np.linalg.norm(xyz[i] - xyz[j]))
        overlap = RADII[atoms[i].element] + RADII[atoms[j].element] - distance
        if overlap > overlap_threshold:
            clashes.append({"atoms": [atoms[i].label, atoms[j].label], "distance_angstrom": distance,
                            "overlap_angstrom": overlap, "interchain": atoms[i].chain != atoms[j].chain})
    strain.sort(key=lambda row: row["normalized_excess"], reverse=True)
    worst = np.array([row["normalized_excess"] for row in strain[:10]])
    return {
        "clashes_per_1000_heavy_atoms": 1000 * len(clashes) / len(atoms) if atoms else 0.,
        "covalent_bond_strain": float(np.sqrt(np.mean(worst ** 2))) if len(worst) else 0.,
        "heavy_atom_count": len(atoms), "clash_count": len(clashes),
        "interchain_clash_count": sum(row["interchain"] for row in clashes),
        "worst_clashes": sorted(clashes, key=lambda row: -row["overlap_angstrom"])[:20],
        "bond_count": len(bonds), "worst_bonds": strain[:20],
        "evaluated_disulfide_count": sum(kind[0] == "disulfide" for kind in bonds.values()),
        "methods": {"topology": "Standard amino-acid atom-name bonds, adjacent-residue peptide bonds, fixed explicit native disulfides; no geometry-dependent disulfide inference. Excludes nonbonded 1–2/1–3/1–4.",
                    "bond_strain": "RMS of the ten largest bond-length deviations outside the declared tolerance, divided by tolerance; dimensionless ranking diagnostic, not physical energy.",
                    "clashes": "Occupied protein heavy-atom van der Waals overlaps; geometric count, not MolProbity clashscore."},
    }


class PartialComplexReward:
    def __init__(self, reference_cif, reference_pdb, settings=None):
        self.settings = dict(DEFAULT_SETTINGS)
        if settings:
            self.settings.update(settings)
        self.references = {c: reference_ca(reference_cif, c) for c in "ABCD"}
        self.prediction_references = dict(self.references, E=self.references["B"], F=self.references["C"])
        self.reference_records, self.native, _ = atom_records(reference_pdb, self.references)
        self.motifs = {}
        self.motifs["W60_packing"] = self._packing("C", 59, W60_RING, (33, 34, 87, 89, 99))
        self.motifs["Y42_engagement"] = self._packing("C", 41, Y42_RING, None)
        q = lambda name: ("B", 85, name)
        l = lambda name: ("D", 50, name)
        self.q_distances = [(q("OE1"), l("N")), (q("NE2"), l("O"))]
        self.q_angles = [(q("CD"), q("OE1"), l("N")), (q("CD"), q("NE2"), l("O")),
                         (l("C"), l("O"), q("NE2")), (l("CA"), l("N"), q("OE1"))]
        self.q_targets = [float(np.linalg.norm(self.native[a] - self.native[b])) for a, b in self.q_distances]
        self.q_angle_targets = [angle(*(self.native[key] for key in keys)) for keys in self.q_angles]
        self.fixed_disulfides = self._disulfides(reference_pdb)
        lengths = defaultdict(list)
        for pair, kind in chemical_bonds(self.reference_records, []).items():
            lengths[kind].append(np.linalg.norm(np.array(self.reference_records[pair[0]].xyz) - self.reference_records[pair[1]].xyz))
        self.bond_targets = {kind: float(np.mean(values)) for kind, values in lengths.items()}
        self.native_contacts = {c: _contacts(self.native, c, ("A", "D")) for c in "BC"}
        self.metadata = {
            "reward_kind": "2h62_partial", "reference_policy": "Only experimental observed.cif/pdb; no symmetry_completion coordinates used.",
            "unobserved_sites": "Local motifs inferred by swapping BOTH ligand identities; no invented receptor coordinates or experimental-contact claims.",
            "assignment": "Enumerate both ligand swaps and both receptor permutations per type; one coherent bijection assigns each receptor to a distinct ligand-derived site. Choose the highest total declared reward.",
            "distance_policy": "Native-centered flat window with Gaussian decay on either side; full native geometry scores 1 including native distances >4 Å.",
            "Q86_policy": "Both complementary heavy-atom distances and four native-centered approach angles required. This is heavy-atom geometry, not explicit-hydrogen hydrogen-bond validation.",
            "positive_maximum": sum(self.settings["positive_weights"].values()),
            "motifs": self.motifs, "Q86_native_distances_angstrom": self.q_targets,
            "Q86_native_approach_angles_degrees": self.q_angle_targets,
            "fixed_disulfides": self.fixed_disulfides,
            "bond_target_provenance": "Empirical occupied 2H62 bond-type mean lengths by residue and atom names; peptide mean pooled separately; standard chemical fallback for types absent from observed reference, disulfides fixed at 2.05 Å. Not a force field or CCD-derived parameterization.",
            "bond_targets_angstrom": {":".join(kind): value for kind, value in self.bond_targets.items()},
        }

    def _packing(self, receptor, position, ring, pocket):
        rec_keys = [key for key in self.native if key[0] == receptor and key[1] == position and key[2] in ring]
        residues = sorted({(c, p) for c, p, _ in self.native if c in ("A", "D") and (pocket is None or p in pocket)})
        candidates = []
        for chain, p in residues:
            lig_keys = [key for key in self.native if key[:2] == (chain, p)]
            distance = min(float(np.linalg.norm(self.native[a] - self.native[b])) for a in rec_keys for b in lig_keys)
            candidates.append(dict(ligand_chain=chain, ligand_position=p, target_distance_angstrom=distance,
                                   receptor_atoms=[list(key) for key in rec_keys], ligand_atoms=[list(key) for key in lig_keys]))
        if pocket is not None:
            return [min((row for row in candidates if row["ligand_position"] == p), key=lambda row: row["target_distance_angstrom"]) for p in pocket]
        return [row for row in candidates if row["target_distance_angstrom"] <= 4.5]

    def _disulfides(self, path):
        structure = gemmi.read_structure(str(path))
        positions = {c: dict(zip(record.author_residues, record.full_positions)) for c, record in self.references.items()}
        result = set()
        copies = {"A": ("A", "B"), "D": ("A", "B"), "B": ("C", "D"), "C": ("E", "F")}
        for connection in structure.connections:
            if connection.type != gemmi.ConnectionType.Disulf:
                continue
            a, b = connection.partner1, connection.partner2
            if a.chain_name != b.chain_name or a.chain_name not in positions:
                continue
            first = positions[a.chain_name][str(a.res_id.seqid)]
            second = positions[b.chain_name][str(b.res_id.seqid)]
            for copy in copies[a.chain_name]:
                result.add(tuple(sorted(((copy, first), (copy, second)))))
        # Conserved native BMP2 homodimer Cys78, independently of SG separation.
        result.add((("A", 77), ("B", 77)))
        return sorted(result)

    @staticmethod
    def _key(key, receptor, ligand_mapping, site):
        chain, position, name = key
        if chain in ("A", "D"):
            if site:
                chain = "D" if chain == "A" else "A"
            chain = ligand_mapping[chain]
        else:
            chain = receptor
        return chain, position, name

    def _copy_features(self, atoms, receptor, ligand_mapping, site, component):
        convert = lambda key: self._key(key, receptor, ligand_mapping, site)
        result, details = {}, {}
        for feature in (("W60_packing", "Y42_engagement") if component == "ACVR2B" else ()):
            rows = []
            for motif in self.motifs[feature]:
                rec = [convert(key) for key in motif["receptor_atoms"]]
                lig = [convert(key) for key in motif["ligand_atoms"]]
                present_rec, present_lig = [key for key in rec if key in atoms], [key for key in lig if key in atoms]
                distance = min((float(np.linalg.norm(atoms[a] - atoms[b])) for a in present_rec for b in present_lig), default=None)
                coverage = (len(present_rec) + len(present_lig)) / (len(rec) + len(lig))
                score = window_score(distance, motif["target_distance_angstrom"], self.settings["distance_tolerance_angstrom"]) * coverage
                rows.append(dict(prediction_ligand_chain=convert((motif["ligand_chain"], 0, "CA"))[0],
                                 ligand_position_one_based=motif["ligand_position"] + 1,
                                 target_distance_angstrom=motif["target_distance_angstrom"], distance_angstrom=distance,
                                 atom_coverage=coverage, score=score))
            result[feature] = float(np.mean([row["score"] for row in rows])) if rows else 0.
            details[feature] = rows
        if component == "BMPR1A":
            distances, angles = [], []
            for pair, target in zip(self.q_distances, self.q_targets):
                keys = [convert(key) for key in pair]
                distance = float(np.linalg.norm(atoms[keys[0]] - atoms[keys[1]])) if all(key in atoms for key in keys) else None
                distances.append(dict(atom_keys=keys, distance_angstrom=distance, target_angstrom=target,
                                      score=window_score(distance, target, self.settings["distance_tolerance_angstrom"])))
            for triplet, target in zip(self.q_angles, self.q_angle_targets):
                keys = [convert(key) for key in triplet]
                value = angle(*(atoms[key] for key in keys)) if all(key in atoms for key in keys) else None
                angles.append(dict(atom_keys=keys, angle_degrees=value, target_degrees=target,
                                   score=window_score(value, target, self.settings["orientation_tolerance_degrees"])))
            result["Q86_geometry"] = min(row["score"] for row in distances) * min(row["score"] for row in angles)
            details["Q86_geometry"] = dict(distances=distances, approach_angles=angles)
        return dict(component=component, prediction_chain=receptor, site=site,
                    site_evidence="experimental local motif" if site == 0 else "inferred ligand-swapped local motif",
                    features=result, details=details)

    def _assignment_features(self, copies):
        return {key: min(copy["features"][key] for copy in copies if key in copy["features"])
                for key in self.settings["positive_weights"]}

    def _penalty_contributions(self, features):
        result = {}
        for key, spec in self.settings["penalties"].items():
            value = features[key] / spec["scale"]
            if spec["cap"] is not None:
                value = min(value, spec["cap"])
            result[key] = -spec["weight"] * value
        return result

    def score(self, path):
        chains = read_prediction(path, self.prediction_references)
        records, atoms, omitted = atom_records(path, chains)
        geometry = fixed_geometry(records, self.fixed_disulfides, self.bond_targets,
                                  self.settings["bond_tolerance_angstrom"], self.settings["clash_overlap_threshold_angstrom"])
        fixed = np.concatenate([self.references[c].coordinates for c in ("A", "D")])
        candidates = []
        for ligand_order in (("A", "B"), ("B", "A")):
            ligand = dict(zip(("A", "D"), ligand_order))
            moving = np.concatenate([chains[ligand[c]].coordinates[self.references[c].full_positions] for c in ("A", "D")])
            rotation, translation, errors = kabsch(moving, fixed)
            local = {(component, pc, site): self._copy_features(atoms, pc, ligand, site, component)
                     for component, copies in (("BMPR1A", "CD"), ("ACVR2B", "EF")) for pc in copies for site in (0, 1)}
            for bmpr, acvr in itertools.product(itertools.permutations("CD"), itertools.permutations("EF")):
                copies = [local[component, pc, site] for component, order in (("BMPR1A", bmpr), ("ACVR2B", acvr)) for site, pc in enumerate(order)]
                mapping = dict(ligand, B=bmpr[0], C=acvr[0])
                placement = {rc: float(np.sqrt(np.mean(np.sum((chains[mapping[rc]].coordinates[self.references[rc].full_positions] @ rotation + translation - self.references[rc].coordinates) ** 2, axis=1)))) for rc in "BC"}
                features = self._assignment_features(copies)
                features.update(BMP2_dimer_CA_RMSD_angstrom=float(np.sqrt(np.mean(errors ** 2))),
                                observed_receptor_placement_CA_RMSD_angstrom=max(placement.values()),
                                clashes_per_1000_heavy_atoms=geometry["clashes_per_1000_heavy_atoms"],
                                covalent_bond_strain=geometry["covalent_bond_strain"])
                contributions = {key: weight * features[key] for key, weight in self.settings["positive_weights"].items()}
                contributions.update(self._penalty_contributions(features))
                candidates.append(dict(reward=sum(contributions.values()), features=features, contributions=contributions,
                                       assignment=dict(experimental_reference_to_prediction=mapping,
                                                       BMPR1A_site_to_prediction=list(bmpr), ACVR2B_site_to_prediction=list(acvr)),
                                       copy_features=copies, observed_receptor_placement_CA_RMSD_angstrom=placement))
        best = max(candidates, key=lambda row: row["reward"])
        mapping = best["assignment"]["experimental_reference_to_prediction"]
        paired = {key: atoms[(mapping[key[0]], key[1], key[2])] for key in self.native if (mapping[key[0]], key[1], key[2]) in atoms}
        interfaces = []
        for rc, component in (("B", "BMPR1A"), ("C", "ACVR2B")):
            native = self.native_contacts[rc]
            predicted = _contacts(paired, rc, ("A", "D"))
            recovery = len(native & predicted)
            interfaces.append(dict(component=component, reference_chain=rc, prediction_chain=mapping[rc],
                                   native_contacts=len(native), predicted_contacts=len(predicted), recovered_contacts=recovery,
                                   native_contact_recall=recovery / len(native) if native else 0.,
                                   native_contact_precision=recovery / len(predicted) if predicted else 0.))
        best.update(geometry=geometry, interfaces=interfaces,
                    coverage=dict(matched_experimental_heavy_atoms=len(paired), experimental_heavy_atoms=len(self.native),
                                  experimental_heavy_atom_fraction=len(paired) / len(self.native), omitted_atoms=omitted),
                    assignment_candidates=[dict(reward=row["reward"], assignment=row["assignment"]) for row in candidates])
        return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cif", type=Path, required=True)
    parser.add_argument("--reference-pdb", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--settings-json", type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings_json.read_text()) if args.settings_json else None
    reward = PartialComplexReward(args.reference_cif, args.reference_pdb, settings)
    rows = [dict(source=str(path), **reward.score(path)) for path in sorted(args.prediction_dir.glob("*.pdb"))]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(dict(metadata=reward.metadata, samples=rows), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
