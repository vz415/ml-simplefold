#!/usr/bin/env python3
"""Coordinate-only protein heavy-atom overlaps; this is not MolProbity clashscore.

Standard amino-acid connectivity defines covalent 1–2, 1–3 and 1–4 exclusions.
Hydrogens, HETATM records, occupancy-zero atoms, and alternative conformers
other than the highest-occupancy conformer are omitted. No coordinates change.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80,
         'SE': 1.90, 'F': 1.47, 'CL': 1.75, 'BR': 1.85, 'I': 1.98}
# Chemical atom-name topology, independent of coordinate distances.
SIDECHAINS = {
    'GLY': '', 'ALA': 'CA-CB', 'ARG': 'CA-CB CB-CG CG-CD CD-NE NE-CZ CZ-NH1 CZ-NH2',
    'ASN': 'CA-CB CB-CG CG-OD1 CG-ND2', 'ASP': 'CA-CB CB-CG CG-OD1 CG-OD2',
    'CYS': 'CA-CB CB-SG', 'GLN': 'CA-CB CB-CG CG-CD CD-OE1 CD-NE2',
    'GLU': 'CA-CB CB-CG CG-CD CD-OE1 CD-OE2',
    'HIS': 'CA-CB CB-CG CG-ND1 ND1-CE1 CE1-NE2 NE2-CD2 CD2-CG',
    'ILE': 'CA-CB CB-CG1 CB-CG2 CG1-CD1', 'LEU': 'CA-CB CB-CG CG-CD1 CG-CD2',
    'LYS': 'CA-CB CB-CG CG-CD CD-CE CE-NZ', 'MET': 'CA-CB CB-CG CG-SD SD-CE',
    'PHE': 'CA-CB CB-CG CG-CD1 CG-CD2 CD1-CE1 CD2-CE2 CE1-CZ CE2-CZ',
    'PRO': 'CA-CB CB-CG CG-CD CD-N', 'SER': 'CA-CB CB-OG',
    'THR': 'CA-CB CB-OG1 CB-CG2',
    'TRP': 'CA-CB CB-CG CG-CD1 CG-CD2 CD1-NE1 NE1-CE2 CE2-CD2 CD2-CE3 CE3-CZ3 CZ3-CH2 CH2-CZ2 CZ2-CE2',
    'TYR': 'CA-CB CB-CG CG-CD1 CG-CD2 CD1-CE1 CD2-CE2 CE1-CZ CE2-CZ CZ-OH',
    'VAL': 'CA-CB CB-CG1 CB-CG2', 'MSE': 'CA-CB CB-CG CG-SE SE-CE',
}
DEFAULT_COMPONENTS = {'A': 'BMP2', 'B': 'BMP2', 'C': 'BMPR1A',
                      'D': 'BMPR1A', 'E': 'ActRIIA', 'F': 'ActRIIA'}


@dataclass(frozen=True)
class Atom:
    serial: int
    chain: str
    number: int
    insertion: str
    residue: str
    name: str
    element: str
    xyz: tuple[float, float, float]
    occupancy: float
    alternate: str
    segment: int

    @property
    def residue_key(self):
        return self.segment, self.chain, self.number, self.insertion

    @property
    def label(self):
        return f'{self.chain}:{self.number}{self.insertion}:{self.residue}:{self.name}'


def _read(path):
    candidates = defaultdict(list)
    conect, ssbonds = [], []
    lines = Path(path).read_text().splitlines()
    # Connectivity records may follow ENDMDL; collect metadata independently.
    for line in lines:
        record = line[:6].strip()
        if record == 'CONECT':
            numbers = [int(line[p:p + 5]) for p in range(6, len(line), 5) if line[p:p + 5].strip()]
            if numbers:
                conect.extend((numbers[0], value) for value in numbers[1:])
        elif record == 'SSBOND':
            ssbonds.append(((line[15:16].strip(), int(line[17:21]), line[21:22].strip()),
                            (line[29:30].strip(), int(line[31:35]), line[35:36].strip())))
    segment = 0
    model_seen = False
    omitted = Counter()
    for line in lines:
        record = line[:6].strip()
        if record == 'MODEL':
            if model_seen:
                break
            model_seen = True
        elif record == 'ENDMDL':
            break
        elif record == 'TER':
            segment += 1
        elif record == 'ATOM':
            name = line[12:16].strip()
            element = line[76:78].strip().upper() or name.lstrip('0123456789')[0].upper()
            if element in {'H', 'D'}:
                omitted['hydrogen'] += 1
                continue
            if element not in RADII:
                raise ValueError(f'Unsupported element {element!r} for {name} in {path}')
            occupancy = float(line[54:60].strip() or '1')
            if occupancy <= 0:
                omitted['zero_occupancy'] += 1
                continue
            atom = Atom(int(line[6:11]), line[21:22].strip(), int(line[22:26]),
                        line[26:27].strip(), line[17:20].strip(), name, element,
                        tuple(float(line[p:p + 8]) for p in (30, 38, 46)), occupancy,
                        line[16:17].strip(), segment)
            candidates[(atom.residue_key, name)].append(atom)
    # Choose a coherent alternate residue conformer by total occupancy;
    # shared blank-location atoms take priority over alternate copies.
    alternate_scores = defaultdict(Counter)
    for group in candidates.values():
        for atom in group:
            if atom.alternate:
                alternate_scores[atom.residue_key][atom.alternate] += atom.occupancy
    chosen = {key: sorted(scores, key=lambda alt: (-scores[alt], alt != 'A', alt))[0]
              for key, scores in alternate_scores.items()}
    atoms = []
    for group in candidates.values():
        eligible = [a for a in group if not a.alternate or a.alternate == chosen.get(a.residue_key)]
        if eligible:
            atoms.append(sorted(eligible, key=lambda a: (bool(a.alternate), -a.occupancy, a.serial))[0])
        omitted['alternate_or_duplicate'] += len(group) - bool(eligible)
    return atoms, conect, ssbonds, dict(omitted)


def _consecutive(first, second):
    if first[0:2] != second[0:2]:
        return False
    if second[2] == first[2] + 1 and not second[3]:
        return True
    if second[2] == first[2] and second[3]:
        return (not first[3] and second[3] == 'A') or ord(second[3]) == ord(first[3] or '@') + 1
    return False


def _topology(atoms, conect, ssbonds, include_1_4=False):
    neighbors = [set() for _ in atoms]
    residues = defaultdict(dict)
    serials = {atom.serial: i for i, atom in enumerate(atoms)}
    unknown_residues = set()
    assumed_disulfides = []

    def bond(i, j):
        if i != j:
            neighbors[i].add(j)
            neighbors[j].add(i)

    for i, atom in enumerate(atoms):
        residues[atom.residue_key][atom.name] = i
    for key, names in residues.items():
        residue = atoms[next(iter(names.values()))].residue
        if residue not in SIDECHAINS:
            unknown_residues.add(residue)
        template = 'N-CA CA-C C-O C-OXT ' + SIDECHAINS.get(residue, '')
        for pair in template.split():
            a, b = pair.split('-')
            if a in names and b in names:
                bond(names[a], names[b])
    keys = sorted(residues)
    peptide_residue_pairs = set()
    for first, second in zip(keys, keys[1:]):
        if _consecutive(first, second):
            peptide_residue_pairs.add(frozenset((first, second)))
            if 'C' in residues[first] and 'N' in residues[second]:
                bond(residues[first]['C'], residues[second]['N'])
    for a, b in conect:
        if a in serials and b in serials:
            bond(serials[a], serials[b])
    sulfurs = [i for i, a in enumerate(atoms) if a.residue == 'CYS' and a.name == 'SG']
    sulfur_ids = {(atoms[i].chain, atoms[i].number, atoms[i].insertion): i for i in sulfurs}
    explicit_disulfides = []
    for first, second in ssbonds:
        if first in sulfur_ids and second in sulfur_ids:
            i, j = sulfur_ids[first], sulfur_ids[second]
            bond(i, j)
            explicit_disulfides.append([atoms[i].label, atoms[j].label])
    # Standard inference only for chemically plausible S–S distances, never
    # a general distance-based exclusion of overlapping atoms.
    plausible_sulfur_pairs = []
    sulfur_partners = Counter()
    for offset, i in enumerate(sulfurs):
        for j in sulfurs[offset + 1:]:
            if j in neighbors[i]:
                continue
            distance = float(np.linalg.norm(np.asarray(atoms[i].xyz) - atoms[j].xyz))
            if 1.8 <= distance <= 2.3:
                plausible_sulfur_pairs.append((i, j, distance))
                sulfur_partners[i] += 1
                sulfur_partners[j] += 1
    for i, j, distance in plausible_sulfur_pairs:
        # Ambiguous sulfur clusters must not be hidden as a covalent network.
        if sulfur_partners[i] != 1 or sulfur_partners[j] != 1:
            continue
        if any(atoms[k].name == 'SG' for k in neighbors[i] | neighbors[j]):
            continue
        bond(i, j)
        assumed_disulfides.append({'atoms': [atoms[i].label, atoms[j].label],
                                   'distance_angstrom': distance})
    excluded = set()
    max_bond_distance = 2 if include_1_4 else 3
    for i in range(len(atoms)):
        visited = {i}
        frontier = {i}
        for _ in range(max_bond_distance):
            frontier = {neighbor for j in frontier for neighbor in neighbors[j]} - visited
            visited.update(frontier)
        excluded.update(tuple(sorted((i, j))) for j in visited if i != j)
    return excluded, peptide_residue_pairs, {
        'unknown_residue_templates': sorted(unknown_residues),
        'explicit_disulfides': explicit_disulfides,
        'inferred_disulfides': assumed_disulfides,
    }


def analyze_clashes(path, chain_components=None, overlap_threshold=0.4, include_1_4=False):
    """Count nonbonded protein heavy-atom overlaps >0.4 Å, excluding 1–2/1–3/1–4.

    Set include_1_4=True to retain close dihedral contacts (excludes only 1–2/1–3).
    Component defaults follow SimpleFold 2GOO prediction chains; pass an
    explicit chain_components mapping for a reference or another protein.
    """
    components = DEFAULT_COMPONENTS if chain_components is None else chain_components
    atoms, conect, ssbonds, omitted = _read(path)
    excluded, peptide_pairs, assumptions = _topology(atoms, conect, ssbonds, include_1_4=include_1_4)
    xyz = np.asarray([a.xyz for a in atoms], dtype=float).reshape((-1, 3))
    if not np.isfinite(xyz).all():
        raise ValueError(f'Nonfinite coordinates in {path}')
    clashes = []
    intrachain = 0
    component_counts = defaultdict(lambda: {'clash_count': 0, 'intrachain_clash_count': 0,
                                          'interchain_clash_count': 0, 'max_overlap_angstrom': 0.0})
    component_pairs = Counter()
    ca_clashes = 0
    candidates = cKDTree(xyz).query_pairs(2 * max(RADII.values()) - overlap_threshold) if len(atoms) else []
    for i, j in candidates:
        a, b = atoms[i], atoms[j]
        distance = float(np.linalg.norm(xyz[i] - xyz[j]))
        if (a.name == b.name == 'CA' and distance < 2.5
                and a.residue_key != b.residue_key
                and frozenset((a.residue_key, b.residue_key)) not in peptide_pairs):
            ca_clashes += 1
        if (i, j) in excluded:
            continue
        overlap = RADII[a.element] + RADII[b.element] - distance
        if overlap <= overlap_threshold:
            continue
        same_chain = a.chain == b.chain
        intrachain += same_chain
        component_a = components.get(a.chain, a.chain or '(blank chain)')
        component_b = components.get(b.chain, b.chain or '(blank chain)')
        component_pairs[' / '.join(sorted((component_a, component_b)))] += 1
        for component in {component_a, component_b}:
            counts = component_counts[component]
            counts['clash_count'] += 1
            counts['intrachain_clash_count' if same_chain else 'interchain_clash_count'] += 1
            counts['max_overlap_angstrom'] = max(counts['max_overlap_angstrom'], overlap)
        clashes.append({'first': a.label, 'second': b.label, 'distance_angstrom': distance,
                        'overlap_angstrom': overlap, 'interchain': not same_chain,
                        'components': [component_a, component_b]})
    clashes.sort(key=lambda item: (-item['overlap_angstrom'], item['first'], item['second']))
    heavy_atoms_by_component = Counter(components.get(a.chain, a.chain or '(blank chain)') for a in atoms)
    for component, count in heavy_atoms_by_component.items():
        component_counts[component]['heavy_atom_count'] = count
        component_counts[component]['worst_clashes'] = [c for c in clashes if component in c['components']][:5]
    return {
        'method': ('Nonbonded protein heavy-atom geometric vdW overlaps; excludes '
                   + ('1–2/1–3' if include_1_4 else '1–2/1–3/1–4')
                   + '; not MolProbity all-atom clashscore'),
        'overlap_threshold_angstrom': overlap_threshold,
        'heavy_atom_count': len(atoms), 'clash_count': len(clashes),
        'intrachain_clash_count': int(intrachain),
        'interchain_clash_count': len(clashes) - int(intrachain),
        'clashes_per_1000_heavy_atoms': 1000 * len(clashes) / len(atoms) if atoms else None,
        'max_overlap_angstrom': clashes[0]['overlap_angstrom'] if clashes else 0.0,
        'worst_clashes': clashes[:20], 'by_component': dict(sorted(component_counts.items())),
        'component_pairs': dict(sorted(component_pairs.items())),
        'CA_nonadjacent_clash_count': ca_clashes, 'omitted_atoms': omitted,
        'assumptions': {'radii_angstrom': RADII, 'excluded_bond_orders': [1, 2] if include_1_4 else [1, 2, 3],
                        'protein_ATOM_only': True, 'first_model_only': True,
                        'includes_nonbonded_1_4_contacts': include_1_4,
                        'inferred_disulfides_require_unique_sulfur_partner': True,
                        'component_chain_mapping': components, **assumptions},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdb', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze_clashes(args.pdb), indent=2))


if __name__ == '__main__':
    main()
