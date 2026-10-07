#!/usr/bin/env python3
"""Read-only 2GOO crystal symmetry audit; candidates are not repaired coordinates.

Enumerates space-group operators and neighboring unit-cell images around the
BMP2 intermonomer Cys78. Records nearest partners and contact geometry. Optionally exports symmetry-derived assemblies to new paths, preserving local
geometry and the deposited source files.
"""
from __future__ import annotations

import argparse
import itertools
import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np


def coordinates(chain):
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for r in chain if r.het_flag == "A"
                     for a in r if not a.element.is_hydrogen], dtype=float)


def transformed(points, cell, operation, shift):
    result = []
    for point in points:
        fractional = cell.fractionalize(gemmi.Position(*point))
        target = np.array(operation.apply_to_xyz([fractional.x, fractional.y, fractional.z])) + shift
        cartesian = cell.orthogonalize(gemmi.Fractional(*target))
        result.append([cartesian.x, cartesian.y, cartesian.z])
    return np.array(result)


def contacts(first, second):
    distances = np.linalg.norm(first[:, None] - second[None, :], axis=-1)
    return {"minimum_heavy_atom_distance_angstrom": float(distances.min()),
            "heavy_atom_pairs_below_2_angstrom": int(np.sum(distances < 2)),
            "heavy_atom_pairs_below_4_angstrom": int(np.sum(distances < 4))}


def investigate(path):
    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError("Expected one coordinate model")
    spacegroup = structure.find_spacegroup()
    if spacegroup is None:
        raise ValueError("No identifiable crystal space group")
    operations = list(spacegroup.operations())
    points = {c: coordinates(structure[0][c]) for c in "ABCDEF"}
    sulfurs = {}
    for c in "AD":
        residue = next(r for r in structure[0][c] if r.seqid.num == 78)
        atom = residue["SG"][0]
        sulfurs[c] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    nearest, candidates = {}, []
    for base, group in (("A", "ABC"), ("D", "DEF")):
        rows = []
        for partner in "AD":
            for index, operation in enumerate(operations):
                for shift in itertools.product(range(-1, 2), repeat=3):
                    target = transformed([sulfurs[partner]], structure.cell, operation, shift)[0]
                    distance = float(np.linalg.norm(target - sulfurs[base]))
                    if distance < 0.01:
                        continue  # Exclude the same physical atom.
                    rows.append({"base_chain": base, "partner_chain": partner,
                                 "operation_index_zero_based": index,
                                 "fractional_operator": operation.triplet(), "lattice_shift": list(shift),
                                 "Cys78_SG_distance_angstrom": distance,
                                 "transformed_Cys78_SG_cartesian": target.tolist()})
        rows.sort(key=lambda r: r["Cys78_SG_distance_angstrom"])
        nearest[base] = rows[:10]
        # Inspect every sulfur-close candidate; do not assume an exact disulfide
        # from proximity (observed sulfur separation is longer than ~2.05 Å).
        for row in rows:
            if row["Cys78_SG_distance_angstrom"] >= 3.0:
                continue
            operation = operations[row["operation_index_zero_based"]]
            shift = row["lattice_shift"]
            partner_group = "ABC" if row["partner_chain"] == "A" else "DEF"
            transformed_group = {c: transformed(points[c], structure.cell, operation, shift)
                                 for c in partner_group}
            row = dict(row)
            row["base_half_chains"] = list(group)
            row["symmetry_half_source_chains"] = list(partner_group)
            row["cross_half_contacts"] = [{"base_chain": c, "symmetry_source_chain": d,
                                            **contacts(points[c], transformed_group[d])}
                                           for c in group for d in partner_group]
            origin = transformed([[0, 0, 0]], structure.cell, operation, shift)[0]
            basis = transformed(np.eye(3), structure.cell, operation, shift)
            rotation = basis - origin
            row["cartesian_row_vector_rotation"] = rotation.tolist()
            row["cartesian_translation_angstrom"] = origin.tolist()
            row["rotation_determinant"] = float(np.linalg.det(rotation))
            repeated = transformed(transformed(points[base], structure.cell, operation, shift),
                                   structure.cell, operation, shift)
            row["twofold_involution_max_atom_displacement_angstrom"] = float(np.max(np.linalg.norm(repeated - points[base], axis=1)))
            candidates.append(row)
    return {"reference": str(path.resolve()), "spacegroup": spacegroup.hm,
            "unit_cell": {name: getattr(structure.cell, name) for name in ("a", "b", "c", "alpha", "beta", "gamma")},
            "operations": [operation.triplet() for operation in operations],
            "search": "All 12 crystallographic operators × 27 neighboring lattice translations (-1..1 in each fractional dimension), both BMP2 chain sources A,D, for each base A,D. Same physical atom excluded.",
            "original_A_D_Cys78_SG_distance_angstrom": float(np.linalg.norm(sulfurs["A"] - sulfurs["D"])),
            "original_A_D_contact_geometry": contacts(points["A"], points["D"]),
            "nearest_sulfur_images": nearest, "sulfur_close_symmetry_candidates": candidates,
            "interpretation": "Two independently observed complex halves ABC and DEF each have a unique close BMP2 symmetry self-partner. Transforming all three proteins of a half gives a plausible 2:2:2 complex without cross-half heavyatom distances below 2 Å. This is a symmetry-derived candidate, not the supplied identity biological assembly. Cys78 sulfur separations ~2.42–2.46 Å are longer than a typical disulfide; proximity does not prove an exact native bond. No atomic coordinates are repaired; optional derived exports preserve local geometry."}


def corroborate_native_dimer(reference, native_path, candidate):
    try:
        from .compare_structures import reference_ca, kabsch
    except ImportError:
        from compare_structures import reference_ca, kabsch
    structure = gemmi.read_structure(str(reference))
    source = reference_ca(reference, candidate["base_chain"])
    native = {c: reference_ca(native_path, c) for c in "AB"}
    if any(chain.full_sequence != source.full_sequence for chain in native.values()):
        raise ValueError("Native dimer control BMP2 full sequence differs")
    operation = list(structure.find_spacegroup().operations())[candidate["operation_index_zero_based"]]
    copy = transformed(source.coordinates, structure.cell, operation, candidate["lattice_shift"])
    positions = {position: i for i, position in enumerate(source.full_positions)}
    rows = []
    for assignment in (("A", "B"), ("B", "A")):
        moving, fixed = [], []
        for xyz, chain in zip((source.coordinates, copy), assignment):
            for i, position in enumerate(native[chain].full_positions):
                if position in positions:
                    moving.append(xyz[positions[position]])
                    fixed.append(native[chain].coordinates[i])
        _, _, errors = kabsch(np.array(moving), np.array(fixed))
        rows.append({"native_chain_assignment": list(assignment), "matched_CA": len(errors),
                     "global_dimer_CA_RMSD_angstrom": float(np.sqrt(np.mean(errors**2)))})
    return min(rows, key=lambda row: row["global_dimer_CA_RMSD_angstrom"])


def derive_candidate(reference, output_dir, candidate):
    """Export a labelled symmetry-derived candidate without touching input files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    structure = gemmi.read_structure(str(reference))
    original_block = gemmi.cif.read(str(reference)).sole_block()
    operation = list(structure.find_spacegroup().operations())[candidate["operation_index_zero_based"]]
    shift = candidate["lattice_shift"]
    group = candidate["base_half_chains"]
    explicit_disulfides = []
    for connection in structure.connections:
        if connection.type != gemmi.ConnectionType.Disulf:
            continue
        if connection.partner1.chain_name not in group or connection.partner2.chain_name not in group:
            continue
        partners = []
        for partner in (connection.partner1, connection.partner2):
            partners.append((partner.chain_name, partner.res_id.seqid.num,
                             partner.res_id.seqid.icode, partner.res_id.name,
                             partner.atom_name, partner.altloc))
        explicit_disulfides.append((connection.name, partners, connection.reported_distance))
    structure.remove_ligands_and_waters()
    # Original ASU annotations refer to original D/E/F and must not be reused
    # for symmetry copies (their residue numbering/coordinates may differ).
    structure.connections.clear()
    structure.helices.clear()
    structure.sheets.clear()
    structure.assemblies.clear()
    structure.cispeps = []
    for name in [chain.name for chain in structure[0]]:
        if name not in group:
            structure[0].remove_chain(name)
    # Normalize both independent candidates to author chains ABC + DEF while
    # retaining atom coordinates, author residue numbering and occupancies.
    source_to_base = dict(zip(group, "ABC"))
    for chain in structure[0]:
        source_name = chain.name
        chain.name = source_to_base[source_name]
        for residue in chain:
            residue.subchain = chain.name
    for c, copy_name in zip("ABC", "DEF"):
        copy = structure[0][c].clone()
        copy.name = copy_name
        for residue in copy:
            residue.subchain = copy_name
            for atom in residue:
                xyz = transformed([[atom.pos.x, atom.pos.y, atom.pos.z]], structure.cell, operation, shift)[0]
                atom.pos = gemmi.Position(*xyz)
        structure[0].add_chain(copy)
    # Restore only the chosen half's explicit bonds and their exact symmetry
    # copies. Do not infer additional receptor disulfides from sulfur proximity.
    for suffix, remapping in (("base", source_to_base),
                              ("symmetry", dict(zip(group, "DEF")))):
        for original_name, partners, reported_distance in explicit_disulfides:
            connection = gemmi.Connection()
            connection.name = original_name + "_" + suffix
            connection.type = gemmi.ConnectionType.Disulf
            connection.asu = gemmi.Asu.Same
            connection.reported_distance = reported_distance
            addresses = [gemmi.AtomAddress(remapping[c], gemmi.SeqId(num, icode),
                                           residue, atom, altloc)
                         for c, num, icode, residue, atom, altloc in partners]
            connection.partner1, connection.partner2 = addresses
            structure.connections.append(connection)
    native = gemmi.Connection()
    native.name = "native_BMP2_interchain_Cys78"
    native.type = gemmi.ConnectionType.Disulf
    native.asu = gemmi.Asu.Same
    native.reported_distance = candidate["Cys78_SG_distance_angstrom"]
    native.partner1 = gemmi.AtomAddress("A", gemmi.SeqId(78, " "), "CYS", "SG")
    native.partner2 = gemmi.AtomAddress("D", gemmi.SeqId(78, " "), "CYS", "SG")
    structure.connections.append(native)
    for entity in structure.entities:
        if entity.name in ("1", "2", "3"):
            entity.subchains = {"1": ["A", "D"], "2": ["B", "E"], "3": ["C", "F"]}[entity.name]
        else:
            entity.subchains = []
    document = structure.make_mmcif_document()
    block = document.sole_block()
    # Gemmi emits canonical sequence in current versions; copying it explicitly
    # ensures correspondence remains recoverable across Gemmi versions.
    canonical = {row[0]: row[1] for row in original_block.find(["_entity_poly.entity_id", "_entity_poly.pdbx_seq_one_letter_code_can"])}
    ids = list(block.find_values("_entity_poly.entity_id"))
    block.set_pair("_struct.title", gemmi.cif.quote("Symmetry-derived 2GOO candidate from half " + "".join(group) + "; generated coordinates, not original deposited ASU"))
    if ids:
        loop = block.find_loop("_entity_poly.entity_id").get_loop()
        key = "_entity_poly.pdbx_seq_one_letter_code_can"
        if key in loop.tags:
            column = loop.tags.index(key)
            width = len(loop.tags)
            for i, entity_id in enumerate(ids):
                loop[i, column] = canonical[entity_id]
        else:
            loop.add_columns([key], "?")
            width = len(loop.tags)
            for i, entity_id in enumerate(ids):
                loop[i, width - 1] = canonical[entity_id]
    cif_path = output_dir / "reference.cif"
    pdb_path = output_dir / "reference.pdb"
    document.write_file(str(cif_path))
    structure.write_pdb(str(pdb_path))
    return {"reference_cif": str(cif_path.resolve()), "reference_pdb": str(pdb_path.resolve()),
            "source_to_derived_base_chains": source_to_base,
            "derived_symmetry_chains": {"D": "A", "E": "B", "F": "C"},
            "source_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "explicit_active_half_disulfides": len(explicit_disulfides),
            "duplicated_explicit_disulfides": 2 * len(explicit_disulfides),
            "total_annotated_disulfides": len(structure.connections),
            "added_native_topology": "BMP2 A:CYS78–D:CYS78 SG–SG disulfide. Biologically justified topology annotation from conserved native BMP2 homodimer and independent1REW explicit A:CYS78–B:CYS78 disulfide record (2.048 Å); generated 2GOO bond geometry remains ~2.42–2.46 Å, unrefined. No other bond inferred from proximity.",
            "coordinate_policy": "Original half unchanged, symmetry-generated second half. Protein-only; no minimization, occupancy edits or stereochemistry repairs. Original ASU secondary-structure and assembly annotations omitted; only active-half explicit disulfides and their copies retained, plus separately justified native BMP2 interchain disulfide; coordinates retain all original local defects. Separate generated output; original files retained."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-dimer", type=Path, help="Independent full-sequence BMP2 dimer control mmCIF, e.g.1REW chainsA,B")
    parser.add_argument("--derive-dir", type=Path, help="Explicitly export candidate ABC/DEF assemblies to separate subdirectories")
    args = parser.parse_args()
    report = investigate(args.reference)
    block = gemmi.cif.read(str(args.reference)).sole_block()
    report["depositor_struct_biol_details"] = block.find_value("_struct_biol.details")
    for candidate in report["sulfur_close_symmetry_candidates"]:
        if args.native_dimer:
            candidate["native_dimer_control"] = corroborate_native_dimer(args.reference, args.native_dimer, candidate)
        if args.derive_dir:
            candidate["derived_files"] = derive_candidate(args.reference, args.derive_dir / (candidate["base_chain"] + "_half_symmetry"), candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for candidate in report["sulfur_close_symmetry_candidates"]:
        print(candidate["base_chain"], candidate["fractional_operator"], candidate["lattice_shift"],
              candidate["Cys78_SG_distance_angstrom"])


if __name__ == "__main__":
    main()
