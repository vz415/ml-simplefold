#!/usr/bin/env python3
"""Prepare the experimental 2H62 subcomplex and a provisional six-chain target.

The added receptors are ligand-copy-swap placements, not experimental truth.
This preparation does not perform inference or constrain the SimpleFold sampler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np

try:
    from .compare_structures import kabsch, reference_ca
except ImportError:
    from compare_structures import kabsch, reference_ca


# Match the interleaved reference convention used by the 2GOO analysis.
CHAIN_SOURCES = {"A": "A", "B": "C", "C": "D", "D": "B", "E": "C", "F": "D"}
CHAIN_PROTEINS = {"A": "BMP2", "B": "BMPR1A", "C": "ACVR2B",
                  "D": "BMP2", "E": "BMPR1A", "F": "ACVR2B"}
EXPERIMENTAL_CHAINS = "ABCD"
MODELED_CHAINS = "EF"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_swap_transform(source_cif):
    """Jointly fit both ligand monomers to their exchanged positions."""
    first, second = (reference_ca(source_cif, chain) for chain in "AB")
    a = dict(zip(first.full_positions, first.coordinates))
    b = dict(zip(second.full_positions, second.coordinates))
    positions = sorted(a.keys() & b.keys())
    xyz_a, xyz_b = (np.array([lookup[i] for i in positions]) for lookup in (a, b))
    moving, fixed = np.concatenate([xyz_a, xyz_b]), np.concatenate([xyz_b, xyz_a])
    rotation, translation, errors = kabsch(moving, fixed)
    repeated = (moving @ rotation + translation) @ rotation + translation
    return rotation, translation, {
        "definition": "row coordinates: x_new = x_old @ R + t; jointly fit BMP2 A→B and B→A",
        "R": rotation.tolist(), "t_angstrom": translation.tolist(),
        "common_ligand_full_positions_zero_based": positions,
        "matched_CA_per_monomer": len(positions),
        "copy_swap_CA_RMSD_angstrom": float(np.sqrt(np.mean(errors ** 2))),
        "rotation_degrees": float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "involution_max_displacement_angstrom": float(np.linalg.norm(repeated - moving, axis=1).max()),
    }


def column_indices(tags):
    return {tag.split(".", 1)[1]: i for i, tag in enumerate(tags)}


def category_rows(block, category):
    table = block.find_mmcif_category(category)
    return list(table.tags), [list(row) for row in table]


def add_category(block, category, tags, rows):
    loop = block.init_loop(category, [tag.split(".", 1)[1] for tag in tags])
    for row in rows:
        loop.add_row(row)


def coordinate_document(source, output_chains, rotation, translation):
    """Copy atom-site tokens; only names, atom IDs and added coordinates change."""
    document = gemmi.cif.Document()
    block = document.add_new_block("2h62_inpaint")
    modeled = any(chain in MODELED_CHAINS for chain in output_chains)
    title = ("Provisional 2H62 six-chain completion; E/F are modeled ligand-copy-swap placements, not experimental truth"
             if modeled else "Experimental four-chain 2H62 protein subcomplex; chains remapped, coordinates unchanged")
    block.set_pair("_struct.title", gemmi.cif.quote(title))
    atom_tags, source_atoms = category_rows(source, "_atom_site.")
    columns = column_indices(atom_tags)
    atoms, chain_entities = [], {}
    for destination in output_chains:
        native = CHAIN_SOURCES[destination]
        for source_row in source_atoms:
            if source_row[columns["auth_asym_id"]] != native or source_row[columns["label_seq_id"]] in (".", "?"):
                continue
            row = source_row.copy()
            row[columns["id"]] = str(len(atoms) + 1)
            row[columns["label_asym_id"]] = row[columns["auth_asym_id"]] = destination
            chain_entities[destination] = row[columns["label_entity_id"]]
            if destination in MODELED_CHAINS:
                xyz = np.array([float(row[columns[axis]]) for axis in ("Cartn_x", "Cartn_y", "Cartn_z")])
                xyz = xyz @ rotation + translation
                for axis, value in zip(("Cartn_x", "Cartn_y", "Cartn_z"), xyz):
                    row[columns[axis]] = f"{value:.10f}"
            atoms.append(row)
    for category in ("_entity_poly.", "_entity_poly_seq."):
        tags, rows = category_rows(source, category)
        index = column_indices(tags)
        rows = [row.copy() for row in rows if row[index["entity_id"]] in chain_entities.values()]
        if category == "_entity_poly." and "pdbx_strand_id" in index:
            for row in rows:
                row[index["pdbx_strand_id"]] = gemmi.cif.quote(",".join(c for c, entity in chain_entities.items() if entity == row[index["entity_id"]]))
        add_category(block, category, tags, rows)
    asym = block.init_loop("_struct_asym.", ["id", "entity_id", "details"])
    for chain, entity in chain_entities.items():
        detail = "modeled receptor copy" if chain in MODELED_CHAINS else "experimental coordinates"
        asym.add_row([chain, entity, gemmi.cif.quote(detail)])
    add_category(block, "_atom_site.", atom_tags, atoms)

    # Preserve deposited explicit bonds, remapped to their actual chains. The
    # copies inherit only source intrachain bonds; no new proximity bonds added.
    conn_tags, conn_rows = category_rows(source, "_struct_conn.")
    if conn_tags:
        index = column_indices(conn_tags)
        output_rows = []
        mappings = [{CHAIN_SOURCES[c]: c for c in EXPERIMENTAL_CHAINS},
                    {CHAIN_SOURCES[c]: c for c in MODELED_CHAINS if c in output_chains}]
        for suffix, mapping in zip(("experimental", "modeled"), mappings):
            for original in conn_rows:
                chain1, chain2 = (original[index[f"ptnr{i}_auth_asym_id"]] for i in (1, 2))
                if chain1 not in mapping or chain2 not in mapping:
                    continue
                row = original.copy()
                row[index["id"]] = gemmi.cif.quote(gemmi.cif.as_string(row[index["id"]]) + "_" + suffix)
                for partner in (1, 2):
                    for kind in ("auth", "label"):
                        key = f"ptnr{partner}_{kind}_asym_id"
                        if key in index:
                            row[index[key]] = mapping[original[index[f"ptnr{partner}_auth_asym_id"]]]
                output_rows.append(row)
        add_category(block, "_struct_conn.", conn_tags, output_rows)
    return document


def known_regions(observed_cif, sequences):
    block = gemmi.cif.read(str(observed_cif)).sole_block()
    tags, rows = category_rows(block, "_atom_site.")
    columns = column_indices(tags)
    chains = {
        chain: {"protein": CHAIN_PROTEINS[chain], "source_author_chain": CHAIN_SOURCES[chain],
                "experimental": chain in EXPERIMENTAL_CHAINS, "full_sequence": sequence,
                "full_sequence_length": len(sequence), "known_residue_mask": [False] * len(sequence),
                "occupied_heavy_atoms": []}
        for chain, sequence in sequences.items()
    }
    for row in rows:
        if row[columns["type_symbol"]].upper() in ("H", "D") or float(row[columns["occupancy"]]) <= 0:
            continue
        chain = row[columns["auth_asym_id"]]
        position = int(row[columns["label_seq_id"]]) - 1
        chains[chain]["known_residue_mask"][position] = True
        key = {"full_position_zero_based": position, "label_seq_id": position + 1,
               "author_residue_number": row[columns["auth_seq_id"]],
               "insertion_code": row[columns["pdbx_PDB_ins_code"]],
               "residue_name": row[columns["label_comp_id"]],
               "atom_name": gemmi.cif.as_string(row[columns["label_atom_id"]]),
               "altloc": row[columns["label_alt_id"]], "occupancy": float(row[columns["occupancy"]]),
               "element": row[columns["type_symbol"]],
               "coordinates_angstrom": [float(row[columns[axis]]) for axis in ("Cartn_x", "Cartn_y", "Cartn_z")]}
        chains[chain]["occupied_heavy_atoms"].append(key)
    for record in chains.values():
        record["known_full_positions_zero_based"] = [i for i, known in enumerate(record["known_residue_mask"]) if known]
        record["unknown_full_positions_zero_based"] = [i for i, known in enumerate(record["known_residue_mask"]) if not known]
    return {"schema_version": 1, "target": "2h62-inpaint",
            "policy": "Experimental occupied heavy atoms only. E/F and unresolved residues are unknown. Alternate conformers retained; choose a conformer before applying coordinate constraints.",
            "sampler_status": "Preparation metadata only; not a SimpleFold inpainting or atom-padding mask.",
            "experimental_chains": list(EXPERIMENTAL_CHAINS), "excluded_modeled_chains": list(MODELED_CHAINS),
            "chains": chains}


def placement_contacts(cif_path):
    """Report short distances for provisional copies without treating them as truth."""
    tags, rows = category_rows(gemmi.cif.read(str(cif_path)).sole_block(), "_atom_site.")
    columns = column_indices(tags)
    points = {c: [] for c in CHAIN_SOURCES}
    for row in rows:
        if row[columns["type_symbol"]] in ("H", "D") or float(row[columns["occupancy"]]) <= 0 or row[columns["label_alt_id"]] not in (".", "?", "A"):
            continue
        points[row[columns["auth_asym_id"]]].append([float(row[columns[axis]]) for axis in ("Cartn_x", "Cartn_y", "Cartn_z")])
    points = {c: np.array(xyz) for c, xyz in points.items()}
    pairs = []
    for i, first in enumerate(CHAIN_SOURCES):
        for second in list(CHAIN_SOURCES)[i + 1:]:
            if first not in MODELED_CHAINS and second not in MODELED_CHAINS:
                continue
            distances = np.linalg.norm(points[first][:, None] - points[second][None, :], axis=-1)
            pairs.append({"chains": [first, second], "minimum_heavy_atom_distance_angstrom": float(distances.min()),
                          "heavy_atom_pairs_below_2_angstrom": int((distances < 2).sum()),
                          "heavy_atom_pairs_below_4_angstrom": int((distances < 4).sum())})
    return {"definition": "Occupied heavy atoms, blank/A alternates, strict distance cutoffs; diagnostic, not a van der Waals clashscore.",
            "involving_modeled_receptors": pairs,
            "total_modeled_interchain_heavy_atom_pairs_below_2_angstrom": sum(p["heavy_atom_pairs_below_2_angstrom"] for p in pairs)}


def prepare(source_cif, output_dir, fasta, provenance):
    source_cif, output_dir, fasta, provenance = map(Path, (source_cif, output_dir, fasta, provenance))
    source = gemmi.cif.read(str(source_cif)).sole_block()
    rotation, translation, transform = copy_swap_transform(source_cif)
    sequences = {c: reference_ca(source_cif, native).full_sequence for c, native in CHAIN_SOURCES.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, chains in (("observed", EXPERIMENTAL_CHAINS), ("symmetry_completion", "ABCDEF")):
        document = coordinate_document(source, chains, rotation, translation)
        cif_path, pdb_path = output_dir / f"{name}.cif", output_dir / f"{name}.pdb"
        document.write_file(str(cif_path))
        structure = gemmi.make_structure_from_block(document.sole_block())
        structure.write_pdb(str(pdb_path))
        outputs[f"{name}_cif"], outputs[f"{name}_pdb"] = cif_path, pdb_path
    fasta.parent.mkdir(parents=True, exist_ok=True)
    fasta.write_text("".join(f">{chain}|protein\n{sequence}\n" for chain, sequence in sequences.items()))
    outputs["fasta"] = fasta
    mask_path = output_dir / "known_regions.json"
    mask_path.write_text(json.dumps(known_regions(outputs["observed_cif"], sequences), indent=2) + "\n")
    outputs["known_regions"] = mask_path
    report = {
        "target": "2h62-inpaint", "pdb_id": "2H62", "source_url": "https://www.rcsb.org/structure/2H62",
        "source_cif": str(source_cif), "source_cif_sha256": sha256(source_cif),
        "source_assembly": "Deposited author biological assembly 1 is BMP2:BMPR1A:ACVR2B = 2:1:1, identity operation only.",
        "requested_stoichiometry": "BMP2:BMPR1A:ACVR2B = 2:2:2",
        "chains": {c: len(sequence) for c, sequence in sequences.items()},
        "total_residues": sum(map(len, sequences.values())),
        "chain_proteins": CHAIN_PROTEINS, "source_author_chain_mapping": CHAIN_SOURCES,
        "experimental_chains": list(EXPERIMENTAL_CHAINS), "modeled_chains": list(MODELED_CHAINS),
        "input": "Full canonical deposited construct sequences including unresolved termini; no coordinates passed to baseline inference.",
        "coordinate_policy": "Native four-chain protein coordinates unchanged. E/F are approximate ligand-copy-swap receptor placements. No residues/atoms repaired, no minimization. Alternate locations and occupancies retained. Explicit source bonds remapped/copied; no new bonds inferred.",
        "reference_warning": "The six-chain symmetry_completion is PROVISIONAL MODELING, NOT EXPERIMENTAL GROUND TRUTH. Its short-distance overlaps must not become enforced native contacts. Use observed.cif for experimental validation and future known-coordinate constraints.",
        "transform": transform, "placement_diagnostics": placement_contacts(outputs["symmetry_completion_cif"]),
        "fasta_sha256": sha256(fasta),
        "outputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
    }
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cif", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.source_cif, args.output_dir, args.fasta, args.provenance)
    print(json.dumps({"target": report["target"], "total_residues": report["total_residues"],
                      "copy_swap_CA_RMSD_angstrom": report["transform"]["copy_swap_CA_RMSD_angstrom"],
                      "modeled_copy_pairs_below_2_angstrom": report["placement_diagnostics"]["total_modeled_interchain_heavy_atom_pairs_below_2_angstrom"]}))


if __name__ == "__main__":
    main()
