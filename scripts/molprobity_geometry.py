#!/usr/bin/env python3
"""Coordinate-only MolProbity geometry using cctbx 2026.9 / Reduce2 / Probe2.

Run in the separate cctbx-base + chem_data environment, not the folding env:
    python scripts/molprobity_geometry.py input.pdb output.json

The input file is never rewritten. Geometry is measured on observed occupied
heavy atoms; hydrogens are rebuilt only on an in-memory copy for clashscore.
No Asn/Gln/His flips, relaxation, crystal mates, or assembly expansion is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import sys


def _percent(count, total):
    return 100.0 * count / total if total else None


def _clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean_json(v) for v in value]
    return value


def _version_info():
    versions = {}
    for name in ("cctbx-base", "cctbx", "chem_data"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    # chem_data is a conda-only data package and may have no dist-info.
    for path in (Path(sys.prefix) / "conda-meta").glob("*.json"):
        item = json.loads(path.read_text())
        if item.get("name") in ("cctbx-base", "cctbx", "chem_data"):
            versions[item["name"]] = {
                key: item.get(key) for key in ("version", "build", "channel", "sha256")
            }
    return versions


def _residue_id(residue, chain, altloc):
    return {
        "chain_id": chain.id,
        "resseq": residue.resseq.strip(),
        "icode": residue.icode.strip(),
        "altloc": altloc,
        "resname": residue.resname,
    }


def _heavy_signature(hierarchy):
    result = {}
    for atom in hierarchy.atoms():
        if atom.element.strip().upper() in ("H", "D"):
            continue
        labels = atom.fetch_labels()
        key = (labels.model_id, labels.chain_id, labels.resseq, labels.icode,
               labels.altloc, labels.resname, atom.name)
        if key in result:
            raise RuntimeError(f"Duplicate atom identifier: {key}")
        result[key] = tuple(atom.xyz)
    return result


def _same_heavy_atoms(before, after):
    if before.keys() != after.keys():
        raise RuntimeError("Hydrogen placement changed heavy-atom identities")
    displacement = max((math.dist(xyz, after[key]) for key, xyz in before.items()), default=0.0)
    if displacement > 1e-6:
        raise RuntimeError(f"Hydrogen placement moved heavy atoms by {displacement} A")
    return displacement


def _validation_summary(validation):
    return {
        "outliers": int(validation.n_outliers),
        "evaluated": int(validation.n_total),
        "outlier_percent": _percent(validation.n_outliers, validation.n_total),
        "outlier_details": [_outlier_json(item) for item in validation.results
                            if item.is_outlier()],
    }


def _outlier_json(item):
    # Unlike bond/angle/chirality, cctbx2026.9 planarity inherits the empty
    # entity.as_JSON implementation. Preserve its actual atom locations here.
    details = json.loads(item.as_JSON())
    if details or not hasattr(item, "atoms_info"):
        return details
    details = {
        name: getattr(item, name) for name in
        ("score", "outlier", "xyz", "rms_deltas", "delta_max", "residual")
        if hasattr(item, name)
    }
    details["atoms_info"] = [{
        name: getattr(atom, name) for name in
        ("name", "element", "model_id", "chain_id", "resseq", "icode", "resname",
         "altloc", "occupancy", "xyz") if hasattr(atom, name)
    } for atom in item.atoms_info]
    return details


def _missing_atoms(hierarchy):
    from mmtbx.monomer_library import server
    from mmtbx.rotamer.rotamer_eval import eval_residue_completeness, mon_lib_query

    library = server.server()
    incomplete = []
    unknown = []
    evaluated = 0
    for chain in hierarchy.only_model().chains():
        for group in chain.residue_groups():
            for conformer in group.conformers():
                residue = conformer.only_residue()
                identity = _residue_id(residue, chain, conformer.altloc)
                if mon_lib_query(residue.resname.strip().upper(), library) is None:
                    unknown.append(identity)
                    continue
                evaluated += 1
                missing = eval_residue_completeness(residue, library, ignore_hydrogens=True)
                if missing:
                    incomplete.append({**identity, "missing_heavy_atoms": missing})
    return {
        "evaluated_residue_conformers": evaluated,
        "incomplete_residue_conformers": len(incomplete),
        "incomplete_percent": _percent(len(incomplete), evaluated),
        "missing_heavy_atom_count": sum(len(item["missing_heavy_atoms"]) for item in incomplete),
        "details": incomplete,
        "unknown_residues": unknown,
        "definition": "Observed residues compared with monomer-library heavy atoms; entirely absent residues are not inferred.",
    }


def analyze(path) -> dict:
    """Return JSON-serializable diagnostics; tool/parsing failures propagate."""
    path = Path(path).resolve()
    original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    log = io.StringIO()
    with redirect_stdout(log), redirect_stderr(log):
        result = _analyze(path, log)
    if hashlib.sha256(path.read_bytes()).hexdigest() != original_sha:
        raise RuntimeError("Input structure changed during geometry validation")
    result.update({"input_path": str(path), "input_sha256": original_sha,
                   "tool_versions": _version_info(), "tool_log": log.getvalue()})
    return _clean_json(result)


def _analyze(path, log):
    from cctbx import crystal
    from iotbx.data_manager import DataManager
    import iotbx.pdb
    import iotbx.phil
    from mmtbx import model as model_module
    from mmtbx.probe import Helpers
    from mmtbx.validation.clashscore2 import clashscore2
    from mmtbx.validation.ramalyze import ramalyze
    from mmtbx.validation.rotalyze import rotalyze
    from mmtbx.validation.restraints import combined
    from mmtbx.validation.utils import molprobity_score
    from scitbx.array_family import flex

    # Hierarchy construction reads coordinates without expanding MTRIX/BIOMT.
    hierarchy = iotbx.pdb.input(file_name=str(path)).construct_hierarchy()
    if len(hierarchy.models()) != 1:
        raise RuntimeError("Validate one structure/model at a time")
    input_atoms = len(hierarchy.atoms())
    keep = flex.bool([atom.occ > 0 and atom.element.strip().upper() not in ("H", "D")
                      for atom in hierarchy.atoms()])
    hierarchy = hierarchy.select(keep, copy_atoms=True)
    if not len(hierarchy.atoms()):
        raise RuntimeError("No occupied heavy atoms to validate")

    # Put the supplied assembly in a roomy, noncrystallographic P1 box. Original
    # crystal symmetry is deliberately discarded. A common translation is the
    # only coordinate change; it leaves all distances and torsions invariant.
    xyz = [tuple(atom.xyz) for atom in hierarchy.atoms()]
    lower = [min(point[axis] for point in xyz) for axis in range(3)]
    upper = [max(point[axis] for point in xyz) for axis in range(3)]
    shift = [20.0 - value for value in lower]
    for atom in hierarchy.atoms():
        atom.xyz = tuple(atom.xyz[axis] + shift[axis] for axis in range(3))
    cell = tuple(math.ceil(upper[axis] - lower[axis] + 40.0) for axis in range(3)) + (90, 90, 90)
    symmetry = crystal.symmetry(unit_cell=cell, space_group_symbol="P 1")
    hierarchy.atoms().reset_i_seq()
    signature = _heavy_signature(hierarchy)

    model = model_module.manager(pdb_hierarchy=hierarchy, crystal_symmetry=symmetry,
                                 stop_for_unknowns=True, log=log)
    params = model_module.manager.get_default_pdb_interpretation_params()
    params.pdb_interpretation.clash_guard.nonbonded_distance_threshold = None
    params.pdb_interpretation.proceed_with_excessive_length_bonds = True
    params.pdb_interpretation.max_reasonable_bond_distance = None
    params.pdb_interpretation.flip_symmetric_amino_acids = False
    model.process(make_restraints=True, pdb_interpretation_params=params)
    _same_heavy_atoms(signature, _heavy_signature(model.get_hierarchy()))
    hierarchy = model.get_hierarchy()

    rama = ramalyze(pdb_hierarchy=hierarchy, outliers_only=True, out=log, quiet=True)
    rota = rotalyze(pdb_hierarchy=hierarchy, outliers_only=True, out=log, quiet=True)
    restraints = combined(pdb_hierarchy=hierarchy,
                          xray_structure=model.get_xray_structure(),
                          geometry_restraints_manager=model.get_restraints_manager().geometry,
                          ignore_hd=True, sigma_cutoff=4.0, outliers_only=True)
    missing = _missing_atoms(hierarchy)

    dm = DataManager(datatypes=["model"], logger=log)
    dm.add_model(str(path), model.deep_copy())
    probe = iotbx.phil.parse(Helpers.probe_phil_parameters).extract().probe
    clashes = clashscore2(probe_parameters=probe, data_manager=dm,
                         keep_hydrogens=False, do_flips=False, nuclear=False,
                         fast=False, condensed_probe=True, verbose=False,
                         b_factor_cutoff=None, save_probe_output=False,
                         ignore_missing_restraints=False, out=log)
    hmodel = clashes.hydrogenated_model
    heavy_displacement = _same_heavy_atoms(signature, _heavy_signature(hmodel.get_hierarchy()))
    native_clashes = json.loads(clashes.as_JSON())
    clash_count = sum(item["num_clashes"] for item in native_clashes["summary_results"].values())
    h_atoms = sum(atom.element.strip().upper() in ("H", "D") for atom in hmodel.get_hierarchy().atoms())
    raw_score = clashes.get_clashscore()
    if raw_score is None or not math.isfinite(raw_score):
        raise RuntimeError("MolProbity did not return a finite clashscore")

    rama_summary = _validation_summary(rama)
    rama_summary["favored"] = int(rama.n_favored)
    rama_summary["favored_percent"] = _percent(rama.n_favored, rama.n_total)
    rama_summary["allowed"] = int(rama.n_allowed)
    rota_summary = _validation_summary(rota)
    geometries = {}
    for name in ("bonds", "angles", "chiralities", "planarities"):
        validation = getattr(restraints, name)
        geometries[name] = {**_validation_summary(validation),
                            "rms_deviation": validation.mean,
                            "max_deviation": validation.max,
                            "rms_z": validation.z_mean,
                            "max_z": validation.z_max}
    chiral_types = Counter(item.get("outlier_type")
                          for item in geometries["chiralities"]["outlier_details"])
    geometries["chiralities"]["outlier_type_counts"] = dict(chiral_types)

    return {
        "schema_version": 1,
        "method": "cctbx MolProbity coordinate-only validation; Reduce2/Probe2 clashscore",
        "settings": {
            "hydrogens": "Regenerated and optimized in memory; never saved over input",
            "do_sidechain_flips": False,
            "nuclear_distances": False,
            "b_factor_cutoff": None,
            "geometry_sigma_cutoff": 4.0,
            "geometry_uses_hydrogens": False,
            "crystal_neighbors": False,
            "unit_cell": list(cell),
            "translation_only_angstrom": shift,
            "zero_occupancy_atoms_excluded": True,
            "heavy_atom_max_displacement_during_hydrogenation": heavy_displacement,
            "covalent_topology": {
                "method": "cctbx monomer library plus distance-inferred inter-residue links",
                "link_distance_cutoff_angstrom": params.pdb_interpretation.link_distance_cutoff,
                "disulfide_distance_cutoff_angstrom": params.pdb_interpretation.disulfide_distance_cutoff,
                "ssbond_header_consumed": False,
                "limitations": [
                    "Peptide links beyond the default 3 A cutoff and their associated angle/dihedral restraints are not represented.",
                    "Disulfides are inferred from nearby sulfur atoms; intended stretched disulfides are not tested.",
                    "Hierarchy-only input does not consume SSBOND/LINK headers as covalent topology.",
                ],
            },
            "geometry_denominators": (
                "Native cctbx evaluated restraint counts, not sequence-expected bonds or intended disulfide counts. "
                "Bond/angle RMS and native denominators can include inferred links with covalent origin classes; "
                "native outlier enumeration filters origin_id=0 ('covalent geometry'). "
                "Consequently these outlier fractions do not test all inferred disulfide restraints."
            ),
        },
        "atom_counts": {"input": input_atoms, "occupied_heavy": len(signature),
                        "added_hydrogens": h_atoms,
                        "hydrogenated_total": len(hmodel.get_hierarchy().atoms())},
        "clashscore": float(raw_score),
        "clash_count": int(clash_count),
        "clashscore_normalization": {
            "atoms_scored": int(clashes.probe_clashscore_manager.n_atoms),
            "clashes_scored": int(clashes.probe_clashscore_manager.n_clashes),
            "occupancy_cutoff_percent": float(clashes.probe_clashscore_manager.occupancy_frac),
        },
        "clashes": native_clashes,
        "ramachandran": rama_summary,
        "rotamers": rota_summary,
        "geometry": geometries,
        "missing_atoms": missing,
        "molprobity_score": (molprobity_score(float(raw_score), rota_summary["outlier_percent"],
                                                rama_summary["favored_percent"])
                              if rama.n_total and rota.n_total else None),
        "sources": [
            "https://github.com/cctbx/cctbx_project/tree/v2026.9/mmtbx/validation",
            "https://github.com/cctbx/cctbx_project/blob/v2026.9/README.md#monomer-library",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
