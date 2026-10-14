#!/usr/bin/env python3
"""Compare BMP6 refinement with the historical reference assignment held fixed.

Inputs must have exact A–F FASTA identities and 1-based construct numbering.
This performs coordinate analysis only. Official DockQ is required by default;
--dockq skip is an explicitly incomplete local diagnostic. No chain assignment
is optimized at either stage, including when computing guidance scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bmp90_evaluate as ev
from fk_mapped_distance_reward import MappedDistanceReward, _fasta
from structure_clashes import RADII, SIDECHAINS, _read

BACKBONE = ("N", "CA", "C", "O")
IDENTITY = dict(zip("ABCDEF", "ABCDEF"))


def receipt(path):
    return {"path": str(Path(path).resolve()), "sha256": ev.sha256(path)}


def rmsd(a, b):
    return float(np.sqrt(np.mean(np.sum((a-b)**2, axis=1))))


def validate_assignment(assignment, chains):
    by_chain = {c["input_chain"]: c for c in chains}
    if set(assignment) != set(by_chain) or set(assignment.values()) != set(by_chain):
        raise ValueError("Historical assignment must be a complete chain bijection")
    for reference, model in assignment.items():
        if any(by_chain[reference][field] != by_chain[model][field] for field in ("gene", "sequence")):
            raise ValueError("Historical assignment exchanges different genes or sequences")


def historical_model(evaluation, sample_index):
    rows = [m for m in evaluation["models"] if
            re.search(rf"_sampled_{sample_index}\.pdb$", m["model_path"])]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one historical sample {sample_index}, found {len(rows)}")
    return rows[0]


def strict_atoms(path, chains):
    """Require complete standard heavy atoms; OXT is optional at chain ends."""
    lines = Path(path).read_text().splitlines()
    if sum(line.startswith("MODEL ") for line in lines) > 1:
        raise ValueError(f"Multiple models in {path}")
    parsed, conect, ssbonds, omitted = _read(path)
    if omitted.get("alternate_or_duplicate", 0) or omitted.get("zero_occupancy", 0):
        raise ValueError(f"Alternate/duplicate/zero-occupancy heavy atoms in {path}")
    if len({a.serial for a in parsed}) != len(parsed):
        raise ValueError(f"Duplicate heavy-atom serials in {path}")
    if any(a.alternate or not np.isfinite(a.occupancy) for a in parsed):
        raise ValueError(f"Noncanonical altloc or nonfinite occupancy in {path}")
    atoms, _ = ev.read_atoms(path)
    coverage = ev.validate_atoms(atoms, chains, require_complete=True)
    reverse = {v: k for k, v in ev.AA1.items()}
    missing, unexpected, expected = [], [], set()
    for chain in chains:
        c, sequence = chain["input_chain"], chain["sequence"]
        for position, letter in enumerate(sequence, 1):
            names = set(BACKBONE)
            names.update(SIDECHAINS[reverse[letter]].replace("-", " ").split())
            expected.update((c, position, name) for name in names)
    missing = sorted(expected-set(atoms))
    for key in set(atoms)-expected:
        chain = next(c for c in chains if c["input_chain"] == key[0])
        if key != (key[0], len(chain["sequence"]), "OXT"):
            unexpected.append(key)
    if missing or unexpected:
        raise ValueError(f"Incomplete/noncanonical heavy atoms in {path}: missing={missing[:12]}, unexpected={unexpected[:12]}")
    return atoms, {"sequence_CA_coverage": coverage, "expected_standard_heavy_atoms": len(expected),
                   "observed_heavy_atoms": len(atoms), "missing_standard_heavy_atoms": missing,
                   "optional_terminal_OXT": sorted(set(atoms)-expected), "omitted_atoms": omitted,
                   "input_SSBOND_records": ssbonds, "input_CONECT_edges": len(conect),
                   "topology_policy": "Input CONECT and coordinate-inferred bonds are not used for exclusions"}


def coordinate_identity(first, second):
    return (set(first) == set(second) and all(first[k].residue == second[k].residue
            and first[k].element == second[k].element and first[k].xyz == second[k].xyz for k in first))


def fixed_topology(chains, atoms, reference, assignment, topology=None):
    """Atom-name graph independent of both stages' coordinates/CONECT records."""
    if topology:
        bonds = [tuple((p["chain"], p["input_position_1based"], p["atom"])
                       for p in (row["left"], row["right"])) for row in topology["native_disulfides"]]
        basis = "Explicit supplied annotation-verified native_disulfides; fixed in input-chain space"
    else:
        bonds = [tuple((assignment[c], p, name) for c, p, name in row["reference_atom_keys"])
                 for row in ev.cysteine_geometry(reference, reference)]
        basis = "Fallback: reference SG pairs 1.8–2.3 Å, transferred once using the original assignment; geometric, not annotated topology"
    sulfur_use = Counter(k for pair in bonds for k in pair)
    if any(n != 1 for n in sulfur_use.values()) or len({frozenset(b) for b in bonds}) != len(bonds):
        raise ValueError("Duplicate or multiply paired disulfide endpoints")
    for key in sulfur_use:
        if key not in atoms or atoms[key].residue != "CYS" or key[2] != "SG":
            raise ValueError(f"Disulfide endpoint absent or not CYS SG: {key}")
    if topology and (len(bonds) != topology["disulfide_count"] or set(sulfur_use) !=
                     {k for k, a in atoms.items() if a.residue == "CYS" and k[2] == "SG"}):
        raise ValueError("Annotated BMP6 topology must account for every cysteine exactly once")
    graph = defaultdict(set)
    def bond(a, b):
        graph[a].add(b)
        graph[b].add(a)
    reverse = {v: k for k, v in ev.AA1.items()}
    for chain in chains:
        c = chain["input_chain"]
        for pos, aa in enumerate(chain["sequence"], 1):
            for pair in ("N-CA CA-C C-O C-OXT " + SIDECHAINS[reverse[aa]]).split():
                a, b = pair.split("-")
                bond((c, pos, a), (c, pos, b))
            if pos > 1:
                bond((c, pos-1, "C"), (c, pos, "N"))
    for a, b in bonds:
        bond(a, b)
    excluded = set()
    for key in graph:
        visited, frontier = {key}, {key}
        for _ in range(3):
            frontier = {k for previous in frontier for k in graph[previous]} - visited
            visited.update(frontier)
        excluded.update(frozenset((key, other)) for other in visited if key != other)
    return bonds, excluded, basis


def physical(atoms, excluded, disulfides, reference, assignment):
    keys = sorted(atoms)
    xyz = np.array([atoms[k].xyz for k in keys])
    covered = {(assignment[c], p) for c, p, a in reference if a == "CA"}
    heavy, severe, severe_intra = [], [], []
    for i, j in cKDTree(xyz).query_pairs(2*max(RADII.values())-.4):
        a, b = atoms[keys[i]], atoms[keys[j]]
        distance = float(np.linalg.norm(xyz[i]-xyz[j]))
        overlap = RADII[a.element]+RADII[b.element]-distance
        row = {"atoms": [keys[i], keys[j]], "distance_A": distance, "overlap_A": overlap,
               "interchain": a.chain != b.chain,
               "reference_scope": "core_core" if all(k[:2] in covered for k in (keys[i], keys[j]))
               else "tail_tail" if all(k[:2] not in covered for k in (keys[i], keys[j])) else "core_tail"}
        if a.element in {"C", "N", "O"} and b.element in {"C", "N", "O"} and overlap > 1.:
            if row["interchain"]:
                severe.append(row)  # Historical diagnostic: raw interchain pairs, no topology exclusions.
            elif frozenset((keys[i], keys[j])) not in excluded:
                severe_intra.append(row)
        if overlap > .4 and frozenset((keys[i], keys[j])) not in excluded:
            heavy.append(row)
    heavy.sort(key=lambda x: -x["overlap_A"])
    severe.sort(key=lambda x: -x["overlap_A"])
    covalent = []
    for c, pos, name in keys:
        if name != "CA":
            continue
        for a, b, delta, target in (("N", "CA", 0, 1.458), ("CA", "C", 0, 1.525),
                                    ("C", "O", 0, 1.231), ("C", "N", 1, 1.329)):
            pair = ((c, pos, a), (c, pos+delta, b))
            if all(k in atoms for k in pair):
                d = float(np.linalg.norm(np.array(atoms[pair[0]].xyz)-atoms[pair[1]].xyz))
                covalent.append({"atoms": pair, "distance_A": d, "target_A": target, "deviation_A": d-target})
    disulfide_rows = [{"atoms": pair, "distance_A": float(np.linalg.norm(
        np.array(atoms[pair[0]].xyz)-atoms[pair[1]].xyz))} for pair in disulfides]
    expected_ss = {frozenset(pair) for pair in disulfides}
    sg = [k for k in keys if k[2] == "SG" and atoms[k].residue == "CYS"]
    unexpected_ss = []
    for a, b in itertools.combinations(sg, 2):
        distance = float(np.linalg.norm(np.array(atoms[a].xyz)-atoms[b].xyz))
        if distance < 2.3 and frozenset((a, b)) not in expected_ss:
            unexpected_ss.append({"atoms": [a, b], "distance_A": distance})
    return {"ordinary_heavy_atom_overlaps": {
                "threshold_A": .4, "count": len(heavy), "interchain_count": sum(r["interchain"] for r in heavy),
                "intrachain_count": sum(not r["interchain"] for r in heavy), "worst_20": heavy[:20],
                "per_1000_heavy_atoms": 1000*len(heavy)/len(keys),
                "method": "Standard radii; fixed atom-name plus declared disulfide graph; exclude 1–2/1–3/1–4"},
            "severe_interchain_CNO_overlaps": {"threshold_A": 1., "count": len(severe),
                "by_chain_pair": dict(Counter("/".join(sorted(k[0] for k in r["atoms"])) for r in severe)),
                "by_reference_scope": dict(Counter(r["reference_scope"] for r in severe)), "worst_20": severe[:20],
                "method": "Unique raw interchain C/N/O pairs with radius sum minus distance >1 Å; historical diagnostic definition"},
            "severe_intrachain_nonbonded_CNO_count": len(severe_intra),
            "backbone_bond_geometry": {"count": len(covalent), "diagnostic_deviation_cutoff_A": .2,
                "count_outside_cutoff": sum(abs(r["deviation_A"]) > .2 for r in covalent),
                "worst_20": sorted(covalent, key=lambda r: -abs(r["deviation_A"]))[:20],
                "interpretation": "Simple bond-length diagnostic, not calibrated stereochemical validation"},
            "fixed_disulfide_distances": disulfide_rows, "unexpected_close_SG_pairs": unexpected_ss,
            "physical_acceptance": "unassessed; no calibrated thresholds, hydrogen/protonation, rotamer or Ramachandran validation"}


def drift(original, refined, chains):
    result = {}
    for label, names in (("CA", ("CA",)), ("backbone", BACKBONE)):
        _, lig0, lig1 = ev.matched(original, refined, IDENTITY, ["A", "B"], names)
        lig_r, lig_t, lig_rmsd = ev.fit(lig1, lig0)
        _, all0, all1 = ev.matched(original, refined, IDENTITY, list(IDENTITY), names)
        all_r, all_t, all_rmsd = ev.fit(all1, all0)
        rows = {}
        for chain in chains:
            c = chain["input_chain"]
            keys, before, after = ev.matched(original, refined, IDENTITY, [c], names)
            placed = after@lig_r+lig_t
            local_r, _, local_rmsd = ev.fit(placed, before)
            rows[c] = {"atoms": len(keys), "independently_fitted_RMSD_A": local_rmsd,
                       "ligand_frame_RMSD_A": rmsd(placed, before),
                       "assembly_frame_RMSD_A": rmsd(after@all_r+all_t, before),
                       "center_shift_in_ligand_frame_A": float(np.linalg.norm(placed.mean(axis=0)-before.mean(axis=0))),
                       "rotation_in_ligand_frame_degrees": float(np.degrees(np.arccos(np.clip((np.trace(local_r)-1)/2, -1, 1))))}
        result[label] = {"assembly_fitted_RMSD_A": all_rmsd, "ligand_fitted_RMSD_A": lig_rmsd, "per_input_chain": rows}
    return result


def fixed_evaluation(path, atoms, reference, chains, assignment, specs, dockq_python, workdir, available):
    assigned = ev.rename_assignment(atoms, assignment)
    _, lig_ref, lig_mod = ev.matched(reference, atoms, assignment, ["A", "B"])
    lig_r, lig_t, lig_rmsd = ev.fit(lig_mod, lig_ref)
    lengths = {c["input_chain"]: len(c["sequence"]) for c in chains}
    excluded, _ = ev.guidance_info(specs["current"], chains)
    interfaces, folds = [], {}
    for chain in chains:
        c = chain["input_chain"]
        keys, target, model = ev.matched(reference, atoms, assignment, [c])
        rot, trans, error = ev.fit(model, target)
        _, target_bb, model_bb = ev.matched(reference, atoms, assignment, [c], BACKBONE)
        folds[c] = {"input_chain": assignment[c], "observed_reference_CA_count": len(keys),
                    "independently_fitted_CA_RMSD_A": error,
                    "independently_fitted_backbone_RMSD_A": ev.fit(model_bb, target_bb)[2]}
        if chain["gene"] == "ACVR1":
            n60 = [r["input_index_1based"] for r in chain["residues"] if r["uniprot_position"] == 60]
            if len(n60) != 1 or chain["sequence"][n60[0]-1] != "N":
                raise ValueError("Cannot map ACVR1 N60 exactly")
            key = (c, n60[0], "CA")
            if key not in keys:
                raise ValueError("N60 lacks reference coordinates")
            folds[c]["N60"] = {"canonical_position": 60, "input_position": n60[0],
                "CA_error_after_receptor_CA_fit_A": float(np.linalg.norm((model@rot+trans-target)[keys.index(key)]))}
            residuals = np.linalg.norm(model@rot+trans-target, axis=1)
            canonical = {r["input_index_1based"]: r["uniprot_position"] for r in chain["residues"]}
            folds[c]["N60_loop_57_63_CA_errors_after_receptor_fit_A"] = {
                str(canonical[k[1]]): float(e) for k, e in zip(keys, residuals) if 57 <= canonical[k[1]] <= 63}
        if c in "AB":
            continue
        native = ev.native_contacts(reference, c, ["A", "B"])
        predicted = ev.native_contacts(assigned, c, ["A", "B"])
        unused = {k: v for k, v in native.items() if k[:2] not in excluded and k[2:] not in excluded}
        dq = {"status": "unassessed", "reason": "Explicit --dockq skip", "interface": None}
        if available:
            native_path, model_path = workdir/f"{c}_native.pdb", workdir/f"{c}_model.pdb"
            ev.export_binary(native_path, reference, reference, c, ["A", "B"], lengths)
            ev.export_binary(model_path, assigned, reference, c, ["A", "B"], lengths)
            dq = ev.run_dockq(dockq_python, workdir/f"{c}_dockq.json", model_path, native_path)
            if dq.get("status") != "assessed" or not np.isfinite(dq["interface"]["DockQ"]):
                raise RuntimeError(f"Official DockQ failed for {c}: {dq}")
        recovered = sorted(set(native)&set(predicted))
        interfaces.append({"reference_receptor_chain": c, "input_chain": assignment[c], "gene": chain["gene"],
            "official_dockq": dq, "all_native_contacts": ev.contact_recovery(native, predicted),
            "unrewarded_reference_contacts": ev.contact_recovery(unused, predicted),
            "recovered_native_contact_keys": recovered, "lost_native_contact_keys": sorted(set(native)-set(predicted)),
            "receptor_CA_RMSD_in_reference_ligand_frame_A": rmsd(model@lig_r+lig_t, target)})
    scores = {}
    for name, (scorer, spec) in specs["scorers"].items():
        # Deliberately restrict the existing adapter to ONE valid historical
        # bijection. No file mutation and no separately optimized reward copy.
        fixed = copy.copy(scorer)
        if assignment not in scorer._assignments:
            raise ValueError("Historical assignment is not an allowed guidance bijection")
        fixed._assignments = [dict(assignment)]
        scores[name] = fixed.score(path)
    return {"fixed_assignment": assignment, "ligand_CA_fit_RMSD_A": lig_rmsd,
            "interfaces": interfaces, "fold_against_reference": folds, "fixed_assignment_guidance": scores,
            "local_geometry_under_original_reference_assignment": scores["current"].get("local_geometry"),
            "reference_close_cysteine_sulfur_geometry": ev.cysteine_geometry(reference, assigned)}


def numeric_deltas(before, after):
    result = {}
    for k in before.keys() & after.keys():
        a, b = before[k], after[k]
        if isinstance(a, dict) and isinstance(b, dict):
            result[k] = numeric_deltas(a, b)
        elif type(a) in (int, float) and type(b) in (int, float):
            result[k] = b-a
    return result


def analyze(args):
    mapping = json.loads(args.mapping.read_text())
    chains = mapping["chains"]
    expected = {c["input_chain"]: c["sequence"] for c in chains}
    if set(expected) != set(IDENTITY) or sum(map(len, expected.values())) != 716:
        raise ValueError("Expected the unchanged 716-residue A–F BMP6 construct")
    if {c["input_chain"]: c["gene"] for c in chains} != dict(zip("ABCDEF", ["BMP6"]*2+["ACVR1"]*2+["ACVR2B"]*2)):
        raise ValueError("Unexpected BMP6 receptor identities")
    if mapping["expected_saved_chain_to_input_chain"] != IDENTITY or _fasta(args.fasta) != expected:
        raise ValueError("Expected exact identity chain map and matching FASTA sequences")
    if mapping["fasta_sha256"] != ev.sha256(args.fasta):
        raise ValueError("Mapping FASTA hash mismatch")
    historical = json.loads(args.original_evaluation.read_text())
    old = historical_model(historical, args.sample_index)
    for key, path in (("reference", args.reference), ("mapping", args.mapping)):
        if historical[key]["sha256"] != ev.sha256(path):
            raise ValueError(f"Historical evaluation {key} hash mismatch")
    assignment = old["coherent_assignment"]["reference_to_model_input_chain"]
    validate_assignment(assignment, chains)
    originals, original_checks = strict_atoms(args.original, chains)
    refined, refined_checks = strict_atoms(args.refined, chains)
    source = args.original_source or args.original
    if ev.sha256(source) != old["model_sha256"]:
        raise ValueError("Original source hash does not match selected historical evaluation")
    if source != args.original:
        source_atoms, _ = strict_atoms(source, chains)
        if not coordinate_identity(source_atoms, originals):
            raise ValueError("Canonicalized original differs from historical source heavy-atom coordinates/identities")
    reference, _ = ev.read_atoms(args.reference)
    coverage, covered_chains, species = ev.validate_reference(reference, mapping)
    if set(covered_chains) != set(IDENTITY) or species:
        raise ValueError("Expected all six chains covered in exact-sequence BMP6 9N4K reference")
    specs = {"current": json.loads(args.spec.read_text()), "parent": json.loads(args.parent_spec.read_text())}
    if specs["current"].get("schema_version") != 3 or not specs["current"].get("local_geometry_modules"):
        raise ValueError("Expected BMP6 v3 local-geometry spec")
    for spec in specs.values():
        ev.guidance_info(spec, chains)
    for module in specs["current"]["local_geometry_modules"]:
        for c in module["copies"]:
            for measurement in [c["distance"], *c["angles"]]:
                if any((p["chain"], p["position"], p["atom"]) not in reference for p in measurement["atoms"]):
                    raise ValueError("Local geometry endpoint lacks reference coordinates")
    specs["scorers"] = {"current": (MappedDistanceReward(args.fasta, args.spec), specs["current"]),
                         "parent": (MappedDistanceReward(args.fasta, args.parent_spec), specs["parent"])}
    topology = json.loads(args.topology.read_text()) if args.topology else None
    if topology:
        for key, path in (("mapping", args.mapping), ("fasta", args.fasta), ("reference", args.reference)):
            if topology["sources"][key]["sha256"] != ev.sha256(path):
                raise ValueError(f"Topology {key} hash mismatch")
    disulfides, exclusions, topology_basis = fixed_topology(chains, originals, reference, assignment, topology)
    with tempfile.TemporaryDirectory(prefix="bmp6-refinement-analysis-") as tmp:
        tmp = Path(tmp)
        backend = {"status": "unassessed", "reason": "Explicit --dockq skip"}
        if args.dockq != "skip":
            backend = ev.run_dockq(args.dockq_python, tmp/"backend.json")
            if backend.get("status") != "available":
                raise RuntimeError(f"Official DockQ required: {backend}")
        stages = {}
        for stage, path, atoms, checks in (("original", args.original, originals, original_checks),
                                           ("refined", args.refined, refined, refined_checks)):
            work = tmp/stage
            work.mkdir()
            result = fixed_evaluation(path, atoms, reference, chains, assignment, specs,
                                      args.dockq_python, work, args.dockq != "skip")
            result.update({"model": receipt(path), "sequence_and_atom_checks": checks,
                           "physical_checks": physical(atoms, exclusions, disulfides, reference, assignment)})
            stages[stage] = result
    per_copy = []
    for a, b in zip(stages["original"]["interfaces"], stages["refined"]["interfaces"]):
        per_copy.append({"reference_receptor_chain": a["reference_receptor_chain"], "input_chain": a["input_chain"],
                         "after_minus_before": numeric_deltas(a, b)})
    core_keys = {(assignment[c], p, a) for c, p, a in reference}
    core_original = {k: a for k, a in originals.items() if k in core_keys}
    core_refined = {k: a for k, a in refined.items() if k in core_keys}
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_id": "BMP6_ACVR1_ACVR2B", "inputs": {k: receipt(getattr(args, k)) for k in
            ("original", "refined", "mapping", "reference", "spec", "parent_spec", "fasta", "original_evaluation")},
        "original_source": receipt(source), "topology": receipt(args.topology) if args.topology else None,
        "historical_sample_index": args.sample_index, "historical_model_path": old["model_path"],
        "original_fixed_assignment": assignment, "historical_interfaces": old["interfaces"],
        "reference_scope": {"coverage": coverage, "observed_residues": sum(c["observed_CA_residues"] for c in coverage.values()),
            "source": "9N4K observed core; engineered, glycosylated receptor trap; free unglycosylated model chains",
            "interpretation": "All four observed interfaces; 552/716 observed residues. Reference recovery, not biological acceptance or independent validation."},
        "dockq_backend": backend, "dockq_python": args.dockq_python,
        "methods": {"assignment": "Historical coherent reference-to-input bijection fixed for both stages and both guidance recipes; no reassignment optimization",
            "DockQ": "Existing official DockQ worker, fixed XY:XY no_align=True; merged ligand dimer and each receptor; only reference-observed residues",
            "topology": topology_basis, "drift": "Exact original chain identities; full construct CA and backbone fitted globally, to the ligand dimer, or per chain",
            "N60": "Canonical ACVR1 N60 mapped from supplied residue map; CA error after independent observed-core receptor CA fit",
            "unrewarded_contacts": "Endpoint-excluded relative to current score; scoring-withheld only, not independent evidence",
            "missing_atoms": "All standard nonhydrogen residue atoms required; optional terminal OXT tracked separately; hydrogens excluded"},
        "stages": stages, "comparisons": {"per_copy": per_copy,
            "fold_after_minus_before": numeric_deltas(stages["original"]["fold_against_reference"], stages["refined"]["fold_against_reference"]),
            "physical_after_minus_before": numeric_deltas(stages["original"]["physical_checks"], stages["refined"]["physical_checks"]),
            "fixed_guidance_reward_deltas": {k: stages["refined"]["fixed_assignment_guidance"][k]["reward"]-
                stages["original"]["fixed_assignment_guidance"][k]["reward"] for k in ("current", "parent")},
            "drift_from_original": drift(originals, refined, chains),
            "observed_core_drift_from_original": drift(core_original, core_refined, chains),
            "heavy_atom_key_changes": {"lost": sorted(set(originals)-set(refined)), "gained": sorted(set(refined)-set(originals))}},
        "limitations": ["No structural or physical acceptance assigned; geometric overlaps are not MolProbity clashscore.",
            "Hydrogen chemistry, protonation, rotamer/Ramachandran probabilities, glycans, membrane placement and binding are unassessed.",
            "Refinement cannot independently establish the correct docking arrangement; reference/source evidence was reused for tuning.",
            "Guidance rewards are constrained to the historical reference assignment and may differ from sampling's independently optimized reward."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "refined", "mapping", "reference", "spec", "parent-spec", "fasta", "original-evaluation", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--dockq-python", default=sys.executable)
    parser.add_argument("--dockq", choices=("required", "skip"), default="required")
    parser.add_argument("--original-source", type=Path, help="Raw historical PDB if original was canonicalized; exact heavy-atom coordinate identity required")
    parser.add_argument("--topology", type=Path, help="Annotation-verified topology_and_restraints.json; otherwise explicitly labeled reference-distance fallback")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new receipt path")
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "sha256": ev.sha256(args.output), "dockq_backend": result["dockq_backend"]}))


if __name__ == "__main__":
    main()
