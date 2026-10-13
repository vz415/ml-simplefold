#!/usr/bin/env python3
"""Evaluate mapped BMP assemblies without per-interface chain reassignment.

Requires numpy/scipy in this interpreter. Official DockQ is optional and may run
in a separate interpreter (--dockq-python); absent DockQ is null/unassessed.
No custom approximation is reported as DockQ. This script never samples or
installs anything and assigns no calibrated acceptance label.

Example (run with a numpy/scipy interpreter; actual inference outputs must have
the complete FASTA sequence, so omit --allow-partial-model for model results)::

    python scripts/bmp90_evaluate.py --models prediction.pdb \
      --reference artifacts/bmp90/preparation/bmp6/9N4K_deposited_native_reference.pdb \
      --mapping examples/bmp90/BMP6_ACVR1_ACVR2B_deposited_native_no_tags.mapping.json \
      --guidance-spec artifacts/bmp90/specifications/bmp6_joint_v1.json \
      --dockq required --dockq-python /path/to/analysis-venv/bin/python \
      --output artifacts/bmp90/evaluation/population.json

Supply all raw particles to --models and keep selected-particle reporting
separate. A self-reference check needs --allow-partial-model because only 552 of
716 sequence residues are experimentally resolved. Test with:
python -m unittest discover -s tests -p test_bmp90_evaluate.py -v
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

from structure_clashes import _read, analyze_clashes

AA1 = dict(zip("ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(), "ACDEFGHIKLMNPQRSTVWY"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_atoms(path):
    atoms, _, _, omitted = _read(path)
    result = {}
    for a in atoms:
        if a.insertion:
            raise ValueError(f"Insertion code is unsupported in mapped input: {a.label}")
        key = (a.chain, a.number, a.name)
        if key in result:
            raise ValueError(f"Ambiguous atom {key} in {path}")
        if not all(math.isfinite(x) for x in a.xyz):
            raise ValueError(f"Nonfinite coordinates for {key}")
        result[key] = a
    if not result:
        raise ValueError(f"No protein atoms in {path}")
    return result, omitted


def validate_atoms(atoms, chains, require_complete):
    sequences = {c["input_chain"]: c["sequence"] for c in chains}
    for (chain, pos, _), a in atoms.items():
        if chain not in sequences or not 1 <= pos <= len(sequences[chain]):
            raise ValueError(f"Unexpected chain/position {chain}:{pos}")
        if AA1.get(a.residue) != sequences[chain][pos - 1]:
            raise ValueError(f"Amino-acid mismatch {chain}:{pos} {a.residue}")
    coverage = {}
    for chain, seq in sequences.items():
        missing = [i for i in range(1, len(seq)+1) if (chain, i, "CA") not in atoms]
        coverage[chain] = {"input_residues": len(seq), "observed_CA_residues": len(seq)-len(missing),
                           "missing_CA_input_indices": missing}
        if require_complete and missing:
            raise ValueError(f"Model {chain} is missing {len(missing)} expected CA residues; use --allow-partial-model only for explicit partial-reference validation")
        if len(seq)-len(missing) < 3:
            raise ValueError(f"Insufficient CA coordinates in chain {chain}")
    return coverage


def remap_saved_atoms(atoms, saved_to_input):
    result = {}
    for (chain, pos, name), a in atoms.items():
        if chain not in saved_to_input:
            raise ValueError(f"Unknown saved chain {chain}")
        result[(saved_to_input[chain], pos, name)] = a
    return result


def fit(mobile, reference):
    mobile, reference = np.asarray(mobile), np.asarray(reference)
    if len(mobile) < 3:
        raise ValueError("At least three matched atoms are required for alignment")
    mc, rc = mobile.mean(axis=0), reference.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile-mc).T @ (reference-rc))
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    translation = rc - mc @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((mobile @ rotation + translation-reference)**2, axis=1))))
    return rotation, translation, rmsd


def matched(reference, model, assignment, chain_ids, atom_names=("CA",)):
    keys = [k for k in sorted(reference) if k[0] in chain_ids and k[2] in atom_names
            and (assignment[k[0]], k[1], k[2]) in model]
    ref = np.asarray([reference[k].xyz for k in keys])
    mod = np.asarray([model[(assignment[k[0]], k[1], k[2])].xyz for k in keys])
    return keys, ref, mod


def coherent_assignment(reference, model, chains, ligand_chains):
    by_identity = defaultdict(list)
    for c in chains:
        by_identity[(c["gene"], c["sequence"])].append(c["input_chain"])
    groups = list(by_identity.values())
    if math.prod(math.factorial(len(g)) for g in groups) > 4096:
        raise ValueError("Too many identity-preserving assignments")
    receptor_chains = [c["input_chain"] for c in chains if c["input_chain"] not in ligand_chains]
    alternatives = []
    for perm_groups in itertools.product(*(itertools.permutations(g) for g in groups)):
        assignment = dict(x for group, perm in zip(groups, perm_groups) for x in zip(group, perm))
        _, ref, mod = matched(reference, model, assignment, ligand_chains)
        rotation, translation, ligand_rmsd = fit(mod, ref)
        receptor_rmsd = {}
        for chain in receptor_chains:
            _, ref, mod = matched(reference, model, assignment, [chain])
            if len(ref) < 3:
                raise ValueError(f"Insufficient shared CA atoms in receptor {chain}")
            receptor_rmsd[chain] = float(np.sqrt(np.mean(np.sum((mod @ rotation+translation-ref)**2, axis=1))))
        objective = ligand_rmsd**2 + sum(x*x for x in receptor_rmsd.values())
        alternatives.append({"reference_to_model_input_chain": assignment,
                             "ligand_CA_fit_rmsd_A": ligand_rmsd,
                             "receptor_CA_rmsd_in_ligand_frame_A": receptor_rmsd,
                             "objective_A_squared": objective})
    alternatives.sort(key=lambda a: (a["objective_A_squared"], tuple(a["reference_to_model_input_chain"].values())))
    return alternatives[0], alternatives


def native_contacts(atoms, receptor, ligands, threshold=5.0):
    rec = [(k, a) for k, a in atoms.items() if k[0] == receptor]
    lig = [(k, a) for k, a in atoms.items() if k[0] in ligands]
    if not rec or not lig:
        return {}
    rx, lx = np.asarray([a.xyz for _, a in rec]), np.asarray([a.xyz for _, a in lig])
    result = {}
    for i, neighbors in enumerate(cKDTree(rx).query_ball_tree(cKDTree(lx), threshold)):
        for j in neighbors:
            distance = float(np.linalg.norm(rx[i]-lx[j]))
            if distance < threshold:
                rk, lk = rec[i][0], lig[j][0]
                key = (rk[0], rk[1], lk[0], lk[1])
                result[key] = min(distance, result.get(key, float("inf")))
    return result


def rename_assignment(model, assignment):
    reverse = {v: k for k, v in assignment.items()}
    return {(reverse[c], p, n): a for (c, p, n), a in model.items()}


def contact_recovery(native, predicted):
    recovered = set(native).intersection(predicted)
    return {"native_contact_count": len(native), "recovered_native_contacts": len(recovered),
            "fraction_native_contacts": len(recovered)/len(native) if native else None,
            "contact_definition": "Minimum protein heavy-atom distance strictly below 5 Angstrom; fixed full native denominator"}


def guidance_info(spec, chains):
    if spec is None:
        return set(), []
    seqs = {c["input_chain"]: c["sequence"] for c in chains}
    if len(spec["chains"]) != len(seqs) or {c["input_chain"] for c in spec["chains"]} != set(seqs):
        raise ValueError("Guidance must cover every evaluation chain exactly once")
    for c in spec["chains"]:
        if c["input_chain"] not in seqs or hashlib.sha256(seqs[c["input_chain"]].encode()).hexdigest() != c["sequence_sha256"]:
            raise ValueError("Guidance sequence hash does not match evaluation mapping")
    excluded = set()
    pairs = spec["pairs"] + spec.get("integrity_pairs", [])
    for pair in pairs:
        for name in ("left", "right"):
            p = pair[name]
            c, i = p["chain"], p["position"]
            if c not in seqs or not 1 <= i <= len(seqs[c]) or seqs[c][i-1] != p["residue"]:
                raise ValueError(f"Guidance endpoint identity mismatch: {pair['id']}")
            for other, seq in seqs.items():
                if seq == seqs[c]:
                    excluded.add((other, i))
    return excluded, pairs


def pair_measurements(pairs, assigned):
    values = []
    for pair in pairs:
        keys = [(pair[s]["chain"], pair[s]["position"], pair[s]["atom"]) for s in ("left", "right")]
        observed = all(k in assigned for k in keys)
        distance = float(np.linalg.norm(np.asarray(assigned[keys[0]].xyz)-assigned[keys[1]].xyz)) if observed else None
        values.append({"id": pair["id"], "evidence_labels": pair.get("evidence_labels", []),
                       "observed": observed, "distance_A": distance,
                       "target_distance_A": pair["distance_angstrom"],
                       "deviation_A": distance-pair["distance_angstrom"] if observed else None,
                       "sigma_A": pair["sigma_angstrom"]})
    return values


def cysteine_geometry(reference, assigned):
    """Observed close SG pairs are geometric diagnostics, not assigned chemistry."""
    sulfurs = [(k, a) for k, a in reference.items() if k[2] == "SG" and a.residue == "CYS"]
    result = []
    for (ka, a), (kb, b) in itertools.combinations(sulfurs, 2):
        native = float(np.linalg.norm(np.asarray(a.xyz)-b.xyz))
        if not 1.8 <= native <= 2.3:
            continue
        actual = float(np.linalg.norm(np.asarray(assigned[ka].xyz)-assigned[kb].xyz)) if ka in assigned and kb in assigned else None
        result.append({"reference_atom_keys": [ka, kb], "reference_distance_A": native,
                       "model_distance_A": actual, "deviation_A": actual-native if actual is not None else None,
                       "interpretation": "Reference close-cysteine sulfur geometry; no covalent topology or energetic claim"})
    return result


def export_binary(path, atoms, reference, receptor, ligand_chains, lengths):
    # Full numbering preserves unresolved gaps. Ligand chains are merged into X;
    # disjoint offsets keep their residue identities separate. Y is one receptor.
    offsets, offset = {}, 0
    for chain in ligand_chains:
        offsets[chain] = offset
        offset += lengths[chain]
    native_residues = {(c, p) for c, p, _ in reference}
    keys = [k for k in sorted(atoms) if (k[0] in ligand_chains or k[0] == receptor)
            and k[:2] in native_residues]
    lines, previous_chain = [], None
    serial = 0
    for c, p, name in keys:
        a = atoms[(c, p, name)]
        out_chain, out_pos = ("X", p+offsets[c]) if c in ligand_chains else ("Y", p)
        if previous_chain is not None and previous_chain != out_chain:
            lines.append("TER")
        previous_chain = out_chain
        serial += 1
        field = f" {name:<3}" if len(a.element) == 1 and len(name) < 4 else f"{name:<4}"
        lines.append(f"ATOM  {serial:5d} {field} {a.residue:>3} {out_chain}{out_pos:4d}    "
                     f"{a.xyz[0]:8.3f}{a.xyz[1]:8.3f}{a.xyz[2]:8.3f}{a.occupancy:6.2f}{0:6.2f}          {a.element:>2}  ")
    Path(path).write_text("\n".join(lines+["TER", "END"]) + "\n")


def run_dockq(python, output, model=None, native=None):
    worker = Path(__file__).with_name("bmp90_dockq_worker.py")
    command = [python, str(worker), "--output", str(output)]
    if model is not None:
        command += ["--model", str(model), "--native", str(native)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if proc.returncode:
        raise RuntimeError(f"Official DockQ worker failed ({proc.returncode}): {proc.stderr[-4000:]}")
    return json.loads(Path(output).read_text())


def evaluate_model(model_path, reference, chains, saved_to_input, ligand_chains,
                   spec=None, allow_partial=False, dockq_python=None, dockq_available=False,
                   work_dir=None):
    raw, omitted = read_atoms(model_path)
    model = remap_saved_atoms(raw, saved_to_input)
    coverage = validate_atoms(model, chains, require_complete=not allow_partial)
    best, alternatives = coherent_assignment(reference, model, chains, ligand_chains)
    assignment = best["reference_to_model_input_chain"]
    input_to_saved = {inp: saved for saved, inp in saved_to_input.items()}
    best["reference_to_model_saved_chain"] = {ref: input_to_saved[inp] for ref, inp in assignment.items()}
    assigned = rename_assignment(model, assignment)
    excluded, pairs = guidance_info(spec, chains)
    lengths = {c["input_chain"]: len(c["sequence"]) for c in chains}
    interfaces = []
    for c in chains:
        rec = c["input_chain"]
        if rec in ligand_chains:
            continue
        native = native_contacts(reference, rec, ligand_chains)
        predicted = native_contacts(assigned, rec, ligand_chains)
        residual = {k: v for k, v in native.items() if k[:2] not in excluded and k[2:] not in excluded}
        dockq = {"status": "unassessed", "reason": "Official DockQ unavailable or disabled", "interface": None}
        if dockq_available:
            binary_native = Path(work_dir)/f"{rec}_native.pdb"
            binary_model = Path(work_dir)/f"{rec}_model.pdb"
            export_binary(binary_native, reference, reference, rec, ligand_chains, lengths)
            export_binary(binary_model, assigned, reference, rec, ligand_chains, lengths)
            dockq = run_dockq(dockq_python, Path(work_dir)/f"{rec}_dockq.json", binary_model, binary_native)
        ref_keys = {k for k in reference if k[0] in ligand_chains or k[0] == rec}
        observed = ref_keys.intersection(assigned)
        interfaces.append({"reference_receptor_chain": rec, "gene": c["gene"],
                           "model_input_chain": assignment[rec], "all_native_contacts": contact_recovery(native, predicted),
                           "unrewarded_reference_contacts": contact_recovery(residual, predicted) if spec is not None else None,
                           "guided_endpoint_contact_count_excluded": len(native)-len(residual) if spec is not None else None,
                           "shared_reference_atom_count": len(observed), "reference_atom_count": len(ref_keys),
                           "reference_atom_coverage_fraction": len(observed)/len(ref_keys),
                           "receptor_CA_rmsd_in_ligand_frame_A": best["receptor_CA_rmsd_in_ligand_frame_A"][rec],
                           "official_dockq": dockq})
    folds = {}
    for c in chains:
        name = c["input_chain"]
        keys, ref, mod = matched(reference, model, assignment, [name], ("N", "CA", "C", "O"))
        folds[name] = {"matched_backbone_atoms": len(keys), "fitted_backbone_rmsd_A": fit(mod, ref)[2]}
    def summary(values):
        return {"mean": sum(values)/len(values), "weakest": min(values)} if values and all(x is not None for x in values) else {"mean": None, "weakest": None}
    dockq_values = [i["official_dockq"].get("interface", {}).get("DockQ") if i["official_dockq"].get("interface") else None for i in interfaces]
    component_map = {saved: next(c["gene"] for c in chains if c["input_chain"] == inp)+":"+inp for saved, inp in saved_to_input.items()}
    return {"model_path": str(model_path), "model_sha256": sha256(model_path), "sequence_coverage": coverage,
            "omitted_atom_counts": omitted, "coherent_assignment": best, "assignment_alternatives": alternatives,
            "assignment_objective": "Ligand CA fitted RMSD squared plus sum of receptor-copy CA mean squared deviations in that same frame; one bijection for every metric",
            "interfaces": interfaces, "fold_preservation": folds,
            "reference_close_cysteine_sulfur_pairs": cysteine_geometry(reference, assigned),
            "guidance_distances_under_reference_assignment": pair_measurements(pairs, assigned),
            "native_contact_recovery_summary": summary([i["all_native_contacts"]["fraction_native_contacts"] for i in interfaces]),
            "official_dockq_summary": summary(dockq_values),
            "heavy_atom_sterics": analyze_clashes(model_path, chain_components=component_map),
            "physical_status": "unassessed", "threshold_status": "pending_calibration", "assembly_label": None,
            "unassessed_checks": ["Hydrogen/protonation-dependent MolProbity validation", "hydrogen-bond geometry", "buried unsatisfied polar groups", "rotamer/backbone outlier probabilities", "disulfide covalent topology assignment", "glycan-mediated interactions", "membrane placement", "experimental binding/signaling"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, nargs="+", required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--guidance-spec", type=Path)
    p.add_argument("--ligand-chains", default="A,B")
    p.add_argument("--dockq-python", default=sys.executable)
    p.add_argument("--dockq", choices=["auto", "required", "skip"], default="auto")
    p.add_argument("--allow-partial-model", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    mapping = json.loads(args.mapping.read_text())
    chains = mapping["chains"]
    reference, omitted = read_atoms(args.reference)
    reference_coverage = validate_atoms(reference, chains, require_complete=False)
    saved = mapping["expected_saved_chain_to_input_chain"]
    if set(saved.values()) != {c["input_chain"] for c in chains} or len(saved) != len(chains):
        raise ValueError("Saved chain map must be a complete bijection")
    spec = json.loads(args.guidance_spec.read_text()) if args.guidance_spec else None
    if spec is not None and {c["saved_chain"]: c["input_chain"] for c in spec["chains"]} != saved:
        raise ValueError("Guidance and evaluation saved-chain mappings disagree")
    excluded, _ = guidance_info(spec, chains)
    ligand_chains = args.ligand_chains.split(",")
    if len(set(ligand_chains)) != len(ligand_chains) or not set(ligand_chains).issubset(saved.values()):
        raise ValueError("Invalid ligand chains")
    with tempfile.TemporaryDirectory(prefix="bmp90-evaluation-") as tmp:
        backend = {"status": "unassessed", "reason": "Disabled", "version": None}
        if args.dockq != "skip":
            backend = run_dockq(args.dockq_python, Path(tmp)/"backend.json")
        available = backend["status"] == "available"
        if args.dockq == "required" and not available:
            raise RuntimeError(f"Official DockQ required: {backend}")
        results = []
        for model in args.models:
            results.append(evaluate_model(model, reference, chains, saved, ligand_chains,
                                          spec, args.allow_partial_model, args.dockq_python, available, tmp))
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "reference": {"path": str(args.reference), "sha256": sha256(args.reference), "coverage": reference_coverage,
                            "omitted_atom_counts": omitted},
              "mapping": {"path": str(args.mapping), "sha256": sha256(args.mapping)},
              "guidance_spec": {"path": str(args.guidance_spec), "sha256": sha256(args.guidance_spec)} if args.guidance_spec else None,
              "ligand_chains": ligand_chains, "dockq_backend": backend, "dockq_python": args.dockq_python,
              "dockq_scope": "Official DockQ per receptor against the merged ligand dimer, fixed XY:XY mapping with no_align=True. Full input numbering is preserved; prediction residues outside reference-observed coverage are excluded from DockQ only.",
              "evidence_allocation": {"guidance_endpoint_exclusions": sorted(excluded),
                                      "exclusion_rule": "Exclude every native contact touching any guidance/integrity endpoint and all identical-sequence equivalents",
                                      "unrewarded_contacts_status": "Coordinate holdout relative to the supplied guidance endpoints only; no claim of independence from tuning, ranking, selection or model training",
                                      "source_independence": "Same reference and C2-related copies are not independent experiments",
                                      "claimed_independent_heldout_evidence": None},
              "limitations": ["Protein-only comparison to an engineered glycosylated receptor trap; no native signaling claim.",
                              "Unresolved reference tails cannot be assessed for reference accuracy but remain included in model sterics.",
                              "Native-contact recovery and DockQ against guidance coordinates quantify reference recovery.",
                              "No acceptance thresholds calibrated; a geometric heavy-atom overlap count is not MolProbity clashscore.",
                              "FK descendants are correlated; no independent-run recovery probability is estimated.",
                              "Global reference assignment can differ from the assignment that maximizes the guidance reward; both should be retained in run records."],
              "models": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "models": len(results), "dockq_backend": backend}))


if __name__ == "__main__":
    main()
