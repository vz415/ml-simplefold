#!/usr/bin/env python3
"""Compare protein chemistry and docking before/after restrained refinement.

This is coordinate analysis only. Raw inputs are never changed. Chain identity
is explicit; residue correspondence follows the exact within-chain sequence,
not Rosetta's output residue numbers or a best-fitting exchange of copies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import gemmi
import numpy as np

try:
    from .compare_structures import amino_acid, kabsch
    from .complex_interfaces import _contacts
    from .fk_2h62_l61_reward import L61ComplexReward
    from .fk_2h62_reward import atom_records, chemical_bonds, standard_bond_target
except ImportError:
    from compare_structures import amino_acid, kabsch
    from complex_interfaces import _contacts
    from fk_2h62_l61_reward import L61ComplexReward
    from fk_2h62_reward import atom_records, chemical_bonds, standard_bond_target


def canonicalize(path, output, expected_sequences, chain_map=None):
    """Canonicalize known chain IDs/numbers in a temporary analysis copy only."""
    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError(f"{path}: expected one model")
    canonical = gemmi.Structure()
    model = gemmi.Model("1")
    mapping, ignored = {}, []
    for chain in structure[0]:
        residues = [residue for residue in chain if amino_acid(residue.name) is not None]
        ignored.extend(f"{chain.name}:{r.seqid}:{r.name}" for r in chain
                       if amino_acid(r.name) is None)
        if not residues:
            continue
        name = (chain_map or {}).get(chain.name, chain.name)
        if name not in expected_sequences or name in mapping:
            raise ValueError(f"{path}: unexpected/duplicate protein chain {name}; supply a known chain map")
        sequence = "".join(amino_acid(residue.name) for residue in residues)
        if sequence != expected_sequences[name]:
            raise ValueError(f"{path}: chain {name} sequence differs from original construct")
        new_chain = gemmi.Chain(name)
        rows = []
        for index, residue in enumerate(residues):
            if not any(atom.name == "CA" and atom.occ > 0 for atom in residue):
                raise ValueError(f"{path}: missing occupied CA at {chain.name}:{residue.seqid}")
            copy = residue.clone()
            copy.seqid = gemmi.SeqId(index + 1, " ")
            new_chain.add_residue(copy)
            rows.append(dict(source_residue=str(residue.seqid), construct_position=index + 1))
        model.add_chain(new_chain)
        mapping[name] = dict(source_chain=chain.name, residues=rows)
    if set(mapping) != set(expected_sequences):
        raise ValueError(f"{path}: protein chains differ from original construct")
    canonical.add_model(model)
    canonical.write_pdb(str(output))
    return dict(chain_residue_mapping=mapping, ignored_nonprotein_residues=ignored)


def rmsd(first, second):
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def drift(original, current):
    """Keep identical-copy correspondence fixed; distinguish docking/fold motion."""
    names = sorted(original)
    concatenate = lambda chains, keys: np.concatenate([chains[key].coordinates for key in keys])
    original_all, current_all = concatenate(original, names), concatenate(current, names)
    global_r, global_t, _ = kabsch(current_all, original_all)
    ligand_r, ligand_t, _ = kabsch(concatenate(current, "AB"), concatenate(original, "AB"))
    rows = {}
    for name in names:
        before, after = original[name].coordinates, current[name].coordinates
        local_r, local_t, _ = kabsch(after, before)
        placed = after @ ligand_r + ligand_t
        residual_r, _, _ = kabsch(placed, before)
        rows[name] = dict(
            residues=len(before),
            CA_RMSD_after_assembly_fit_angstrom=rmsd(after @ global_r + global_t, before),
            CA_RMSD_in_BMP2_frame_angstrom=rmsd(placed, before),
            CA_RMSD_after_independent_chain_fit_angstrom=rmsd(after @ local_r + local_t, before),
            center_drift_in_BMP2_frame_angstrom=float(np.linalg.norm(placed.mean(axis=0) - before.mean(axis=0))),
            orientation_drift_in_BMP2_frame_degrees=float(np.degrees(np.arccos(np.clip((np.trace(residual_r) - 1.) / 2., -1., 1.)))))
    return dict(
        assembly_CA_RMSD_angstrom=rmsd(current_all @ global_r + global_t, original_all),
        BMP2_dimer_CA_RMSD_angstrom=rmsd(concatenate(current, "AB") @ ligand_r + ligand_t,
                                        concatenate(original, "AB")),
        copy_correspondence="Original chain identity; equivalent copies are never exchanged for drift",
        per_chain=rows)


def bond_diagnostics(records, reward):
    bonds = chemical_bonds(records, reward.fixed_disulfides)
    tolerance = reward.settings["bond_tolerance_angstrom"]
    deviations, disulfides = [], []
    for pair, kind in bonds.items():
        target = (2.05 if kind[0] == "disulfide" else reward.bond_targets.get(
            kind, standard_bond_target(kind, records, pair)))
        distance = float(np.linalg.norm(np.array(records[pair[0]].xyz) - records[pair[1]].xyz))
        deviations.append(abs(distance - target))
        if kind[0] == "disulfide":
            disulfides.append(dict(atoms=[records[index].label for index in pair],
                                   distance_angstrom=distance, target_angstrom=target))
    values = np.array(deviations)
    return dict(bonds_evaluated=len(values), bonds_outside_tolerance=int(np.sum(values > tolerance)),
                fraction_outside_tolerance=float(np.mean(values > tolerance)),
                bond_length_deviation_RMS_angstrom=float(np.sqrt(np.mean(values ** 2))),
                maximum_bond_length_deviation_angstrom=float(values.max()),
                tolerance_angstrom=tolerance, disulfides=disulfides)


def fixed_assignment_contacts(atoms, reward, assignment):
    """Per-copy whole-interface contacts under the ORIGINAL coherent assignment."""
    ligand = {key: assignment["experimental_reference_to_prediction"][key] for key in "AD"}
    rows = []
    for component, reference_chain in (("BMPR1A", "B"), ("ACVR2B", "C")):
        native = reward.native_contacts[reference_chain]
        native_keys = [key for key in reward.native if key[0] in ("A", "D", reference_chain)]
        for site, receptor in enumerate(assignment[f"{component}_site_to_prediction"]):
            copy = reward._copy_features(atoms, receptor, ligand, site, component)
            paired = {key: atoms[reward._key(key, receptor, ligand, site)] for key in native_keys
                      if reward._key(key, receptor, ligand, site) in atoms}
            predicted = _contacts(paired, reference_chain, ("A", "D"))
            recovered = native & predicted
            copy["whole_interface_contacts"] = dict(
                native_contacts=len(native), predicted_contacts=len(predicted), recovered_contacts=len(recovered),
                native_contact_recall=len(recovered) / len(native) if native else None,
                native_contact_precision=len(recovered) / len(predicted) if predicted else None,
                recovered_contact_pairs=[list(pair) for pair in sorted(recovered)],
                matched_reference_heavy_atoms=len(paired), reference_heavy_atoms=len(native_keys))
            rows.append(copy)
    return rows


def numeric_deltas(before, after):
    """Numeric differences, after minus before, without subtracting booleans."""
    return {key: (numeric_deltas(before[key], value) if isinstance(value, dict) and isinstance(before.get(key), dict)
                  else value - before[key])
            for key, value in after.items() if key in before and
            ((isinstance(value, dict) and isinstance(before[key], dict)) or
             (isinstance(value, (int, float)) and not isinstance(value, bool) and
              isinstance(before[key], (int, float)) and not isinstance(before[key], bool)))}


def analyze(original, prepared, refined, reference_cif, reference_pdb, source_receipt, chain_maps=None):
    receipt = json.loads(Path(source_receipt).read_text())
    settings = receipt["config"]["reward"]
    reward = L61ComplexReward(reference_cif, reference_pdb, settings)
    originals = reward._read_prediction(original)
    expected = {name: record.sequence for name, record in originals.items()}
    original_score = reward.score(original)
    fixed_assignment = original_score["assignment"]
    stages = {}
    with tempfile.TemporaryDirectory(prefix="simplefold-refinement-analysis-") as temporary:
        for stage, source in (("original", original), ("prepared", prepared), ("refined", refined)):
            canonical = Path(temporary) / f"{stage}.pdb"
            correspondence = canonicalize(source, canonical, expected, (chain_maps or {}).get(stage))
            chains = reward._read_prediction(canonical)
            records, atoms, _ = atom_records(canonical, chains)
            result = reward.score(canonical)
            stages[stage] = dict(
                source=str(Path(source).resolve()), sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                correspondence=correspondence, reward=result,
                bond_diagnostics=bond_diagnostics(records, reward),
                original_assignment_copy_metrics=fixed_assignment_contacts(atoms, reward, fixed_assignment),
                drift_from_original=drift(originals, chains))
    comparisons = {}
    for before, after in (("original", "prepared"), ("prepared", "refined"), ("original", "refined")):
        a, b = stages[before], stages[after]
        comparisons[f"{before}_to_{after}"] = dict(
            reward_delta=b["reward"]["reward"] - a["reward"]["reward"],
            geometry_deltas=numeric_deltas(a["reward"]["geometry"], b["reward"]["geometry"]),
            bond_diagnostic_deltas=numeric_deltas(a["bond_diagnostics"], b["bond_diagnostics"]),
            original_assignment_copy_deltas=[dict(
                component=x["component"], prediction_chain=x["prediction_chain"], site=x["site"],
                feature_deltas=numeric_deltas(x["features"], y["features"]),
                contact_deltas=numeric_deltas(x["whole_interface_contacts"], y["whole_interface_contacts"]))
                for x, y in zip(a["original_assignment_copy_metrics"], b["original_assignment_copy_metrics"])])
    return dict(
        source_job_id=receipt.get("job_id"), source_receipt=str(Path(source_receipt).resolve()),
        reward_settings=settings, reward_metadata=reward.metadata, original_assignment=fixed_assignment,
        methods=dict(
            chemistry="Original fixed atom-name topology and native disulfides; same empirical bond targets and heavy-atom clash threshold at every stage. Hydrogens are excluded.",
            contacts="Observed 2H62 atom names and positions; site 1 is inferred by ligand identity exchange. Original coherent assignment fixed for per-copy comparisons; independent optimized FK assignment also reported.",
            atom_counts="Full stage atom sets; atom reconstruction may change counts. Raw and per-1000-heavy-atom clashes are both reported.",
            residue_correspondence="Exact construct sequence and within-chain order; output residue numbers are recorded, never used to guess correspondences.",
            interpretation="Coordinate/chemistry diagnostics, not affinity, confidence, Rosetta energy or MolProbity clashscore."),
        stages=stages, comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "prepared", "refined", "reference-cif", "reference-pdb", "source-receipt", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--chain-map", type=Path, help="Optional JSON of stage -> output chain -> original chain; never inferred for identical copies")
    args = parser.parse_args()
    result = analyze(args.original, args.prepared, args.refined, args.reference_cif, args.reference_pdb,
                     args.source_receipt, json.loads(args.chain_map.read_text()) if args.chain_map else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
