#!/usr/bin/env python3
"""Analyze 2GOO hexamer coordinate ensembles; no inference or model loading."""
from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
from pathlib import Path

import gemmi
import numpy as np

try:
    from .compare_structures import (ChainCA, amino_acid, altloc_priority, digest,
                                     kabsch, reference_ca, require_coordinates, transformed_pdb)
except ImportError:
    from compare_structures import (ChainCA, amino_acid, altloc_priority, digest,
                                    kabsch, reference_ca, require_coordinates, transformed_pdb)

GROUPS = (("BMP2", ("A", "D"), ("A", "B")),
          ("BMPR1A", ("B", "E"), ("C", "D")),
          ("ActRIIA", ("C", "F"), ("E", "F")))
REF_ORDER = "ABCDEF"


def mappings():
    """Eight sequence-identity-preserving copy assignments."""
    for swaps in itertools.product((False, True), repeat=3):
        mapping = {}
        for swap, (_, ref, pred) in zip(swaps, GROUPS):
            mapping.update(zip(ref, pred[::-1] if swap else pred))
        yield mapping


def read_prediction(path, references):
    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError(f"{path}: expected exactly one PDB model")
    result = {}
    for chain in structure[0]:
        letters, coordinates, authors = [], [], []
        seen = set()
        for residue in chain:
            letter = amino_acid(residue.name)
            if letter is None:
                continue
            if str(residue.seqid) in seen:
                raise ValueError(f"{path}: duplicate residue {chain.name}:{residue.seqid}")
            seen.add(str(residue.seqid))
            atoms = [a for a in residue if a.name == "CA"]
            if not atoms:
                raise ValueError(f"{path}: missing CA at {chain.name}:{residue.seqid}")
            atom = min(atoms, key=lambda a: altloc_priority(a.occ, a.altloc.strip("\x00")))
            if atom.occ <= 0:
                raise ValueError(f"{path}: zero occupancy prediction CA")
            letters.append(letter)
            coordinates.append([atom.pos.x, atom.pos.y, atom.pos.z])
            authors.append(str(residue.seqid))
        if letters:
            if chain.name in result:
                raise ValueError(f"{path}: duplicate chain {chain.name}")
            sequence = "".join(letters)
            xyz = np.asarray(coordinates)
            require_coordinates(xyz)
            result[chain.name] = ChainCA(sequence, xyz, list(range(len(sequence))),
                                       authors, chain.name, sequence)
    if set(result) != set(REF_ORDER):
        raise ValueError(f"{path}: expected six protein chains A–F, found {list(result)}")
    for _, refs, preds in GROUPS:
        for rc in refs:
            for pc in preds:
                if result[pc].sequence != references[rc].full_sequence:
                    raise ValueError(f"{path}: chain {pc} sequence differs from full reference chain {rc}")
    return result


def matched_coordinates(chains, references, mapping):
    return np.concatenate([chains[mapping[c]].coordinates[references[c].full_positions]
                           for c in REF_ORDER])


def diversity_references(references):
    """Use the same resolved full-sequence positions for both identical copies."""
    result = {}
    for _, ref_chains, _ in GROUPS:
        positions = sorted(set(references[ref_chains[0]].full_positions)
                           & set(references[ref_chains[1]].full_positions))
        if len(positions) < 3:
            raise ValueError("Need at least three shared observed positions per component")
        for c in ref_chains:
            source = references[c]
            indices = {pos: i for i, pos in enumerate(source.full_positions)}
            chosen = [indices[pos] for pos in positions]
            result[c] = ChainCA("".join(source.full_sequence[p] for p in positions),
                                source.coordinates[chosen], positions,
                                [source.author_residues[i] for i in chosen], c, source.full_sequence)
    return result


def best_fit(chains, references, fixed):
    candidates = []
    for mapping in mappings():
        moving = matched_coordinates(chains, references, mapping)
        rotation, translation, distances = kabsch(moving, fixed)
        candidates.append((float(np.sqrt(np.mean(distances**2))), mapping, moving,
                           rotation, translation))
    return min(candidates, key=lambda item: item[0])


def distance_metrics(fixed, moving, labels):
    ref_dist = np.linalg.norm(fixed[:, None] - fixed[None, :], axis=-1)
    pred_dist = np.linalg.norm(moving[:, None] - moving[None, :], axis=-1)
    upper = np.triu(np.ones(ref_dist.shape, dtype=bool), k=1)
    local = upper & (ref_dist < 15)
    if not local.any():
        raise ValueError("No reference CA pairs within 15 Å")
    errors = np.abs(pred_dist - ref_dist)[local]
    lddt = float(np.mean([np.mean(errors < t) for t in (0.5, 1, 2, 4)]))
    inter = upper & (labels[:, None] != labels[None, :])
    ref_contacts = inter & (ref_dist < 8)
    pred_contacts = inter & (pred_dist < 8)
    intersection = int(np.sum(ref_contacts & pred_contacts))
    return {
        "assembly_CA_lddt": lddt,
        "CA_interchain_contact_recall": intersection / int(ref_contacts.sum()) if ref_contacts.any() else None,
        "CA_interchain_contact_precision": intersection / int(pred_contacts.sum()) if pred_contacts.any() else None,
        "reference_CA_interchain_contacts": int(ref_contacts.sum()),
        "predicted_CA_interchain_contacts": int(pred_contacts.sum()),
    }


def summary(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return dict(n=0, mean=None, sd=None, median=None, min=None, max=None)
    return dict(n=len(values), mean=float(values.mean()),
                sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                median=float(np.median(values)), min=float(values.min()), max=float(values.max()))


def analyze(args):
    references = {c: reference_ca(args.reference, c) for c in REF_ORDER}
    fixed = np.concatenate([references[c].coordinates for c in REF_ORDER])
    full_length = sum(len(references[c].full_sequence) for c in REF_ORDER)
    if full_length != 694 or len(fixed) != 565:
        raise ValueError(f"Expected 2GOO 694 deposited/565 observed CAs; found {full_length}/{len(fixed)}")
    if args.expected_samples < 1:
        raise ValueError("--expected-samples must be positive")
    diversity_refs = diversity_references(references)
    diversity_count = sum(len(diversity_refs[c].coordinates) for c in REF_ORDER)
    labels = np.concatenate([np.repeat(c, len(references[c].coordinates)) for c in REF_ORDER])
    # Confirm that the displayed protein-only reference has the scored coordinates.
    displayed = gemmi.read_structure(str(args.reference_pdb))
    if len(displayed) != 1:
        raise ValueError("Displayed reference must have exactly one model")
    display_coords = {}
    for chain in displayed[0]:
        for residue in chain:
            if residue.het_flag != "A":
                raise ValueError("--reference-pdb must be protein-only")
            atoms = [a for a in residue if a.name == "CA"]
            if atoms:
                atom = min(atoms, key=lambda a: altloc_priority(a.occ, a.altloc.strip("\x00")))
                display_coords[(chain.name, str(residue.seqid))] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    expected_keys = {(c, r) for c in REF_ORDER for r in references[c].author_residues}
    if set(display_coords) != expected_keys:
        raise ValueError("Displayed reference CA residue identities differ from mmCIF")
    for c in REF_ORDER:
        actual = np.array([display_coords[(c, r)] for r in references[c].author_residues])
        if not np.allclose(actual, references[c].coordinates, atol=0.001, rtol=0):
            raise ValueError("Displayed reference coordinates differ from mmCIF")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_output = args.output_dir / "reference.pdb"
    if args.reference_pdb.resolve() != reference_output.resolve():
        shutil.copyfile(args.reference_pdb, reference_output)
    models, names = [], set()
    for specification in args.prediction_dir:
        name, separator, directory = specification.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in names:
            raise ValueError("--prediction-dir must be unique MODEL=PATH (safe model name)")
        names.add(name)
        root = Path(directory)
        paths = sorted(root.rglob("*.pdb"))
        if len(paths) != args.expected_samples:
            raise ValueError(f"{name}: expected {args.expected_samples} sample PDBs, found {len(paths)}")
        output = args.output_dir / "aligned_predictions" / name
        output.mkdir(parents=True, exist_ok=True)
        raw_output = args.output_dir / "raw_predictions" / name
        raw_output.mkdir(parents=True, exist_ok=True)
        samples, parsed, matched = [], [], []
        for index, path in enumerate(paths):
            chains = read_prediction(path, references)
            rmsd, mapping, moving, rotation, translation = best_fit(chains, references, fixed)
            sample_id = path.relative_to(root).with_suffix("").as_posix()
            aligned = output / f"sample_{index:03d}.pdb"
            raw = raw_output / f"sample_{index:03d}.pdb"
            if path.resolve() != raw.resolve():
                shutil.copyfile(path, raw)
            transformed_pdb(path, aligned, rotation, translation)
            metrics = {"global_CA_RMSD_angstrom": rmsd, **distance_metrics(fixed, moving, labels)}
            per_chain = []
            for c in REF_ORDER:
                pred = chains[mapping[c]].coordinates[references[c].full_positions]
                _, _, errors = kabsch(pred, references[c].coordinates)
                per_chain.append(dict(reference_chain=c, prediction_chain=mapping[c],
                                      matched_CA=len(pred), CA_RMSD_angstrom=float(np.sqrt(np.mean(errors**2)))))
            for component, ref_chains, _ in GROUPS:
                metrics[f"{component}_chain_CA_RMSD_angstrom"] = float(np.mean([
                    row["CA_RMSD_angstrom"] for row in per_chain if row["reference_chain"] in ref_chains]))
            samples.append(dict(id=sample_id, source=str(path.resolve()), source_sha256=digest(path),
                                raw_pdb=raw.relative_to(args.output_dir).as_posix(),
                                aligned_pdb=aligned.relative_to(args.output_dir).as_posix(),
                                mapping=mapping, metrics=metrics, per_chain=per_chain))
            parsed.append(chains)
            matched.append(matched_coordinates(chains, diversity_refs, mapping))
        pairs = []
        for i, j in itertools.combinations(range(len(samples)), 2):
            # Both copies use their common resolved full-sequence positions,
            # making the eight-copy-permutation minimum independent of pair order.
            rmsd, _, _, _, _ = best_fit(parsed[j], diversity_refs, matched[i])
            pairs.append(dict(first=samples[i]["id"], second=samples[j]["id"], CA_RMSD_angstrom=rmsd))
        models.append(dict(name=name, samples=samples,
                           summary={key: summary([s["metrics"][key] for s in samples]) for key in samples[0]["metrics"]},
                           diversity=dict(pair_count=len(pairs), summary=summary([p["CA_RMSD_angstrom"] for p in pairs]), pairs=pairs)))
    report = dict(reference_pdb="reference.pdb", matched_reference_CA=len(fixed),
                  matched_diversity_CA=diversity_count, prediction_CA=full_length,
                  reference_cif_sha256=digest(args.reference), reference_pdb_sha256=digest(args.reference_pdb),
                  methods={
                      "correspondence": "Exact full polymer sequences; reference label_seq_id selects the 565 observed CAs. No unresolved terminal residue is scored.",
                      "alignment": "Minimum global proper-rotation Kabsch CA RMSD over eight identity-preserving copy permutations; saved prediction chain labels are preserved.",
                      "lddt": "CA pair-weighted lDDT: reference pairs <15 Å, strict error thresholds 0.5/1/2/4 Å, each unordered pair counted once; includes intra- and interchain pairs.",
                      "contacts": "Interchain CA proximity <8 Å on the observed-residue mask; recall and precision, not DockQ or all-atom contacts. Undefined ratios are null.",
                      "component_rmsd": "Each reference chain independently fitted; component metric is the unweighted average of its two copy RMSDs.",
                      "diversity": "All n(n-1)/2 pairs. For each component both copies use the intersection of observed full-sequence positions (BMP2 103, BMPR1A 85, ActRIIA 92 per copy; 560 CAs total). Minimum proper-rotation Kabsch RMSD over eight copy permutations is invariant to sample order and renaming equivalent copies. Reference comparisons separately use all 565 observed CAs. Diversity has no preferred direction.",
                      "summary": "Mean, sample standard deviation (ddof=1), median, minimum, maximum; null values excluded and effective n retained; SD null for n<2.",
                      "limitations": "Experimental coordinates retained without repairs. Missing termini and density/occupancy/stereochemistry issues limit reference interpretation. Metrics measure coordinate agreement, not functional binding or physical validity."
                  }, models=models)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-pdb", type=Path, required=True)
    parser.add_argument("--prediction-dir", action="append", required=True, metavar="MODEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=10)
    report = analyze(parser.parse_args())
    for model in report["models"]:
        print(f"{model['name']}: {len(model['samples'])} samples; {model['diversity']['pair_count']} diversity pairs")


if __name__ == "__main__":
    main()
