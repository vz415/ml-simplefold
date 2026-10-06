#!/usr/bin/env python3
"""Compare sampled PDBs with one experimental mmCIF author chain, without models.

TM-align reports normalize to the *observed* reference CA chain. Sequence-mapped
RMSD/GDT use a separate least-squares fit, preserve mmCIF label_seq_id, and report
full polymer length so unresolved residues cannot silently inflate coverage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
from Bio import Align, SeqIO
from Bio.PDB import MMCIFParser, PDBIO, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


@dataclass
class ChainCA:
    sequence: str
    coordinates: np.ndarray
    full_positions: list[int]
    author_residues: list[str]
    chain: str
    full_sequence: str


def amino_acid(name):
    info = gemmi.find_tabulated_residue(name)
    return info.one_letter_code.upper() if info.is_amino_acid() else None


def altloc_priority(occupancy, altloc):
    """Highest occupancy; blank, then A, then lexical alternate as tie-breakers."""
    return (-float(occupancy), altloc not in ("", ".", "?"), altloc != "A", altloc)


def sequence_pairs(first, second):
    """Return aligned zero-based sequence indices; gaps do not shift numbering."""
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 3
    aligner.mismatch_score = -3
    aligner.open_gap_score = -8
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(first, second)[0]
    return [
        (i, j)
        for (a, b), (c, d) in zip(*alignment.aligned)
        for i, j in zip(range(a, b), range(c, d))
    ]


def reference_ca(path, author_chain):
    data = MMCIF2Dict(str(path))
    n = len(data["_atom_site.label_atom_id"])
    column = lambda key, default: data["_atom_site." + key] if "_atom_site." + key in data else [default] * n
    names = column("label_atom_id", "")
    chains = column("auth_asym_id", "")
    models = column("pdbx_PDB_model_num", "1")
    model = next((models[i] for i in range(n) if chains[i] == author_chain), None)
    if model is None:
        raise ValueError(f"Reference has no author chain {author_chain!r}")
    selected = {}
    entities = set()
    for i in range(n):
        if chains[i] != author_chain or names[i] != "CA" or models[i] != model:
            continue
        residue = column("label_comp_id", "UNK")[i]
        letter = amino_acid(residue)
        if letter is None:
            continue
        pos = column("label_seq_id", "?")[i]
        if pos in (".", "?"):
            raise ValueError("Reference polymer CA lacks label_seq_id; cannot map unresolved residues")
        pos = int(pos) - 1
        author = column("auth_seq_id", "?")[i]
        insertion = column("pdbx_PDB_ins_code", "?")[i]
        author += insertion if insertion not in (".", "?") else ""
        occupancy = column("occupancy", "1")[i]
        altloc = column("label_alt_id", ".")[i]
        coord = np.array([float(column(axis, "nan")[i]) for axis in ("Cartn_x", "Cartn_y", "Cartn_z")])
        candidate = (altloc_priority(occupancy, altloc), letter, coord, author)
        if pos not in selected or candidate[0] < selected[pos][0]:
            selected[pos] = candidate
        entities.add(column("label_entity_id", "?")[i])
    if len(entities) != 1 or not selected:
        raise ValueError("Reference chain must contain one protein entity with observed CA atoms")
    entity = next(iter(entities))
    entity_ids = data.get("_entity_poly.entity_id", [])
    seqs = data.get("_entity_poly.pdbx_seq_one_letter_code_can", [])
    if entity not in entity_ids or not seqs:
        raise ValueError("Reference lacks canonical full polymer sequence (_entity_poly)")
    full_sequence = "".join(seqs[entity_ids.index(entity)].split()).upper()
    if not full_sequence.isalpha():
        raise ValueError("Canonical polymer sequence contains unsupported non-letter codes")
    positions = sorted(selected)
    if min(positions) < 0 or max(positions) >= len(full_sequence):
        raise ValueError("Reference label_seq_id lies outside full polymer sequence")
    result = ChainCA(
        "".join(selected[i][1] for i in positions),
        np.array([selected[i][2] for i in positions]), positions,
        [selected[i][3] for i in positions], author_chain, full_sequence,
    )
    require_coordinates(result.coordinates)
    return result


def require_coordinates(coordinates):
    if len(coordinates) < 3 or not np.isfinite(coordinates).all():
        raise ValueError("Need at least three finite CA coordinates")


def prediction_ca(path):
    structure = gemmi.read_structure(str(path))
    candidates = []
    for chain in structure[0]:
        letters, coords, author = [], [], []
        for residue in chain:
            letter = amino_acid(residue.name)
            atoms = [atom for atom in residue if atom.name == "CA"]
            if letter is None or not atoms:
                continue
            atom = min(atoms, key=lambda atom: altloc_priority(atom.occ, atom.altloc.strip("\x00")))
            letters.append(letter)
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
            author.append(str(residue.seqid))
        if letters:
            sequence = "".join(letters)
            candidates.append(ChainCA(sequence, np.array(coords), list(range(len(letters))), author, chain.name, sequence))
    if len(candidates) != 1:
        raise ValueError(f"Expected one protein chain in {path}, found {len(candidates)}")
    require_coordinates(candidates[0].coordinates)
    return candidates[0]


def kabsch(moving, fixed):
    require_coordinates(moving)
    require_coordinates(fixed)
    if moving.shape != fixed.shape:
        raise ValueError("Coordinate shapes differ")
    center_moving, center_fixed = moving.mean(axis=0), fixed.mean(axis=0)
    u, _, vt = np.linalg.svd((moving - center_moving).T @ (fixed - center_fixed))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    translation = center_fixed - center_moving @ rotation
    distances = np.linalg.norm(moving @ rotation + translation - fixed, axis=1)
    return rotation, translation, distances


def matched_indices(reference, prediction):
    # Use the full polymer alignment, then select only reference observed CAs.
    mapping = dict(sequence_pairs(reference.full_sequence, prediction.sequence))
    pairs = [(i, mapping[pos]) for i, pos in enumerate(reference.full_positions) if pos in mapping]
    if len(pairs) < 3:
        raise ValueError("Fewer than three sequence-corresponding observed CA pairs")
    return np.array(pairs, dtype=int)


def transformed_pdb(path, output_path, rotation, translation):
    structure = gemmi.read_structure(str(path))
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z]) @ rotation + translation
                    atom.pos = gemmi.Position(*xyz)
    structure.write_pdb(str(output_path))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(args):
    try:
        from tmtools import tm_align
    except ImportError as exc:
        raise RuntimeError("Install tmtools in the analysis environment: python -m pip install tmtools") from exc
    reference = reference_ca(args.reference, args.chain)
    if args.fasta:
        records = list(SeqIO.parse(str(args.fasta), "fasta"))
        if len(records) != 1:
            raise ValueError("Expected one target sequence in --fasta")
        target_sequence = str(records[0].seq).upper()
    else:
        target_sequence = reference.full_sequence
    paths = sorted(args.prediction_dir.glob("*.pdb"))
    if not paths:
        raise ValueError(f"No prediction PDBs in {args.prediction_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = args.output_dir / "aligned_predictions"
    aligned_dir.mkdir(exist_ok=True)
    metrics, residue_rows, predictions, pair_distance_errors = [], [], [], {}
    for path in paths:
        predicted = prediction_ca(path)
        predictions.append(predicted)
        if predicted.sequence != target_sequence:
            raise ValueError(f"Prediction {path.name} CA sequence does not exactly match the target sequence")
        pairs = matched_indices(reference, predicted)
        r, p = pairs.T
        rotation, translation, distances = kabsch(predicted.coordinates[p], reference.coordinates[r])
        result = tm_align(predicted.coordinates, reference.coordinates, predicted.sequence, reference.sequence)
        identity = np.mean([reference.sequence[i] == predicted.sequence[j] for i, j in pairs])
        ref_coords, pred_coords = reference.coordinates[r], predicted.coordinates[p]
        ref_distances = np.linalg.norm(ref_coords[:, None] - ref_coords[None, :], axis=-1)
        pred_distances = np.linalg.norm(pred_coords[:, None] - pred_coords[None, :], axis=-1)
        distance_difference = pred_distances - ref_distances
        pair_errors = distance_difference[np.triu_indices(len(pairs), k=1)]
        contacts = np.triu(ref_distances < 15.0, k=1)
        contact_errors = np.abs(distance_difference[contacts])
        if not len(contact_errors):
            raise ValueError("No reference CA contacts within 15 Angstrom for CA lDDT")
        ca_lddt = float(np.mean([np.mean(contact_errors < cutoff) for cutoff in (0.5, 1, 2, 4)]))
        pair_distance_errors[path.name] = (distance_difference, pair_errors)
        np.savez_compressed(args.output_dir / f"{path.stem}_pair_distance_errors.npz",
                            reference_full_positions=np.array([reference.full_positions[i] + 1 for i in r]),
                            predicted_minus_reference_angstrom=distance_difference)
        # Explicitly label this as a single-fit fraction; canonical GDT optimizes
        # separate transforms per threshold, which this reproducible statistic does not.
        gdt_single = np.mean([np.mean(distances <= cutoff) for cutoff in (1, 2, 4, 8)]) * 100
        full_fraction = len(pairs) / len(reference.full_sequence)
        d0 = max(0.5, 1.24 * np.cbrt(len(reference.full_sequence) - 15) - 1.8)
        row = {
            "prediction": path.name,
            "reference_full_length": len(reference.full_sequence),
            "reference_observed_ca": len(reference.coordinates),
            "prediction_ca": len(predicted.coordinates),
            "matched_ca": len(pairs),
            "matched_sequence_identity": float(identity),
            "coverage_full_reference": full_fraction,
            "coverage_observed_reference": len(pairs) / len(reference.coordinates),
            "ca_rmsd_sequence_fit_angstrom": float(np.sqrt(np.mean(distances**2))),
            "gdt_ts_single_fit_matched_percent": float(gdt_single),
            "gdt_ts_single_fit_full_percent": float(gdt_single * full_fraction),
            "sequence_tm_score_full_length_single_fit": float(np.sum(1 / (1 + (distances / d0)**2)) / len(reference.full_sequence)),
            "tm_align_reference_observed": float(result.tm_norm_chain2),
            "tm_align_prediction": float(result.tm_norm_chain1),
            "tm_align_ca_rmsd_angstrom": float(result.rmsd),
            "tm_align_aligned_pairs": sum(a != "-" and b != "-" for a, b in zip(result.seqxA, result.seqyA)),
            "ca_pair_distance_mae_angstrom": float(np.mean(np.abs(pair_errors))),
            "ca_pair_distance_rmse_angstrom": float(np.sqrt(np.mean(pair_errors**2))),
            "ca_pair_distance_absolute_p90_angstrom": float(np.quantile(np.abs(pair_errors), 0.9)),
            "ca_lddt_pair_weighted": ca_lddt,
            "ca_lddt_reference_contacts": int(len(contact_errors)),
        }
        metrics.append(row)
        for (i, j), distance in zip(pairs, distances):
            residue_rows.append({"prediction": path.name, "reference_full_position": reference.full_positions[i] + 1,
                                 "reference_author_residue": reference.author_residues[i], "prediction_ca_index": int(j) + 1,
                                 "reference_aa": reference.sequence[i], "prediction_aa": predicted.sequence[j],
                                 "ca_distance_angstrom": float(distance)})
        transformed_pdb(path, aligned_dir / path.name, rotation, translation)
    pairwise = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            _, _, distances = kabsch(predictions[i].coordinates, predictions[j].coordinates)
            pairwise.append({"prediction_1": paths[i].name, "prediction_2": paths[j].name,
                             "matched_ca": len(distances), "ca_rmsd_angstrom": float(np.sqrt(np.mean(distances**2)))})
    write_csv(args.output_dir / "metrics.csv", metrics)
    write_csv(args.output_dir / "per_residue_distances.csv", residue_rows)
    write_csv(args.output_dir / "pairwise_rmsd.csv", pairwise)
    best = max(metrics, key=lambda row: row["tm_align_reference_observed"])
    report = {
        "reference": str(args.reference.resolve()), "reference_sha256": digest(args.reference),
        "analysis_script_sha256": digest(Path(__file__)),
        "target_fasta": str(args.fasta.resolve()) if args.fasta else None,
        "target_fasta_sha256": digest(args.fasta) if args.fasta else None,
        "author_chain": args.chain, "reference_full_sequence": reference.full_sequence,
        "reference_observed_sequence": reference.sequence,
        "reference_unresolved_positions": sorted(set(range(1, len(reference.full_sequence) + 1)) - {i + 1 for i in reference.full_positions}),
        "target_sequence": target_sequence,
        "prediction_sha256": {path.name: digest(path) for path in paths},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "biopython", "gemmi", "tmtools", "matplotlib", "py3Dmol")},
        "definitions": {
            "tm_align_reference_observed": "Real TM-align; normalized by observed reference CA chain length, excluding unresolved residues. Structural alignment may differ from sequence correspondence.",
            "ca_rmsd_sequence_fit_angstrom": "Least-squares CA RMSD over sequence-corresponding observed residues only; full polymer alignment uses mmCIF label_seq_id, not author numbering.",
            "gdt_ts_single_fit": "Mean fractions within 1/2/4/8 Angstrom after the sequence-based RMSD fit. This is NOT canonical optimized GDT_TS. Full-percent version counts unresolved/unmatched reference residues as failures.",
            "sequence_tm_score_full_length_single_fit": "TM-score-shaped sequence statistic at one RMSD-fit transform, with full polymer length normalization/d0; not optimized TM-align.",
            "aligned_predictions": "All atoms transformed using sequence-correspondence CA least-squares fit; reference author chain coordinates preserved.",
            "pairwise_rmsd": "CA least-squares RMSD over the complete, identical target sequence for each prediction pair.",
            "best_prediction": "Highest observed-reference-normalized TM-align score; analysis selection, not model confidence.",
            "ca_pair_distance_errors": "All unordered CA-pair distances over sequence-corresponding resolved residues: prediction minus reference in Angstrom. Rigid-body invariant; can expose relative placement errors despite preserved local structure. Adjacent and distant pairs are both included.",
            "ca_lddt_pair_weighted": "Rigid-body-invariant CA-only lDDT variant over matched resolved residues: mean fractions of unordered reference CA pairs less than 15 Angstrom whose distance errors are below 0.5/1/2/4 Angstrom. Self-pairs excluded; pair-count weighted rather than residue averaged; not all-atom lDDT.",
        },
        "metrics": metrics, "pairwise_rmsd": pairwise, "best_prediction": best["prediction"],
    }
    report["ensemble_summary"] = {
        field: {"median": float(np.median([row[field] for row in metrics])),
                "min": float(min(row[field] for row in metrics)),
                "max": float(max(row[field] for row in metrics))}
        for field in ("tm_align_reference_observed", "ca_rmsd_sequence_fit_angstrom",
                      "ca_lddt_pair_weighted", "ca_pair_distance_mae_angstrom",
                      "gdt_ts_single_fit_matched_percent")
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    create_visuals(args, paths, metrics, residue_rows, best, pair_distance_errors)
    print(json.dumps({"samples": len(metrics), "best": best, "output_dir": str(args.output_dir)}, indent=2))
    return report


def create_visuals(args, paths, metrics, residue_rows, best, pair_distance_errors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import py3Dmol
    fig, ax = plt.subplots(figsize=(12, 5))
    for path in paths:
        rows = [row for row in residue_rows if row["prediction"] == path.name]
        ax.plot([row["reference_full_position"] for row in rows], [row["ca_distance_angstrom"] for row in rows], alpha=0.65, label=path.stem)
    ax.set(xlabel="Full reference polymer position (label_seq_id)", ylabel="CA distance after sequence fit (Angstrom)", title=f"Reference author chain {args.chain}; unresolved residues omitted")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "ca_distances.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    extent = max(1.0, max(float(np.max(np.abs(pair_distance_errors[path.name][1]))) for path in paths))
    for path in paths:
        ax.hist(pair_distance_errors[path.name][1], bins=np.linspace(-extent, extent, 81),
                histtype="step", density=True, alpha=0.75, label=path.stem)
    ax.set(xlabel="Predicted minus reference CA-pair distance (Angstrom)",
           ylabel="Density", title="Rigid-body-invariant internal distance errors")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "pair_distance_error_distribution.png", dpi=160)
    plt.close(fig)
    matrix = pair_distance_errors[best["prediction"]][0]
    limit = max(1.0, float(np.quantile(np.abs(matrix), 0.95)))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(xlabel="Matched resolved CA index", ylabel="Matched resolved CA index",
           title=f"{best['prediction']}: internal distance difference")
    fig.colorbar(image, ax=ax, label="Prediction minus reference (Angstrom)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "best_pair_distance_error_matrix.png", dpi=160)
    plt.close(fig)
    structure = MMCIFParser(QUIET=True).get_structure("reference", str(args.reference))
    class ReferenceChain(Select):
        def accept_model(self, model):
            return model.id == 0
        def accept_chain(self, chain):
            return chain.id == args.chain
    writer = PDBIO()
    writer.set_structure(structure)
    reference_pdb = args.output_dir / "reference_chain.pdb"
    writer.save(str(reference_pdb), select=ReferenceChain())
    view = py3Dmol.view(width=1000, height=650)
    view.addModel(reference_pdb.read_text(), "pdb")
    view.addModel((args.output_dir / "aligned_predictions" / best["prediction"]).read_text(), "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": "blue"}})
    view.setStyle({"model": 1}, {"cartoon": {"color": "orange"}})
    view.zoomTo()
    header = f"<p>Reference {args.chain}: blue. {best['prediction']}: orange. Sequence-correspondence CA least-squares superposition. Chosen by TM-align observed-reference score.</p>"
    (args.output_dir / "best_superposition.html").write_text(header + view.write_html())
    write_ensemble_viewer(args.output_dir, metrics)


def write_ensemble_viewer(output_dir, metrics, color="orange", label="Sample ensemble", backlink=None):
    """Render existing aligned structures, with optional sweep color and label."""
    best = max(metrics, key=lambda row: row["tm_align_reference_observed"])
    # Embed all aligned PDBs for interactive selection without structure downloads.
    payload = json.dumps({
        "reference": (output_dir / "reference_chain.pdb").read_text(), "best": best["prediction"],
        "predictions": {row["prediction"]: (output_dir / "aligned_predictions" / row["prediction"]).read_text() for row in metrics},
        "metrics": {row["prediction"]: row for row in metrics},
        "color": color, "label": label, "backlink": backlink,
    }).replace("<", "\\u003c")
    ensemble_html = '''<!doctype html>
<html><head><meta charset="utf-8"><title>SimpleFold ensemble vs experimental chain</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>body{font:15px system-ui;margin:20px}#viewer{width:100%;height:650px;position:relative}#metrics{margin:12px 0;line-height:1.7}select,button{font:inherit;padding:5px}.legend{display:flex;gap:24px;flex-wrap:wrap;margin:16px 0}.swatch{display:inline-block;width:18px;height:18px;border:1px solid #333;border-radius:3px;vertical-align:middle;margin-right:7px}</style></head>
<body><a id="back" hidden>← Tau sweep comparison</a><h2 id="title">SimpleFold ensemble vs experimental reference</h2>
<div class="legend"><span><i class="swatch" style="background:blue"></i>Experimental reference</span><span><i id="ensemble-swatch" class="swatch"></i><strong id="ensemble-label"></strong> · <span id="ensemble-count"></span>, solid</span></div>
<p>Drag to rotate; scroll to zoom.</p>
<label>Sample <select id="sample"><option value="all">All samples together</option></select></label> <button id="reset">Reset view</button>
<div id="metrics"></div><div id="viewer"></div>
<p>TM-align optimizes a structural alignment that can match a smaller subset with different residue correspondence.
The viewer and sequence-fit RMSD use all sequence-corresponding observed reference CAs.
CA lDDT is pair-weighted and measures local distances without superposition.
Unresolved reference residues have no coordinates. No confidence ranking is implied.</p>
<script>
const data = __PAYLOAD__;
document.getElementById('ensemble-swatch').style.backgroundColor=data.color;
document.getElementById('ensemble-label').textContent=data.label;
document.getElementById('ensemble-count').textContent=Object.keys(data.predictions).length+' samples';
if (data.label!=='Sample ensemble') {
  document.getElementById('title').textContent=data.label+' vs experimental reference';
  document.title=data.label+' — SimpleFold ensemble';
}
if (data.backlink) {
  const back=document.getElementById('back'); back.href=data.backlink; back.hidden=false;
}
const select = document.getElementById('sample');
for (const name of Object.keys(data.predictions)) {
  const option = document.createElement('option'); option.value = name; option.textContent = name;
  select.appendChild(option);
}
select.value = 'all';
const viewer = $3Dmol.createViewer('viewer', {backgroundColor:'white'});
function showSample() {
  const name=select.value;
  const names=name==='all' ? Object.keys(data.predictions) : [name];
  viewer.removeAllModels(); viewer.addModel(data.reference,'pdb');
  for (const [index, prediction] of names.entries()) {
    viewer.addModel(data.predictions[prediction],'pdb');
    viewer.setStyle({model:index+1},{cartoon:{color:data.color,opacity:1}});
  }
  viewer.setStyle({model:0},{cartoon:{color:'blue'}});
  viewer.zoomTo(); viewer.render();
  if (name==='all') {
    document.getElementById('metrics').textContent =
      `${names.length} samples over the experimental reference. Each is aligned independently using all sequence-corresponding resolved Cα atoms.`;
    return;
  }
  const m=data.metrics[name];
  document.getElementById('metrics').textContent =
    `TM-align (observed reference length ${m.reference_observed_ca}): ${m.tm_align_reference_observed.toFixed(4)}; `+
    `TM-align RMSD: ${m.tm_align_ca_rmsd_angstrom.toFixed(2)} Å on ${m.tm_align_aligned_pairs} structurally aligned pairs. `+
    `Sequence-fit CA RMSD: ${m.ca_rmsd_sequence_fit_angstrom.toFixed(2)} Å on ${m.matched_ca} pairs; `+
    `CA lDDT: ${m.ca_lddt_pair_weighted.toFixed(4)}; `+
    `full-sequence coverage: ${(100*m.coverage_full_reference).toFixed(2)}% (${m.matched_ca}/${m.reference_full_length}).`;
}
select.addEventListener('change',showSample);
document.getElementById('reset').addEventListener('click',()=>{viewer.zoomTo();viewer.render();});
showSample();
</script></body></html>'''.replace("__PAYLOAD__", payload)
    (output_dir / "ensemble_superposition.html").write_text(ensemble_html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--chain", default="A", help="Reference author chain ID")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fasta", type=Path)
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
