#!/usr/bin/env python3
"""Aggregate existing structure comparisons across tau; no models or sampling."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

if __package__:
    from .compare_structures import digest, matched_indices, prediction_ca, reference_ca, write_csv
else:
    from compare_structures import digest, matched_indices, prediction_ca, reference_ca, write_csv


SUMMARY_FIELDS = (
    "tm_align_reference_observed", "ca_rmsd_sequence_fit_angstrom",
    "ca_lddt_pair_weighted", "ca_rg_matched_angstrom", "ca_max_span_matched_angstrom",
    "ca_long_range_pair_distance_mae_angstrom",
    "adjacent_ca_outside_3p6_4p1_fraction", "nonadjacent_ca_under_2p5_pairs",
    "adjacent_ca_distance_median_angstrom", "adjacent_ca_distance_min_angstrom",
    "adjacent_ca_distance_max_angstrom",
)


def resolve_path(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Metric distribution must be nonempty and finite")
    return {"median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}


def geometry(coordinates, full_positions):
    """CA-only geometry; neighbor tests preserve full polymer index gaps."""
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    positions = np.asarray(full_positions)
    separations = np.abs(positions[:, None] - positions[None, :])
    adjacent = distances[np.triu(separations == 1, k=1)]
    if not len(adjacent):
        raise ValueError("No sequence-adjacent CA pairs")
    return {
        "ca_rg_matched_angstrom": float(np.sqrt(np.mean(np.sum((coordinates - coordinates.mean(axis=0))**2, axis=1)))),
        "ca_max_span_matched_angstrom": float(distances.max()),
        "adjacent_ca_distance_median_angstrom": float(np.median(adjacent)),
        "adjacent_ca_distance_min_angstrom": float(adjacent.min()),
        "adjacent_ca_distance_max_angstrom": float(adjacent.max()),
        "adjacent_ca_outside_3p6_4p1_fraction": float(np.mean((adjacent < 3.6) | (adjacent > 4.1))),
        "nonadjacent_ca_under_2p5_pairs": int(np.count_nonzero(np.triu((separations > 1) & (distances < 2.5), k=1))),
    }, distances


def summarize(manifest_path, output_dir):
    manifest_path, output_dir = manifest_path.resolve(), output_dir.resolve()
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    reference_path = resolve_path(manifest["reference"], base)
    reference = reference_ca(reference_path, manifest.get("chain", "A"))
    reference_geometry, reference_distances = geometry(reference.coordinates, reference.full_positions)
    reference_hash = digest(reference_path)
    long_range = np.triu(np.abs(np.subtract.outer(reference.full_positions, reference.full_positions)) >= 20, k=1)
    if not long_range.any():
        raise ValueError("Reference has no observed CA pairs separated by at least 20 polymer positions")
    rows, summaries, groups, provenance = [], [], [], []
    runs = sorted(manifest["runs"], key=lambda run: float(run["tau"]))
    if not runs or any(float(run["tau"]) <= 0 for run in runs):
        raise ValueError("Need at least one run and positive tau values for the logarithmic plot")
    for run in runs:
        tau, job_id = float(run["tau"]), str(run["job_id"])
        analysis_dir = resolve_path(run["analysis_dir"], base)
        prediction_dir = resolve_path(run["prediction_dir"], base)
        comparison_path = analysis_dir / "comparison.json"
        comparison = json.loads(comparison_path.read_text())
        if comparison["reference_sha256"] != reference_hash or comparison["author_chain"] != reference.chain:
            raise ValueError(f"Run {job_id} uses a different reference or author chain")
        if comparison["reference_full_sequence"] != reference.full_sequence:
            raise ValueError(f"Run {job_id} full reference sequence differs")
        samples = comparison["metrics"]
        if len(samples) != 10 or len({row["prediction"] for row in samples}) != 10:
            raise ValueError(f"Run {job_id} needs exactly ten distinct sample metrics, found {len(samples)}")
        run_rows = []
        for metric in samples:
            prediction_path = prediction_dir / metric["prediction"]
            if digest(prediction_path) != comparison["prediction_sha256"][metric["prediction"]]:
                raise ValueError(f"Prediction hash mismatch: {prediction_path}")
            prediction = prediction_ca(prediction_path)
            if prediction.sequence != comparison["target_sequence"]:
                raise ValueError(f"Prediction CA sequence changed: {prediction_path}")
            pairs = matched_indices(reference, prediction)
            ref_indices, pred_indices = pairs.T
            if not np.array_equal(ref_indices, np.arange(len(reference.coordinates))):
                raise ValueError(f"Run {job_id} does not cover the same observed reference CA mask")
            shape, distances = geometry(prediction.coordinates[pred_indices], reference.full_positions)
            # Backbone indicators include all target residues, including unresolved prefix.
            backbone, _ = geometry(prediction.coordinates, list(range(len(prediction.coordinates))))
            for key in backbone:
                if key.startswith("adjacent_") or key.startswith("nonadjacent_"):
                    shape[key] = backbone[key]
            long_range_mae = float(np.mean(np.abs(distances - reference_distances)[long_range]))
            row = {"tau": tau, "job_id": job_id, **metric, **shape,
                   "ca_long_range_pair_distance_mae_angstrom": long_range_mae}
            rows.append(row)
            run_rows.append(row)
        pairwise = comparison["pairwise_rmsd"]
        if len(pairwise) != 45 or any(row["matched_ca"] != len(prediction.sequence) for row in pairwise):
            raise ValueError(f"Run {job_id} needs 45 full-target pairwise CA RMSDs")
        diversity_values = [row["ca_rmsd_angstrom"] for row in pairwise]
        stats = {field: distribution([row[field] for row in run_rows]) for field in SUMMARY_FIELDS}
        stats["ensemble_pairwise_ca_rmsd_angstrom"] = distribution(diversity_values)
        best = max(run_rows, key=lambda row: row["tm_align_reference_observed"])
        summary = {"tau": tau, "job_id": job_id, "samples": len(run_rows),
                   "best_prediction_by_tm": best["prediction"]}
        for field, values in stats.items():
            for statistic, value in values.items():
                summary[f"{field}_{statistic}"] = value
        summaries.append(summary)
        groups.append({"tau": tau, "job_id": job_id, "rows": run_rows,
                       "diversity": diversity_values, "analysis_dir": analysis_dir})
        provenance.append({"tau": tau, "job_id": job_id, "comparison_json_sha256": digest(comparison_path),
                           "comparison_versions": comparison["versions"]})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics.csv", rows)
    write_csv(output_dir / "summary.csv", summaries)
    report = {
        "target": manifest.get("target"), "manifest": manifest,
        "manifest_sha256": digest(manifest_path), "analysis_script_sha256": digest(Path(__file__)),
        "reference_sha256": reference_hash, "reference_author_chain": reference.chain,
        "reference_full_length": len(reference.full_sequence), "reference_observed_ca": len(reference.coordinates),
        "reference_geometry": reference_geometry, "provenance": provenance, "summary": summaries,
        "definitions": {
            "comparison_metrics": "TM-align normalizes to observed reference CA count. Sequence RMSD uses corresponding observed CAs; CA lDDT is pair-count weighted. See each run comparison.json for complete definitions.",
            "compactness": "CA radius of gyration and maximum pair span on the identical observed reference sequence mask for every sample; unobserved reference positions excluded.",
            "long_range_pair_mae": "Absolute CA pair-distance error over observed reference pairs separated by at least 20 full polymer positions; unordered pairs, rigid-body invariant.",
            "ensemble_diversity": "45 pairwise least-squares CA RMSDs on all full-target residues, including unresolved reference prefix. Diversity alone does not imply improved quality.",
            "backbone_indicators": "Prediction indicators use all target CAs: sequence-adjacent CA distances and fraction outside [3.6,4.1] Angstrom; nonadjacent CA pairs below 2.5 Angstrom exclude immediate sequence neighbors. Reference indicators use observed CAs and true full-position adjacency. These are coarse CA proxies, not chemical or all-atom validation.",
            "plot": "Every sample or pairwise RMSD is shown; medians connected. Tau uses logarithmic axis; horizontal jitter affects display only. Same seed across taus supports a controlled sweep, not independent replicates.",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_sweep(output_dir, groups, reference_geometry)
    write_index(output_dir, manifest, summaries, groups, reference_geometry, len(reference.coordinates))
    print(json.dumps({"output_dir": str(output_dir), "runs": len(groups), "samples": len(rows)}, indent=2))
    return report


def plot_sweep(output_dir, groups, reference_geometry):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    panels = [
        ("tm_align_reference_observed", "TM-align, observed-reference normalization"),
        ("ca_rmsd_sequence_fit_angstrom", "Sequence-fit CA RMSD (Å)"),
        ("ca_lddt_pair_weighted", "CA lDDT, pair weighted"),
        ("ca_rg_matched_angstrom", "Resolved-mask CA radius of gyration (Å)"),
        ("ca_max_span_matched_angstrom", "Resolved-mask maximum CA span (Å)"),
        ("ca_long_range_pair_distance_mae_angstrom", "Long-range CA pair-distance MAE (Å)"),
        ("ensemble_diversity", "Full-target pairwise CA RMSD (Å)"),
        ("adjacent_ca_outside_3p6_4p1_fraction", "Adjacent CA distances outside [3.6,4.1] Å"),
        ("nonadjacent_ca_under_2p5_pairs", "Nonadjacent CA pairs below 2.5 Å"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    rng = np.random.default_rng(0)
    taus = [group["tau"] for group in groups]
    for ax, (field, title) in zip(axes.flat, panels):
        medians = []
        for group in groups:
            values = group["diversity"] if field == "ensemble_diversity" else [row[field] for row in group["rows"]]
            x = group["tau"] * np.exp(rng.uniform(-0.06, 0.06, len(values)))
            ax.scatter(x, values, alpha=0.65, s=18)
            medians.append(np.median(values))
        ax.plot(taus, medians, color="black", marker="_", label="Median")
        if field in reference_geometry:
            ax.axhline(reference_geometry[field], color="tab:blue", linestyle="--", label="Experimental reference")
        ax.set_xscale("log")
        ax.set_xticks(taus, labels=[f"{tau:g}" for tau in taus])
        ax.xaxis.set_minor_locator(NullLocator())
        if field in ("adjacent_ca_outside_3p6_4p1_fraction", "nonadjacent_ca_under_2p5_pairs"):
            ax.set_ylim(bottom=0)
        ax.set(xlabel="Tau", title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle("7WF9-A tau sweep: ten samples per tau; geometry and diversity", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "tau_sweep.png", dpi=160)
    plt.close(fig)


def write_index(output_dir, manifest, summaries, groups, reference_geometry, observed_count):
    def interval(row, field, digits=3):
        return f"{row[field+'_median']:.{digits}f} [{row[field+'_min']:.{digits}f}, {row[field+'_max']:.{digits}f}]"
    body = []
    for row, group in zip(summaries, groups):
        relative = Path(os.path.relpath(group["analysis_dir"] / "ensemble_superposition.html", output_dir)).as_posix()
        link = quote(relative, safe="/")
        values = [f"{row['tau']:g}", html.escape(row["job_id"]),
                  interval(row, "tm_align_reference_observed"), interval(row, "ca_rmsd_sequence_fit_angstrom", 2),
                  interval(row, "ca_lddt_pair_weighted"), interval(row, "ca_rg_matched_angstrom", 2),
                  interval(row, "ensemble_pairwise_ca_rmsd_angstrom", 2),
                  interval(row, "adjacent_ca_outside_3p6_4p1_fraction"),
                  interval(row, "nonadjacent_ca_under_2p5_pairs", 0),
                  f'<a href="{html.escape(link, quote=True)}">All ten structures</a>']
        body.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    title = html.escape(str(manifest.get("target", "Tau sweep")))
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>{title} tau sweep</title>
<style>body{{font:15px system-ui;margin:24px}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ddd;padding:8px;text-align:left}}img{{max-width:100%}}.scroll{{overflow-x:auto}}</style></head>
<body><h1>{title} tau sweep</h1><p>Ten samples per tau. Entries show median [minimum, maximum].
Structure viewers use sequence-correspondence CA superposition; TM-align optimizes its own structural alignment.</p>
<p>Compactness uses the same {observed_count} observed reference CAs throughout.
Experimental CA radius of gyration: {reference_geometry['ca_rg_matched_angstrom']:.2f} Å;
maximum span: {reference_geometry['ca_max_span_matched_angstrom']:.2f} Å.
Ensemble diversity uses the full target sequence; increased diversity alone does not imply improved quality.
CA lDDT is pair weighted. Backbone indicators are coarse CA proxies, not chemical validation.</p>
<div class="scroll"><table><thead><tr><th>Tau</th><th>Job</th><th>TM-align</th><th>Sequence RMSD Å</th><th>CA lDDT</th><th>CA Rg Å</th><th>Diversity RMSD Å</th><th>Adjacent outlier fraction</th><th>Close nonadjacent CA pairs</th><th>Viewer</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>
<p><a href="summary.csv">Summary CSV</a> · <a href="metrics.csv">All sample metrics</a> · <a href="summary.json">Definitions and provenance</a></p>
<img src="tau_sweep.png" alt="Tau sweep metric distributions on a logarithmic tau axis">
</body></html>'''
    (output_dir / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
