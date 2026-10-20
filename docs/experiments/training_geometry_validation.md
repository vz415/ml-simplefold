# Geometry validation of fine-tuning outputs

Compare the original released100M, six-only fine-tuning (58274048), and
90%PDB/10%synthetic fine-tuning (58313695). User authorized geometry checks
2026-10-20. This analysis uses existing coordinates; it performs no sampling,
relaxation, training, or updates to source structures.

Inputs:18complex predictions/checkpoint (six targets, three matched seeds),
24CAMEO predictions/checkpoint, six Rosetta-refined synthetic training targets,
and24experimental CAMEO references:156structures total. Predictions are read
from pinned run receipts and verified by SHA256. The original arm is included
once; its identical copies in other benchmark runs are not additional samples.

`configs/training_geometry.yaml` records run IDs, runtime, sampling scope,
workers, and cache paths. Remote artifacts remain under
`/pub/ynkim4/ml-simplefold/artifacts`; the separate CPU environment stays in
`/data/homezvol2/ynkim4/.conda/envs/simplefold-geometry`.

## Measurements

- Preserve existing heavy-atom overlap counts (>0.4Å, excluding covalent
  1–2/1–3/1–4 neighbors). Add depth distributions, severe overlaps≥1Å,
  backbone/side-chain categories, distinct-residue percentages and chain pairs.
  This geometric measure is not MolProbity clashscore.
- cctbx-base2026.9 native Reduce2/Probe2 MolProbity clashscore: hydrogens are
  regenerated/optimized only on a temporary copy; heavy-atom coordinates are
  checked unchanged. Side-chain flips are disabled. B factors do not filter
  predictions, and no crystal mates or biological-assembly expansion are used.
- MolProbity Ramachandran and rotamer outliers, including evaluated counts.
- Bond-length, angle, chirality and planarity outliers using4σ; retain counts,
  denominators, percentages, deviations and atom/residue locations.
- Missing heavy atoms in observed residues, so incomplete inputs cannot
  silently acquire a perfect score. Entirely unresolved residues are not inferred.

The chemistry asset is pinned to cctbx2026.9 with SHA256
`4d5f67a7e2ae1c411e976d023e413aeff3ec310985aebf05ba3a093e5ca1acf1`.
See [cctbx installation](https://github.com/cctbx/cctbx_project/blob/v2026.9/README.md#monomer-library)
and [validation source](https://github.com/cctbx/cctbx_project/tree/v2026.9/mmtbx/validation).

## Run and inspect

Use the approved `scripts/hpc_sample.slurm` dispatcher on a CPU allocation:
`--geometry-setup` installs tools; `--validate-geometry` validates the frozen
comparison. `max_structures=1` is the runtime smoke check. Use full156inputs for
the comparison; checkpoint outputs are never regenerated. Configured caches
are keyed by source hashes, component mapping, backend and analysis code.

`scripts/build_training_geometry_report.py --run RUN --output OUTPUT` builds
the shared comparison page. Add `--geometry RUN/geometry-summary.json` to the
existing CAMEO report builder to include all three checkpoints' geometry tables.
Keep full diagnostics, CSV, summary, input hashes and resolved config together.

Report means and SD across structures. Paired confidence intervals resample
target means (six complexes or24CAMEO targets), preserving the dependence of
three samples from the same complex. Reference columns are contextual:
synthetic targets were relaxed, generated samples were not, and experimental
references can omit unresolved residues. Six-only versus mixed remains
600versus60complex updates; this is not an equal-exposure comparison.
