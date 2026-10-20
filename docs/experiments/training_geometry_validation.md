# Geometry validation of fine-tuning outputs

Compare the original released 100M, six-only fine-tuning (58274048), and
90% PDB / 10% synthetic fine-tuning (58313695). User authorized geometry checks
2026-10-20. This analysis uses existing coordinates; it performs no sampling,
relaxation, training, or updates to source structures.

Inputs: 18 complex predictions/checkpoint (six targets, three matched seeds),
24 CAMEO predictions/checkpoint, six Rosetta-refined synthetic training targets,
and 24 experimental CAMEO references: 156 structures total. Predictions are read
from pinned run receipts and verified by SHA256. The original arm is included
once; its identical copies in other benchmark runs are not additional samples.

`configs/training_geometry.yaml` records run IDs, runtime, sampling scope,
workers, and cache paths. Remote artifacts remain under
`/pub/ynkim4/ml-simplefold/artifacts`; the separate CPU environment stays in
`/data/homezvol2/ynkim4/.conda/envs/simplefold-geometry`.

## Measurements

- Preserve existing heavy-atom overlap counts (>0.4Å, excluding covalent
  1–2/1–3/1–4 neighbors). Add depth distributions, severe overlaps ≥1 Å,
  backbone/side-chain categories, distinct-residue percentages and chain pairs.
  This geometric measure is not MolProbity clashscore.
- cctbx-base 2026.9 native Reduce2/Probe2 MolProbity clashscore: hydrogens are
  regenerated/optimized only on a temporary copy; heavy-atom coordinates are
  checked unchanged. Side-chain flips are disabled. B factors do not filter
  predictions, and no crystal mates or biological-assembly expansion are used.
- MolProbity Ramachandran and rotamer outliers, including evaluated counts.
- Bond-length, angle, chirality and planarity outliers using 4σ; retain counts,
  denominators, percentages, deviations and atom/residue locations.
- Missing heavy atoms in observed residues, so incomplete inputs cannot
  silently acquire a perfect score. Entirely unresolved residues are not inferred.
- Independent sequence-consecutive peptide C–N lengths catch broken links even
  when cctbx distance-based topology omits them. Diagnostic extreme-length
  bounds are 1.1–1.6 Å; links >2 Å are counted separately. TER/numbering gaps are
  respected. Native cctbx covalent counts do not represent all intended
  disulfides or peptide links; retain the recorded topology limitations.

The chemistry asset is pinned to cctbx 2026.9 with SHA256
`4d5f67a7e2ae1c411e976d023e413aeff3ec310985aebf05ba3a093e5ca1acf1`.
See [cctbx installation](https://github.com/cctbx/cctbx_project/blob/v2026.9/README.md#monomer-library)
and [validation source](https://github.com/cctbx/cctbx_project/tree/v2026.9/mmtbx/validation).

## Run and inspect

Use the approved `scripts/hpc_sample.slurm` dispatcher on a CPU allocation:
`--geometry-setup` installs tools; `--validate-geometry` validates the frozen
comparison. `max_structures=1` is the runtime smoke check. Use full 156 inputs for
the comparison; checkpoint outputs are never regenerated. Configured caches
are keyed by source hashes, component mapping, backend and analysis code.

`scripts/build_training_geometry_report.py --run RUN --output OUTPUT` builds
the shared comparison page. Add `--geometry RUN/geometry-summary.json` to the
existing CAMEO report builder to include all three checkpoints' geometry tables.
Keep full diagnostics, CSV, summary, input hashes and resolved config together.

Report means and SD across structures. Paired confidence intervals resample
target means (six complexes or 24 CAMEO targets), preserving the dependence of
three samples from the same complex. Reference columns are contextual:
synthetic targets were relaxed, generated samples were not, and experimental
references can omit unresolved residues. Six-only versus mixed remains
600 versus 60 complex updates; this is not an equal-exposure comparison.


## Protein-only interpretation and shared-storage staging

Disable incidental automatic crosslinks in initial cctbx model interpretation,
Reduce2 and Probe2 (`automatic_linking.link_none=True`). Otherwise severe
nonbonded overlaps can be mistaken for new covalent bonds, suppressing hydrogen
placement and undercounting clashes. Native peptide and disulfide handling stay
active. Symmetric atom renaming is also disabled at every interpretation stage;
heavy-atom identities and coordinates must survive hydrogenation unchanged.
These settings apply to the prepared protein-only inputs, not arbitrary
cofactor-containing or chemically crosslinked structures.

The chemistry package also needs `mmtbx.rebuild_rotarama_cache`; setup builds
and actually loads both lookup tables before declaring the environment ready.

If `/pub` is stalled, provide `input_manifest=MANIFEST` and
`staging_root=TEMPORARY_HOME_DIRECTORY`, plus a temporary `cache_dir`. The
manifest points to staged copies with the same pinned SHA256 values; preserve
the downloaded source paths in its provenance fields. The runner writes all
analysis outputs to staging, then publishes them to the configured `/pub`
`output_dir`. The staging directory survives a blocked publish operation, so
completed analysis remains retrievable. This is a temporary workaround; durable
run outputs and model artifacts remain under `/pub`.

Hydra's own logs and `.hydra` metadata stay under the home checkout's ignored
`logs/training-geometry/JOB_ID`, so Hydra does not touch shared storage before
the staging workaround can take effect. The resolved scientific configuration
is also saved with the durable analysis outputs.


## Completed geometry comparison — 2026-10-20

Full run **58320484** completed successfully in **7m50s** (eight allocated CPU
cores; 55m22s aggregate CPU; peak step RSS 12.5 GiB). Source: `dda1c2c`;
scientific analysis cache key:
`f60565ab95d96c8295824e3eab8e7d03f53f0a5cdcea7ccfeb31c2181a25e42c`.
The final run reused completed checks after moving all Hydra metadata to home
logs. Earlier queued/stalled batches were cancelled; no new sampling or training.

All 156 structures passed. The 54 complex predictions' previous overlap
counts and every coordinate-only metric match exactly. Input hashes match;
heavy atoms are unchanged (maximum hydrogenation displacement 1.72e-13 Å).
Incidental crosslinks are disabled in every structure, with no `Misc. bond`
hydrogen omissions. Every native clashscore normalization was verified.

Downloaded and verified via RCIC: `artifacts/remote-runs/58320484/`.
Compressed output archive SHA256:
`fa695da7ebeb54c76b69cee55fe6668f353084f7f1cfc75c03b436d5fc756242`.
Remote durable outputs: `/pub/ynkim4/ml-simplefold/artifacts/runs/58320484/`.
Setup receipts **58319626**, cache preparation **58319991**, and corrected smoke
**58320294** were also downloaded and verified.

### Means across the six training complexes

18 predictions per checkpoint (three samples per target). The reference is our
synthetic, Rosetta-refined training target, not an independent experimental
complex or a held-out generalization test.

| Metric | Original | Six-only | Mixed 90/10 |
|---|---:|---:|---:|
| MolProbity clashscore ↓ | 501.60 | 130.62 | 322.38 |
| Heavy-atom overlaps ↓ | 3296.22 | 839.50 | 2123.56 |
| Backbone–backbone overlaps ↓ | 647.28 | 128.33 | 442.44 |
| Residues clashing (%) ↓ | 84.12 | 53.24 | 74.63 |
| Bond-length outliers (%) ↓ | 27.69 | 19.24 | 32.10 |
| Bond-angle outliers (%) ↓ | 8.88 | 7.75 | 10.78 |
| Ramachandran outliers (%) ↓ | 1.08 | 0.43 | 0.84 |
| Rotamer outliers (%) ↓ | 1.11 | 0.28 | 0.86 |
| Maximum overlap (Å) ↓ | 3.05 | 2.94 | 3.06 |

### Means across 24 CAMEO targets

One matched-seed prediction per target/checkpoint; this fixed subset does not
reproduce the full paper benchmark.

| Metric | Original | Six-only | Mixed 90/10 |
|---|---:|---:|---:|
| MolProbity clashscore ↓ | 70.84 | 67.20 | 65.18 |
| Bond-length outliers (%) ↓ | 2.73 | 5.61 | 9.35 |
| Bond-angle outliers (%) ↓ | 0.51 | 1.65 | 1.94 |
| Ramachandran outliers (%) ↓ | 0.06 | 0.22 | 0.09 |
| Rotamer outliers (%) ↓ | 0.21 | 0.08 | 0.27 |
| Maximum overlap (Å) ↓ | 1.91 | 1.87 | 1.76 |

### Interpretation and independent audit

Six-only tuning substantially improves packing and local geometry on its six
training complexes, but substantial problems remain: refined training targets
have mean clashscore **2.04** and bond-length outliers **0.145%**, versus
**130.62** and **19.24%** in the six-only predictions. The worst remaining overlap
barely changes (3.05 → 2.94 Å), despite far fewer overlaps.

Mixed tuning reduces complex clashes, but worsens complex bond/angle geometry.
On CAMEO, both tuned models worsen bond and angle geometry despite slightly
lower mean clashscores. Bond outliers worsen on **23/24** targets after six-only
tuning and **24/24** after mixed tuning; angle outliers worsen on every target
in both arms. The apparent preservation of global fold scores does not imply
preservation of local covalent geometry.

The target-paired bootstrap 95% intervals for CAMEO bond-outlier increases are
**+2.88 percentage points [2.01, 3.52]** for six-only and
**+6.62 points [6.15, 7.15]** for mixed. CAMEO clashscore-change intervals include
zero for both arms, so do not claim a confirmed general packing improvement.
These intervals describe this small target set; metrics are correlated.

Independent coordinate audit on matched target **7pl4_A** confirmed all bond
outlier lengths and Z scores directly against the local input PDBs, with equal
233-bond denominators: original **6/233**, six-only **9/233**, mixed **28/233**,
experimental reference **0/233**. The deterioration is not a chain mapping,
restraint denominator, or coordinate transformation artifact.

Interpretation remains exposure-confounded: both training runs had 600 updates,
but six-only saw 600 synthetic batches and mixed saw 60. No new training or
refinement was launched from this geometry result.

### Reports and validation

- Canonical geometry report: `artifacts/analysis/training-geometry/index.html`
  (`http://127.0.0.1:8767/training-geometry/index.html`). Includes all three
  checkpoints, both reference cohorts, per-target/sample selection, CSV,
  detailed residue locations, and paired confidence intervals.
- Existing CAMEO pages now include the same three-checkpoint geometry tables:
  `artifacts/analysis/cameo22-retention-24-v1/index.html` and
  `artifacts/analysis/mixed-training/58313695/cameo22-retention/index.html`.
- Browser QA passed: six complex selectors, 24 CAMEO selectors, complete metrics,
  and both existing 3D viewers/toggles. No JavaScript errors.
- Raw-output and coordinate-preservation audit:
  `artifacts/remote-runs/58320484/analysis-verification.json`.
- All numerical results use the audited protein-only linking settings, not the
  superseded early smoke results.
