# Fixed CAMEO22 retention benchmark after multimer fine-tuning

User authorized 2026-10-20: benchmark the original and six-complex-fine-tuned
100M checkpoints on unrelated examples, using broad folding scores.

## Frozen target set

`configs/datasets/cameo22_retention_24_v1.json` fixes 24 targets, spanning
31–709 residues. The source is the original EigenFold CAMEO22 metadata at
commit `87911791549ae46c7d48b24108a5e0e9e758aa94`; its CSV hash is
`8a278a2dc7d2e6024c1d6412b662907894f813545e5eeabdb1ed2781d449a80d`.
The list is shared with the later AlphaFlow evaluation.

Start from all 183 entries. A heuristic local BLOSUM62 alignment screen excludes
sequences with >=30% identity (including gap columns) over >=70% of the shorter
chain against any of our six multimer targets. This is a practical similarity
screen, not a claim of exhaustive homology exclusion. Exclude obsolete 8AHP,
which SimpleFold's paper replaces with its successor 8QCW. Of 180 eligible
entries, select 24 evenly spaced ranks after sorting by full-sequence length
and target name, including endpoints. Selection uses no model performance.
The original public list contains short proteins under100 residues despite the
paper's approximate100–750 description.

Sequences use full `seqres`; references are the pinned chain-only full-atom
experimental PDBs from EigenFold. Missing residues and construct substitutions
remain explicit. The inference inputs are built from sequence alone, never
reference coordinates.

## Matched comparison

- Original released SimpleFold100M checkpoint.
- Fine-tuned EMA checkpoint from job58274048, trained on all six refined complexes.
- Both checkpoint hashes are pinned in `configs/cameo_benchmark.yaml`.
- One unguided prediction per target per checkpoint: **48 structures**.
- 500 steps, logarithmic times, tau .01, matching the paper's general-folding
  sampling setting; common target-specific seeds and conditioning for both arms.
- Fixed ESM2-3B; no Rosetta relaxation or pLDDT selection.

Use official OpenStructure **2.12.0** `compare-structures`, using the upstream
scoring family: TM-score, GDT-TS (`oligo_gdtts`), all-atom lDDT, backbone lDDT,
and RMSD. Backbone lDDT is not labeled Cα-only lDDT. Run a reference-vs-itself
control before inference. Report mean/median and paired per-target differences
with a target-level bootstrap interval. These subset scores must not be
presented as reproduction of the full published183-target means.

## Execution and storage

Preparation downloads only selected reference PDBs and a shared OpenStructure
SIF, on a Slurm CPU node. A dependent A100 job performs sampling and scoring.
`hpc_sample.slurm --benchmark-cameo` dispatches the existing approved launcher
to `scripts/run_cameo_benchmark.py` with Hydra config `cameo_benchmark.yaml`.
All compute and remote downloads run through Slurm. Source changes are isolated
in `ml-simplefold-cameo22-benchmark` / `experiments/cameo22-retention`.

Dataset: `/pub/ynkim4/ml-simplefold/artifacts/datasets/benchmarks/cameo22-retention-24-v1`.
Shared tool: `/pub/ynkim4/ml-simplefold/artifacts/tools/openstructure/openstructure-2.12.0.sif`.
Outputs: `/pub/ynkim4/ml-simplefold/artifacts/runs/JOB_ID`.
Local downloads: `artifacts/remote-runs/JOB_ID` in the primary laptop checkout.
Do not download or commit model weights, CCD caches, container images or
intermediate conditioning tensors.

Primary sources:
- https://arxiv.org/html/2509.18480v4#S9.SS4
- https://github.com/bjing2016/EigenFold/tree/87911791549ae46c7d48b24108a5e0e9e758aa94
- https://git.scicore.unibas.ch/schwede/openstructure/

## Scoring runtime availability

Initial preparation job58310975 failed because the paper's2.9.1 container tag
is no longer in the official registry. The registry currently publishes2.12.0.
Both arms use that pinned release, with a SIF digest and identity control. This
is a matched retention comparison, not exact reproduction of paper-table scores.
The A100 job58310979 is retained and its dependency updated to the preparation
retry; no duplicate GPU experiment is launched.

## Local report after collection

`build_cameo_benchmark_report.py` builds one persistent summary/example page
from downloaded results. It shows dataset means, per-target before/after metrics,
and a target selector with experimental/original/fine-tuned structure overlays.
Structure alignment is for display only and does not replace the standard scores.

```bash
python scripts/build_cameo_benchmark_report.py --run DOWNLOADED_RUN \
  --output artifacts/analysis/cameo22-retention-24-v1 \
  --library PATH_TO_LOCAL_3Dmol-min.js
```
