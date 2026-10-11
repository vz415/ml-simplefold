# 2H62 cooperative ACVR2B FK sweep

Date: 2026-10-11. Target: BMP2–BMPR1A–ACVR2B, completing experimental
2H62 from an observed 2:1:1 assembly to predicted 2:2:2.

## Hypothesis and objective

Previous FK samples earned W60 packing credit despite failed Y42 engagement
and incorrect ACVR2B docking. Test a cooperative hotspot score supported by
the complete observed ACVR2B contact pattern. This is a geometry objective,
not an affinity or binding-energy model.

For ACVR2B copy i:

```text
J_i = 2 W_i Y_i / (W_i + Y_i)
F_i = 2 C_i P_i / (C_i + P_i)
A_i = 0.5 (J_i + F_i)
R_ACVR2B = 3 min(A_1, A_2)
```

Zero denominators give zero. W/Y retain the existing bounded W60/Y42
distance scores; C/P are native-contact recall and precision using the
observed 2H62 ACVR2B interface. Contacts use matched reference-observed
heavy atoms within 4 Å. Missing predicted atoms retain the native denominator.
The second site exchanges both BMP2 identities and remains an inferred
contact restraint. Combine features within each copy before taking the
worse copy; retain the existing coherent eight-way chain assignment.

Keep the BMPR1A block H + C_BMPR1A + 0.5 P_BMPR1A and the original dimer,
observed-receptor placement, clash, and covalent-length penalties unchanged.
Standalone W60 and Y42 weights become zero. L61 geometry remains diagnostic
with zero penalty weight. Total positive maximum stays 5.5.

The harmonic mean treats W and Y symmetrically. The whole-contact block can
still earn up to 1.5 when joint hotspot engagement is zero. This is a soft
objective, not a hard requirement for all interactions or alternating topology.

## Implementation

- `scripts/fk_2h62_cooperative_reward.py`, `CooperativeComplexReward`.
- `scripts/sample_complex_fk.py`: reward factory dispatch.
- `configs/sampling/2h62_fk_cooperative.yaml`.
- `configs/experiment/2h62-inpaint-fk-cooperative.yaml`.
- `tests/test_2h62_cooperative_reward.py`.

Source/config changes are committed and pushed to the experiments fork before
remote submission; exact source commit and job IDs are recorded below.

## Authorized run design

- GPUs: **A100 for all five jobs**, no A30 fallback in this sweep.
- Tau: **0.01, 0.05, 0.1, 0.3, 0.8**.
- SimpleFold 3B, pretrained weights fixed, ten particles per job, 500 steps.
- Five checkpoint times: **[0.30, 0.60, 0.75, 0.90, 0.97]**.
- Beta 2, ESS threshold 0.8, common start, seeds 42/43/44.
- Full-pool model evaluation (`sampling.model_batch_size=null`).
- No new baseline sampling; the existing baseline is reused and rescored.
- A100 requests: one GPU, four CPUs, 96 GB host memory, 75-minute wall limit.

Submission template:

```bash
sbatch --parsable --job-name=2h62-coop-tauVALUE --gres=gpu:A100:1 \
  --mem=96G --time=01:15:00 scripts/hpc_sample.slurm \
  experiment=2h62-inpaint-fk-cooperative tau=VALUE
```

Launcher executes `python scripts/run_sampling.py` with those Hydra overrides.
Remote outputs: `/pub/ynkim4/ml-simplefold/artifacts/runs/JOB_ID/`.
Downloaded outputs: `artifacts/remote-runs/JOB_ID/`.
Machine-readable submission receipt:
`artifacts/sweeps/2h62_fk_cooperative/2026-10-11/manifest.json`.

## Evaluation and collection

Compare reference-frame receptor placement, native contacts, clashes, and
diversity independently of the reward. Scores across reward versions are not
directly comparable. Saved J/F/A values and per-copy contacts are reported
for diagnosing which block drives selection; L61 measurements stay available.

Download completed outputs/receipts, verify transfers, and update the persistent
complex page with `python scripts/update_complex_page.py --config
configs/viewers/2h62-inpaint.yaml`. Reuse cached coordinate analysis.

The user also requested a separate BMPR1A terminal-trimming control with the
previous score; that experiment is recorded separately and does not change
the full-length input to this cooperative sweep.
