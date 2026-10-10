# BMP2–BMPR1A–ACVR2A frozen-weight FK tau sweep

Requested 2026-10-10: move subsequent experiments to A100 and sweep FK
sampling tau using the previous 7WF9-A values: **0.01, 0.05, 0.1, 0.3, 0.8**.
The active chunked A30 pilot `57994634` remains running; it is not cancelled
or used as an exact paired baseline for these full-batch A100 runs.

## Design

Use the existing Hydra `experiment=2goo_fk` profile, overriding only global
`tau` for each job. Frozen SimpleFold-3B, 694-residue six-chain target,
500 log-grid steps, ten particles, common initial latent seed 42, Brownian
seed 43 and resampling seed 44. FK beta remains 2 and ESS threshold 0.8.
The adopted automated reward and checkpoints remain unchanged.

Each A100 job runs ten beta-zero baseline particles followed by ten FK
particles at the same tau. This gives 20 final structures per tau, 100 across
the five jobs, with matched initialization/noise settings within each pair.
Use full ten-particle model forwards (`sampling.model_batch_size=null`).
GPU chunking can change numerical results, so the A30 pilot is not assumed
to reproduce the A100 tau-0.01 run bitwise.

Submit five separate allocations so they can run concurrently when resources
allow. Hydra records the explicit tau override and resolved configuration
per job. A Hydra multirun inside one allocation would run conditions
sequentially; it is not used for this parallel batch.

Each job requests one A100, 96 GB host RAM and a 1h45 limit, using allocated
`gpu` / `eehui_lab_gpu` / normal QoS. Submission command for each TAU:

```bash
sbatch --parsable --gres=gpu:A100:1 --mem=96G --time=01:45:00 --job-name=2goo-3B-fk-tauTAU scripts/hpc_sample.slurm experiment=2goo_fk tau=TAU sampling.model_batch_size=null
```

## Tracking and analysis

Per-job outputs remain under `/pub/ynkim4/ml-simplefold/artifacts/runs/JOBID/`,
with `baseline/`, `fk/`, Hydra/provenance receipts, reward-event details,
selection histories, saved source states and final validation.
The sweep manifest belongs under
`/pub/ynkim4/ml-simplefold/artifacts/sweeps/bmp2_fk_tau/2026-10-10/manifest.json`;
the local copy belongs under ignored `artifacts/sweeps/bmp2_fk_tau/2026-10-10/`.
Local downloads/coordinate analysis do not load model weights or sample.

Compare baseline against FK at each tau using both weighted and unweighted
reward summaries, worse-copy receptor contact/hotspot recovery and precision,
BMP2 dimer/BMPR1A placement, ACVR2A orientation, heavy-atom clashes and
ensemble diversity. Retain every sample, ESS, number of resampling events and
lineage; ten terminal FK particles need not represent ten independent draws.
Do not select a tau solely by its best sample. This is one shared initial
latent and one noise-stream setting per tau, not replicated evidence across
independent starting seeds.

Submission receipts record the source commit and exact tau-to-job mapping.
No completed sweep or improvement is claimed by this experiment setup.

## Submitted jobs

Source: `f431a171a8f62f1cce5d7efccffc13f918fa0423`. All five were initially
pending for resources. Each job uses the command above with its named tau.

| Tau | A100 job | Expected baseline / FK structures |
| --- | --- | --- |
| 0.01 | 57994904 | 10 / 10 |
| 0.05 | 57994905 | 10 / 10 |
| 0.1 | 57994906 | 10 / 10 |
| 0.3 | 57994907 | 10 / 10 |
| 0.8 | 57994908 | 10 / 10 |

The same mapping and exact submission commands are saved in the local/remote
sweep manifest and per-job Obsidian receipts. No new model weights were
downloaded or loaded locally.
