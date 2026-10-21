# Separate training, sampling and CPU evaluation

Mixed runs now use three Slurm jobs, with `afterok` dependencies:

| Stage | Resource | Work |
|---|---|---|
| `phase=train` | A100 | Fit the folding model and export the final EMA weights |
| `phase=sample` | A100 | Load inference-only weights; generate matched original/tuned complex and CAMEO predictions |
| `phase=score` | CPU | Score saved CAMEO predictions with OpenStructure; finalize the run |

The existing `check_training_run.py` then computes cached DockQ locally and
submits MolProbity to CPU Slurm. Sampling still needs the GPU. Structure scores,
clash counts, bond/angle validation and reports do not.

All phases use the approved `scripts/hpc_sample.slurm --train-mixed` launcher.
**Behavior change:** `phase=train` stops after exporting weights; it no longer
samples or scores. Use the pipeline for the complete comparison. Existing
completed run files and reports are unaffected.

## Submit

After committing and pushing, verify direct laptop `ssh hpc3`, then use `hpc3y`.
Fast-forward the verified clean training checkout before submitting. Run on its
login node (the Python launcher only calls `sbatch`; it performs no compute):

```bash
cd /data/homezvol2/ynkim4/ml-simplefold-receptor-core-train
# Print commands without submitting:
/data/homezvol2/ynkim4/.conda/envs/simplefold/bin/python scripts/submit_mixed_pipeline.py

# For a separately authorized experiment:
/data/homezvol2/ynkim4/.conda/envs/simplefold/bin/python scripts/submit_mixed_pipeline.py submit=true
```

`configs/mixed_pipeline.yaml` selects the training config and each stage's
partition/account/GPU/time/memory. Defaults: 3B uncropped, A100 training and
sampling, CPU `free` scoring. For example, `training_config=mixed_train` selects
100M. Change scientific settings with
`'training_overrides=[max_steps=600,evaluation.tau=0.3]'`.
Those overrides are applied only to training; later stages read its saved config.
All stages pin the checkout commit and stop if it changes before execution.

The launcher records every successfully submitted ID immediately in
`logs/mixed-pipelines/TIMESTAMP/submission.json`. If submission fails halfway,
inspect that receipt before retrying; do not resubmit the whole pipeline blindly.
Record the three IDs/commands in the established Obsidian experiment directory.
No new experiment is authorized merely by this documentation.

## Outputs and analysis

One durable directory is keyed by the **training** job:
`/pub/ynkim4/ml-simplefold/artifacts/runs/TRAIN_JOB/`.
The trained weights remain in `checkpoints/trained/TRAIN_JOB/`.

- `resolved-config.yaml`, training summary, source/draw/dataset receipts remain
  the original training provenance.
- `training-completion.json`, `sampling-completion.json`, and
  `scoring-completion.json` record actual stage job IDs, source commits and times.
- Root `completion.json` advances through `trained` → `sampled` → `completed`.
- Original `pretrained/`, `fine_tuned/`, and `retention/` layouts remain compatible
  with the automated checker. No extra output/weight copies between jobs.
- Sampling verifies both checkpoints against the training summary, retains
  original sample seeds/noise/steps and constructs no EMA or optimizer state.
- CPU scoring verifies the saved scoring runtime and experimental-reference
  hashes again after the job boundary.

From the main local project directory, start collection/reporting after the CPU
job finishes, or let the checker wait for it:

```bash
python ../ml-simplefold-receptor-core-train/scripts/check_training_run.py \
  run_id=TRAIN_JOB completion_job_id=SCORE_JOB wait_for_training=true \
  geometry.wait_seconds=86400 'label=Uncropped mixed 3B'
```

The launcher prints the appropriate checker command. `run_id` always means the
training directory; `completion_job_id` tells the waiter to monitor CPU scoring
rather than collect as soon as GPU training finishes. Historical one-job runs
continue to work without this option.

CPU scoring can be retried without training or sampling. A partial sampling
receipt causes an explicit stop; inspect partial outputs before a manual retry.
This is deliberate duplicate protection, not automatic checkpointed sampling.

This refactor does not claim to eliminate the entire previous ~50-minute
overhead: checkpoint loading, ESM conditioning and file I/O still occur in GPU
stages. It moves CPU scoring outside the GPU reservation and records separate
stage wall times for future profiling.
