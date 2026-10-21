# GPU training and sampling, followed by CPU evaluation

Mixed runs default to two Slurm jobs, with an `afterok` dependency:

| Stage | Resource | Work |
|---|---|---|
| `phase=train_sample` | A100 | Train, export EMA weights, then sample both comparison arms in the same allocation |
| `phase=score` | CPU | Score saved CAMEO predictions with OpenStructure; finalize the run |

Training and sampling share one GPU job ID and require only one A100 queue
wait. After saving training receipts, the process replaces itself with the
existing `phase=sample` entry point. That releases training state before the
inference-only model loads; no second `sbatch` is issued. Scientific settings
still come from the saved training config. The combined GPU time limit defaults
to three hours in the pipeline config.

Set `combine_gpu_stages=false` to retain separate `phase=train`, `phase=sample`,
and `phase=score` jobs. This remains supported for debugging. The already
submitted cropped 3B jobs 58352732/58352733/58352734 retain their original split.
Their remote checkout must stay at commit 342e0e1 until those jobs finish;
pushing new source does not authorize changing their pinned checkout.

The existing `check_training_run.py` then computes cached DockQ locally and
submits MolProbity to CPU Slurm. Sampling still needs the GPU. Structure scores,
clash counts, bond/angle validation and reports do not.

All phases use the approved `scripts/hpc_sample.slurm --train-mixed` launcher.
`phase=train` alone stops after exporting weights. The pipeline defaults to
`phase=train_sample`, followed by CPU scoring. Existing
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
partition/account/GPU/time/memory. Defaults: 3B uncropped, one A100 allocation for training and
sampling, CPU `free` scoring. For example, `training_config=mixed_train` selects
100M. Change scientific settings with
`'training_overrides=[max_steps=600,evaluation.tau=0.3]'`.
Those overrides are applied only to training; later stages read its saved config.
All stages pin the checkout commit and stop if it changes before execution.

The launcher records every successfully submitted ID immediately in
`logs/mixed-pipelines/TIMESTAMP/submission.json`. If submission fails halfway,
inspect that receipt before retrying; do not resubmit the whole pipeline blindly.
Record the submitted IDs/commands in the established Obsidian experiment directory.
No new experiment is authorized merely by this documentation.

## Outputs and analysis

One durable directory is keyed by the **GPU training/sampling** job:
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
