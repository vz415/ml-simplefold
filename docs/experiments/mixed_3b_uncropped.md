# Uncropped SimpleFold 3B mixed fine-tuning

User authorized this run after reviewing cropped versus full-target 100M results.
Start from the released 3B weights, retain frozen ESM2, and reuse the identical
seed-82301 schedule: 600 updates, 540 experimental PDB and 60 synthetic complex
batches. Six uncropped refined targets, three predictions per target/model arm,
and the same 24-target CAMEO benchmark. Config: `mixed_train_3b.yaml`.

Launch the approved `hpc_sample.slurm --train-mixed` path in three stages:

1. CPU `phase=preflight`: verify the existing dataset/replay and hash the released
   3B checkpoint. Storage must pass before reserving an A100.
2. One A100 `phase=memory_probe`, after CPU success: eight optimizer updates,
   covering all six full complexes (largest first) plus two 512-residue general
   examples. Save timing/memory receipts, no model exports. This does not become
   the initialization for the actual run.
3. One A100 `phase=train`, dependent on probe success: restart the original 3B
   checkpoint and complete all 600 scheduled updates and both evaluations.

The probe and full run pin the checkpoint hash from the CPU receipt. Model size,
architecture and checkpoint paths propagate into the CAMEO benchmark. No 100M
weights or architecture may be used in the 3B comparison.

Memory adaptations: non-reentrant activation recomputation in the existing DiT
trunks, AdamW `foreach=false`, batch size one and bf16 AMP. Upstream loss, learning
rate, warmup, EMA decay and noise/settings remain fixed. A tiny CPU regression
test checks output/gradient equality (including dropout RNG and layer index) and
unchanged state-dict keys. The normal forward path remains the default for 100M.

Save only final EMA architecture weights for this 3B pilot (about 10.7 GiB); skip
repeated full optimizer snapshots and the raw export. Thus this pilot has no
optimizer-resume checkpoint. Before evaluating complexes, release optimizer state,
gradients and the unused GPU EMA copy. Parameterize prediction directories as
`predictions_simplefold_3B`. Record `step-resources.jsonl` and `resource-summary.json`
with timings, actual GPU memory peaks and device capacity.

Weights and durable outputs remain under `/pub/ynkim4/ml-simplefold/artifacts`.
Use home for Hydra/Slurm metadata. `/pub` metadata was timing out at launch setup;
the CPU preflight deliberately gates the paid GPU allocations. No extra dataset
or model download is planned.

After training, run `check_training_run.py` on the new job to collect, score,
validate and report, using the same automated pipeline as the 100M comparison.


## Workflow update after the first 3B result

The completed 58333275 run used the original combined allocation. Future mixed
runs use [three separate stages](../mixed_training_pipeline.md): `phase=train`
now exports weights and stops; `phase=sample` needs a GPU; `phase=score` runs
OpenStructure on CPU. This replaces step 3 above for subsequent experiments.
No new 3B training was submitted as part of the workflow change.
