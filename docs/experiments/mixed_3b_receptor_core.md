# Cropped 3B mixed comparison

User authorized repeating the successful 3B mixed fine-tuning experiment with
previously approved receptor-core crops. Config: `mixed_train_3b_receptor_core`.
Only the six synthetic input structures/selection change from uncropped 3B
58333275. Reuse the cropped dataset from 100M run 58325408; no new cropping.

Released 3B initialization (SHA256 pinned), frozen ESM2, 600 updates with the
identical 540-PDB/60-complex seed-82301 replay, LR/warmup/EMA, activation
checkpointing and final-EMA-only export remain fixed. The general PDB data and
24-target CAMEO evaluation are unchanged. Sample three predictions per complex
per weight arm, with the original seeds, steps and tau.

Submit via `submit_mixed_pipeline.py training_config=mixed_train_3b_receptor_core
submit=true`: GPU training → GPU sampling → CPU OpenStructure scoring. Collect
with `check_training_run.py run_id=TRAIN_JOB completion_job_id=SCORE_JOB
wait_for_training=true`; cached DockQ and CPU MolProbity follow automatically.
All durable outputs/weights remain under `/pub/ynkim4/ml-simplefold/artifacts`.

Primary comparison: whether cropping reduces the systematic covalent-geometry
regression (bonds/angles) observed in uncropped 3B while preserving interface
improvement and general-protein folds. Compare held-out metrics directly with
uncropped 3B; complex metrics compare reconstruction of each run's own synthetic
training targets and must not be treated as equivalent cropped/full endpoints.
No new memory probe is needed: the larger uncropped inputs already completed
on A100 80GB. The training job still validates replay/dataset identity before fit.
