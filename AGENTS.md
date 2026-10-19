# SimpleFold experiments

- Before BMP90 work, read [the BMP90 workflow](docs/experiments/bmp90_workflow.md).
  It links the scoring, interface-evidence, and failure registries and the evaluation
  protocol, and defines target priority, transfer, analysis, and recording rules.
  Follow `configs/bmp90/evaluation_protocol.json` and
  `configs/bmp90/reference_gap_audit.json` throughout reconstruction; these are
  standing project guidance, not receipt-only handoff notes.
  Its reference-gap subsection links the dated audit and requires extra scrutiny
  for uncovered interfaces; no reference found is not proof of absence.
  It does not authorize execution; follow current user instructions for submissions.
- Rank BMP90 recipes by the best overall evaluated structures obtained across
  tau 0.1, 0.3, and 0.8, recording the winning settings. A weaker result at one
  tau does not disqualify a recipe or require a separate investigation. Retain
  all outcomes and robustness context; prioritize remaining structural defects.
  See the workflow's guidance on selecting recipes across noise levels.
- Read the global HPC access runbook in `~/.codex/docs/runbooks/hpc-access.md`.
- Use direct laptop SSH to `hpc3y` (account `ynkim4`). The checkout is
  `/data/homezvol2/ynkim4/ml-simplefold`; the conda prefix is
  `/data/homezvol2/ynkim4/.conda/envs/simplefold`.
- Do not sample locally. Local work is installation, imports, editing, and
  analysis of downloaded artifacts. Run remote installation, downloads, and
  inference through Slurm; login nodes are for Git, submission, and status.
- See `docs/hpc3y.md` for launch commands and environment details.
- A100 is the default sampling GPU. For an authorized A30 fallback on the
  ten-particle FK complex, override `--gres=gpu:A30:1` and use
  `sampling.model_batch_size=2`; retain the full ten-particle selection pool.
  Keep fallback submissions explicit rather than launching duplicate jobs.
- Vincent authorized paid A30 BMP90 batches on `gpu` / `eehui_lab_gpu` and
  proactive informative experiments within the five-target milestone. Overlap
  validated transfer hypotheses instead of waiting for every predecessor tau
  result. Keep one submission owner, bounded batch receipts, and periodic
  findings/resource-efficiency recaps. Following the BMP4 review, Vincent
  approved the 2026-10-16 expansion to BMP4–BMPR1A–ACVR2A,
  GDF5–BMPR1A–ACVR2A, and GDF5–BMPR1A–ACVR2B: nine paired sampling jobs
  across tau 0.1/0.3/0.8, followed by restrained relaxation of the three
  selected BMP7 candidates. Submit sampling first, then cleanup; keep the
  type-I GDF5 mutant-reference qualification explicit.
- On 2026-10-17 Vincent requested stronger per-copy type-II constraints for
  BMP4–BMPR1A–ACVR2A and GDF5–BMPR1A–ACVR2B, plus relaxation of the recent
  candidates. Wave18 is bounded to six paired sampling jobs at tau 0.1/0.3/0.8
  (three A30, three A100; at most 18 requested GPU-hours), followed by four
  CPU relaxations: BMP4 job58197150 FK3, GDF5–ACVR2A job58197152 baseline8
  and FK7, and GDF5–ACVR2B job58197156 FK0. Preserve the unguided origin of
  baseline8, which is identical to FK8. This does not authorize new targets.
- Vincent approved wave19 on 2026-10-17: nine A30 paired sampling jobs across
  tau 0.1/0.3/0.8 for BMP4–BMPR1A–ACVR2A type-II orientation,
  BMP6–ACVR1–ACVR2B joint N60/wrist requirements, and
  BMP7–ACVR1–ACVR2A required Q89 face contacts. Cap this batch at 27 requested
  GPU-hours; preserve the queued GDF5 A100 jobs 58209925/26/27. No additional
  targets or relaxation jobs are part of this batch. Keep previous best recipes
  until new outcomes support a ranking change.
- On 2026-10-18 Vincent requested another BMP6 ACVR1 scoring iteration and
  further BMP4/BMP7 iterations, submitted before GDF5 relaxation. Wave20 is
  nine A30 paired sampling jobs across tau 0.1/0.3/0.8, capped at 27 requested
  GPU-hours, followed by two restrained CPU cleanups of GDF5–BMPR1A–ACVR2B
  job58209925 FK2 and baseline7. Preserve baseline7's unguided origin (equal
  to FK7), mutant-reference qualifications, and the distinction between local
  packing cleanup and validating inferred type-II placement. No new targets.
- After the wave20 review, Vincent requested the next experiment batch on
  2026-10-18 Pacific (2026-10-19 UTC). Wave21 is nine A30 paired sampling jobs
  across tau 0.1/0.3/0.8, capped at 27 requested GPU-hours, plus nine dependent
  CPU evaluations. Revise BMP4 second-copy type-II placement, BMP6 N60/wrist
  geometry, and BMP7 joint T83/Q89/type-II placement using retained successes
  and recorded failures. No new targets, relaxation jobs, or automatic retries.
  Preserve previous best recipes until the new outcomes support a change.
- Store remote model weights, trained checkpoints, synthetic datasets, and
  canonical run outputs under `/pub/ynkim4/ml-simplefold/artifacts`. Launchers default
  `SIMPLEFOLD_ARTIFACT_DIR` to that path; `SIMPLEFOLD_CACHE_DIR` defaults to the
  same root for checkpoint, Torch, and CCD caches. Keep pretrained checkpoints
  in `checkpoints/`, future trained checkpoints in `checkpoints/trained/`, and
  future synthetic datasets in `datasets/synthetic/`. Run outputs use `runs/`.
- Vincent requested temporary result copies in the experiment checkout's home
  storage on 2026-10-14. New sampling launches write to
  `artifacts/transfer-stage/runs/JOB_ID/`, then mirror to canonical `/pub` storage.
  New campaign evaluators must read/write this home run first, falling back to
  canonical storage for historical runs. Keep it unsealed until all evaluation
  writers finish. Retrieve with `scripts/retrieve_results.py`, passing every
  evaluation job ID; it verifies SHA-256 on the download and canonical copy
  before removing only that temporary run. Keep the cache on failure. Do not
  cache model weights or delete canonical results. See `docs/hpc3y.md`.
  Preserve frozen queued campaigns; cache their completed results separately.
- Keep installation/pip caches and environment locks in the home checkout's
  `artifacts/pip-cache/` and `artifacts/environment/`. The checkout and conda
  prefix stay in home storage. Use explicit paths without artifact symlinks.
- Keep local downloads under ignored `artifacts/` (including
  `artifacts/remote-runs/`) and logs under ignored `logs/`. Never commit model
  weights.
- Push experiment changes to the `experiments` remote (the user's fork).
  `origin` is the Apple upstream repository.
- The initial request authorizes environment setup and small toy sampling
  jobs. Larger experiment batches need a concrete user request.
