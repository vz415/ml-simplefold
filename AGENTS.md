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
  findings/resource-efficiency recaps; stop after BMP4 for review.
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
