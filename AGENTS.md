# SimpleFold experiments

- Read the global HPC access runbook in `~/.codex/docs/runbooks/hpc-access.md`.
- Use direct laptop SSH to `hpc3y` (account `ynkim4`). The checkout is
  `/data/homezvol2/ynkim4/ml-simplefold`; the conda prefix is
  `/data/homezvol2/ynkim4/.conda/envs/simplefold`.
- Do not sample locally. Local work is installation, imports, editing, and
  analysis of downloaded artifacts. Run remote installation, downloads, and
  inference through Slurm; login nodes are for Git, submission, and status.
- See `docs/hpc3y.md` for launch commands and environment details.
- Store remote model weights, trained checkpoints, synthetic datasets, and run
  outputs under `/pub/ynkim4/ml-simplefold/artifacts`. Launchers default
  `SIMPLEFOLD_ARTIFACT_DIR` to that path; `SIMPLEFOLD_CACHE_DIR` defaults to the
  same root for checkpoint, Torch, and CCD caches. Keep pretrained checkpoints
  in `checkpoints/`, future trained checkpoints in `checkpoints/trained/`, and
  future synthetic datasets in `datasets/synthetic/`. Run outputs use `runs/`.
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
