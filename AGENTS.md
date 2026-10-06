# SimpleFold experiments

- Read the global HPC access runbook in `~/.codex/docs/runbooks/hpc-access.md`.
- Use direct laptop SSH to `hpc3y` (account `ynkim4`). The checkout is
  `/data/homezvol2/ynkim4/ml-simplefold`; the conda prefix is
  `/data/homezvol2/ynkim4/.conda/envs/simplefold`.
- Do not sample locally. Local work is installation, imports, editing, and
  analysis of downloaded artifacts. Run remote installation, downloads, and
  inference through Slurm; login nodes are for Git, submission, and status.
- See `docs/hpc3y.md` for launch commands and environment details.
- Store remote model caches, results, environment locks, and artifact logs
  long-term under `/pub/ynkim4/ml-simplefold/artifacts`. Launchers default
  `SIMPLEFOLD_ARTIFACT_DIR` to that path; set it explicitly to override storage.
  The checkout and conda prefix stay in home storage. The checkout `artifacts`
  path is a symlink to the `/pub` artifact root for compatibility. Check free
  `/pub` space before downloading larger models or growing datasets.
- Keep local downloads under ignored `artifacts/` (including
  `artifacts/remote-runs/`) and logs under ignored `logs/`. Never commit model
  weights.
- Push experiment changes to the `experiments` remote (the user's fork).
  `origin` is the Apple upstream repository.
- The initial request authorizes environment setup and small toy sampling
  jobs. Larger experiment batches need a concrete user request.
