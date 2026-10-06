# SimpleFold experiments

- Read the global HPC access runbook in `~/.codex/docs/runbooks/hpc-access.md`.
- Use direct laptop SSH to `hpc3y` (account `ynkim4`). The checkout is
  `/pub/ynkim4/ml-simplefold`; the conda prefix is
  `/pub/ynkim4/conda/envs/simplefold`.
- Do not sample locally. Local work is installation, imports, editing, and
  analysis of downloaded artifacts. Run remote installation, downloads, and
  inference through Slurm; login nodes are for Git, submission, and status.
- See `docs/hpc3y.md` for launch commands and environment details.
- Keep model caches, results, environment locks, and logs under ignored
  `artifacts/` and `logs/` directories. Never commit model weights.
- Push experiment changes to the `experiments` remote (the user's fork).
  `origin` is the Apple upstream repository.
- The initial request authorizes environment setup and small toy sampling
  jobs. Larger experiment batches need a concrete user request.
