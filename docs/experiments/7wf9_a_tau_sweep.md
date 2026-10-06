# 7WF9-A tau sweep

## Status: waiting for model storage

The user approved ten samples each at tau **0.01, 0.05, 0.1, 0.3, and 0.8** (50 total), with SimpleFold-100M, ESM2-3B, 500 logarithmic sampling steps, seed 42, and pLDDT off. Sampling has not yet succeeded for this sweep.

The first attempt, job **57840798**, failed with exit **1:0** after **24 seconds** on `hpc3-gpu-16-00` on 2026-10-06 at approximately 11:45 PDT. The model-cache `stat` guard timed out and printed `Model storage is unavailable; stopping before inference.` No model was loaded and no protein samples were generated. The remaining four jobs were not submitted because the same public storage is required.

Direct SSH, Git, and Slurm access work. A bounded login-node read of `/pub/ynkim4/ml-simplefold/artifacts/checkpoints/simplefold_100M.ckpt` timed out; a later bounded check at 11:47:54 PDT also returned no file metadata. An earlier SFTP viewer upload had stalled as well. This establishes a storage-access problem, not its cause. No public RCIC notice confirming an October 6 outage was found; [RCIC news](https://rcic.uci.edu/about/news.html) says maintenance announcements are distributed by email. `/pub` is [BeeGFS DFS storage](https://rcic.uci.edu/storage/dfs.html). Preserve existing models and results and avoid repeated metadata probes or GPU retries while storage remains unavailable.

## Ready configuration and commands

Sampling launcher source commit: `63ec53b3d0949119c7cdde5a55c6fa70bff96e91`, pushed and synced to HPC3y. The existing approved `hpc_sample.slurm` now accepts arguments FASTA, steps, samples, tau, seed; its defaults remain unchanged. It checks model-storage access with a bounded timeout and records exact settings in per-run `experiment.json`.

After storage access recovers, submit the approved launcher through the experiment helper, one job per tau:

```bash
for tau in 0.01 0.05 0.1 0.3 0.8; do
  ~/.agents/skills/hpc-experiments/scripts/submit.sh \
    hpc_sample.slurm examples/7wf9_a.fasta 500 10 "$tau" 42
done
```

Each job writes to `/pub/ynkim4/ml-simplefold/artifacts/runs/JOB_ID/`. Keep sampling entirely on Slurm GPU nodes. Keep weights and long-term outputs under `/pub`; do not redownload large model caches or sample locally to work around the storage failure.

Local sweep manifest and deliberately untracked downloaded/analysis outputs use `artifacts/sweeps/7wf9_a_tau/2026-10-06/`. The manifest preserves the failed attempt separately and currently has an empty successful `runs` list. For each completed job, add a run with `tau`, `job_id`, `analysis_dir` (`runs/JOB_ID/comparison`), and `prediction_dir` (`runs/JOB_ID/predictions_simplefold_100M`). Paths can be relative to the manifest directory. Transfer specific PDB/provenance files through `access-hpc3.rcic.uci.edu`.

Run `scripts/compare_structures.py` on each downloaded ten-sample ensemble using `artifacts/references/7WF9/7WF9.cif`, author chain A, and `examples/7wf9_a.fasta`. Then aggregate:

```bash
python scripts/summarize_tau_sweep.py \
  --manifest artifacts/sweeps/7wf9_a_tau/2026-10-06/manifest.json \
  --output-dir artifacts/sweeps/7wf9_a_tau/2026-10-06
```

The aggregator produces all-sample/summary CSVs, JSON provenance, a nine-panel distribution plot, and an HTML index linking each solid-orange ensemble over the blue experimental chain. Its verification against the previously completed baseline is a pipeline check, **not** a new tau-sweep result.

## Comparison design

Use the full 126-residue target sequence and compare the same 123 resolved chain-A CAs at polymer positions 4–126. Keep the sample count, input, model, software, and seed fixed across tau, which supports paired random-noise comparisons by sample index; this is one seed and one target, not independent biological replication.

Report medians and ranges across every sample for observed-reference TM-align, sequence-correspondent CA RMSD, pair-weighted CA lDDT, resolved-mask CA radius of gyration/span, and long-range CA-pair distance error (polymer separation at least 20). Report all 45 full-target pairwise RMSDs per tau for ensemble diversity. Higher tau may increase noise and geometric distortions; increased diversity alone is not improved folding. Include coarse adjacent CA distance/outlier and nonadjacent close-CA indicators, explicitly distinct from chemical/all-atom validation. Keep TM-align structural-subset RMSD distinct from whole observed-chain sequence RMSD.

The previous ten-sample tau-0.01 run, job57838536, is documented separately in [7wf9_a.md](7wf9_a.md). It is available for pipeline verification; rerun the baseline after storage recovers to match sweep provenance.

If the storage failure persists, a [RCIC storage ticket](https://rcic.uci.edu/help/tickets.html#storage-problems) should include account `ynkim4`, exact `/pub` model path, the timestamped stat/SFTP failures, and affected node/job57840798. No support message has been sent.
