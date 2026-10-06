# 7WF9-A tau sweep

## Completed: 50 samples

Ten samples each at tau **0.01, 0.05, 0.1, 0.3, and 0.8** completed on HPC3y on 2026-10-06. Higher tau increased ensemble diversity about fourfold but did not correct the compact fold: whole observed-chain sequence RMSD stayed near 20 Å, and median TM-score and CA lDDT declined. The small RMSD decrease does not establish improved folding.

All conditions used SimpleFold-100M, ESM2-3B, 500 logarithmic sampling steps, seed 42, pLDDT off, and source commit `85c58c0f551e6ed334570d7a390b445ac6ac38bb`. Each job generated ten draws from one RNG stream. All 50 PDBs passed exact 126-residue sequence, atom-count, and finite-coordinate checks. No sampling or model loading ran locally.

| Tau | Job | GPU | Sampling | Slurm elapsed | Result |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 57841734 | V100 16 GB | 29 s | 91 s | COMPLETED 0:0 |
| 0.05 | 57841774 | A30 | 26 s | 68 s | COMPLETED 0:0 |
| 0.1 | 57841775 | A30 | 26 s | 66 s | COMPLETED 0:0 |
| 0.3 | 57841777 | V100 16 GB | 29 s | 86 s | COMPLETED 0:0 |
| 0.8 | 57841778 | A30 | 27 s | 70 s | COMPLETED 0:0 |

Jobs ran between 12:01 and 12:04 PDT. Sampling time excludes startup/model loading; Slurm elapsed includes those costs. GPU families differ across conditions, so runtime is descriptive rather than a controlled performance comparison.

## Results

Values below are medians across all ten predictions, except diversity, which is the median of all 45 full-target pairwise CA RMSDs. Full ranges and every sample are retained in `summary.csv` and `metrics.csv`.

| Tau | TM-score | Sequence CA RMSD (Å) | CA lDDT | Resolved CA Rg (Å) | Long-range distance MAE (Å) | Diversity (Å) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 0.544 | 20.20 | 0.846 | 15.58 | 12.53 | 0.59 |
| 0.05 | 0.542 | 20.16 | 0.844 | 15.55 | 12.52 | 1.06 |
| 0.1 | 0.542 | 20.13 | 0.842 | 15.58 | 12.52 | 1.20 |
| 0.3 | 0.536 | 20.06 | 0.833 | 15.55 | 12.54 | 1.56 |
| 0.8 | 0.529 | 20.04 | 0.817 | 15.48 | 12.61 | 2.36 |

The experimental chain has CA radius of gyration **22.86 Å** and span **77.18 Å** on the same resolved mask. Predictions remain around 15.5 Å and 50 Å respectively at every tau. Long-range CA distance errors remain approximately 12.5 Å. The highest tau therefore adds variation around the same overly compact geometry, rather than recovering experimental helix placement. Its best TM-score is 0.548, below the default condition's best 0.555; no condition is reported using only its best sample.

Use all 123 resolved chain-A CAs at full polymer positions 4–126 for sequence-correspondent RMSD and compactness. TM-align is normalized to the observed reference and may align a different structural subset; its subset RMSD is not the full sequence RMSD. CA lDDT is pair-weighted. Long-range pair distance error uses polymer separation at least 20. Diversity fits all 126 predicted CAs.

No prediction has a nonadjacent CA pair closer than 2.5 Å. At tau 0.8, three of 1,250 adjacent CA distances fall slightly below the coarse 3.6–4.1 Å interval (minimum 3.577 Å); tau 0.1 has one such distance. These are coarse backbone indicators, not all-atom chemical or steric validation.

This is one protein and one seed with ten samples per condition. Fixed seed and batch size support paired comparisons by sample index, but not independent seed replication. Small GPU numerical differences are possible. The new default condition closely reproduces the [earlier baseline](7wf9_a.md). Neither this sweep nor the baseline establishes the cause of the folding failure or exactly reproduces the paper's unspecified checkpoint.

## Artifacts and reproduction

Local sweep root: `artifacts/sweeps/7wf9_a_tau/2026-10-06/`.

Long-term sweep root: `/pub/ynkim4/ml-simplefold/artifacts/sweeps/7wf9_a_tau/2026-10-06/`.

Both contain the reference CIF, manifest, downloaded predictions and provenance, per-run comparisons, all-sample CSVs, summary JSON/CSV, `tau_sweep.png`, and `index.html`. The index links all five ensembles. Plots and table badges retain the original Matplotlib palette with explicit tau legends and distinct markers; legend labels contain tau values only. Each viewer labels its tau and shows all ten predictions solid orange over the blue experimental reference. Raw remote outputs also remain under `/pub/ynkim4/ml-simplefold/artifacts/runs/JOB_ID/`. Weights remain in the existing public checkpoint/Torch caches. The sweep bundle is about 22 MiB and contains no model weights.

Each successful run used the approved launcher:

```bash
~/.agents/skills/hpc-experiments/scripts/submit.sh \
  hpc_sample.slurm examples/7wf9_a.fasta 500 10 TAU 42
```

Launcher arguments are FASTA, steps, sample count, tau, seed. Exact run settings are recorded in `experiment.json`; validation, source commit, and package versions accompany the PDBs. The manifest preserves the failed first attempt separately. Sampling is complete; these commands document reproduction, not outstanding submissions.

Coordinate analysis can run locally without loading a model. Per-run comparisons use `scripts/compare_structures.py`, chain A, the deposited FASTA, and the reference CIF. Aggregate existing comparisons with:

```bash
python scripts/summarize_tau_sweep.py \
  --manifest artifacts/sweeps/7wf9_a_tau/2026-10-06/manifest.json \
  --output-dir artifacts/sweeps/7wf9_a_tau/2026-10-06
```

QA checked all 50 predictions, the common 123-residue reference mask, all 45 pairwise comparisons per condition, relative viewer/data links, and the summary figure. Reference and prediction checksums are recorded in the analysis JSONs.

## Public-storage incident

The first attempt, job **57840798**, failed with exit **1:0** after **24 seconds** around 11:45 PDT. The bounded model-cache `stat` guard timed out before inference; that attempt loaded no model and generated no samples. SSH, Git, and Slurm remained available while even small public-storage metadata reads and SFTP transfers stalled.

Access recovered at **12:00:44 PDT** without changing storage paths or downloading model weights. `/pub` resolves to `/dfs6b/pub`, on [RCIC BeeGFS storage](https://rcic.uci.edu/storage/dfs.html). The account's dfs6b quota was **606.99 GiB / 1 TiB**, with **166.51k / 8 million** inodes/chunks, below both limits. These are account totals, not this project's footprint.

Login-node client logs show metadata requests toward node **6021** interrupted when bounded probes were terminated. This narrows the observed symptom to metadata RPC waits, but does not identify a server, network, or client root cause. All five sampling jobs succeeded after recovery. Retrieval of one tau-0.3 PDB subsequently stalled again and eventually completed without resampling, so the incident was intermittent rather than demonstrably fixed.

No public [RCIC news notice](https://rcic.uci.edu/about/news.html) confirming an October 6 outage was found; notices may be distributed by email. No support message was sent. If stalls recur, preserve cached models/results and use the [RCIC storage ticket guidance](https://rcic.uci.edu/help/tickets.html#storage-problems) with timestamped failures and affected paths/nodes. Do not work around storage stalls by repeatedly launching GPUs, redownloading model caches, or sampling locally.
