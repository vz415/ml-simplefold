# 7WF9-A: ten samples from SimpleFold-100M

All ten samples share a very similar, incorrect global arrangement. Median sequence-correspondent Cα RMSD against the crystal chain is **20.20 Å** (19.99–20.24 Å), while median pairwise RMSD between samples is **0.59 Å** (0.21–1.25 Å). More draws from this default setup did not recover the experimental arrangement in this batch.

## Run and reference

- HPC3y job **57838536**, completed with exit 0 on `hpc3-gpu-k54-05`, 2026-10-06 11:03:59–11:05:09 America/Los_Angeles (70 seconds).
- Smallest released folding checkpoint: **SimpleFold-100M**, with the **ESM2-3B** feature encoder; PyTorch/CUDA backend.
- Launcher defaults: **500 sampling steps, tau 0.01, seed 42, pLDDT disabled, PDB output**. `nsample_per_protein=10` draws ten samples in one seeded RNG stream; these are not ten independent seed jobs.
- Sampling source commit: `6e7aff89d98d5919012cea839c7ffd400b3d8220` on `experiments/hpc3y-setup`. GPU allocation: one GPU, four CPUs, 32 GiB RAM, 15-minute limit. Recorded maximum host RSS was about 17.0 GiB.
- Input: [`examples/7wf9_a.fasta`](../../examples/7wf9_a.fasta), full 126-residue RCSB entity-1 sequence. All ten PDBs passed sequence, residue-count, and finite-coordinate checks.
- [RCSB 7WF9](https://www.rcsb.org/structure/7WF9): mouse SNX25 RGS domain, 2.55 Å X-ray structure. Author chain A resolves **123/126 residues**; polymer positions 1–3 (`PQS`) have no coordinates. Comparisons omit those three residues, use mmCIF `label_seq_id` for sequence correspondence, and cover all 123 observed Cα atoms.
- Reference mmCIF SHA-256: `f831ab0d18ce1294854a7d581a32c262560f442d5e844f79dbb22952eb45ba9d`.
- FASTA SHA-256: `35cf4b6e93ed4fff6352a10b05ee33756ab36c5332ebe0c21434093845d4e9ac`.

## Scores for every sample

Higher TM-align and CA lDDT scores are better; lower RMSD is better.

| Sample index | TM-align / observed reference | Sequence Cα RMSD (Å) | Pair-weighted CA lDDT |
| --- | ---: | ---: | ---: |
| 0 | 0.5433 | 20.20 | 0.8407 |
| 1 | 0.5434 | 19.99 | 0.8452 |
| 2 | 0.5548 | 20.15 | 0.8535 |
| 3 | 0.5452 | 20.24 | 0.8502 |
| 4 | 0.5464 | 20.19 | 0.8465 |
| 5 | 0.5474 | 20.17 | 0.8505 |
| 6 | 0.5404 | 20.24 | 0.8350 |
| 7 | 0.5443 | 20.20 | 0.8493 |
| 8 | 0.5412 | 20.21 | 0.8340 |
| 9 | 0.5424 | 20.19 | 0.8357 |

TM-align median is **0.5439**, range **0.5404–0.5548**. Sample 2 has the highest TM-align score; it still has **20.15 Å** full observed-chain sequence RMSD. This best selection uses the experimental structure, not model confidence.

TM-align permits a structural alignment different from the target sequence correspondence. For sample 2 it reports **2.36 Å RMSD over 79 structurally aligned pairs**, not all 123 corresponding residues. That subset score must not be interpreted as full-chain accuracy. Median absolute internal CA-pair distance error across all resolved residue pairs is **9.21 Å**; sample 2's 90th percentile absolute error is **28.03 Å**.

The pair-weighted CA lDDT variant has median **0.8458** (0.8340–0.8535): local distances can agree while long-range relative placement is wrong. It uses reference CA pairs within 15 Å and thresholds 0.5/1/2/4 Å, averages by pair count, and is **not** the paper's all-atom OpenStructure lDDT. Additional single-fit GDT-like statistics in the CSV/JSON are explicitly labeled; they are not optimized canonical GDT_TS.

An independent check using the same 123 resolved CAs confirms that the predictions are too compact: experimental CA radius of gyration **22.86 Å**, versus **15.49–15.64 Å** for the samples; maximum CA span **77.18 Å**, versus **49.74–50.67 Å**. Mean absolute CA-pair distance errors are only **0.36–0.40 Å** for polymer separations 1–4, but **12.49–12.56 Å** for separations at least 20. An independent Bio.PDB superposition reproduces sample 2's sequence RMSD and verifies mapping of polymer positions 4–126 to author residues 283–405.

## Relation to the paper

The [paper's failure-mode discussion (F.3 / Figure 11)](https://arxiv.org/pdf/2509.18480v1#page=27) describes largely correct helices with incorrect relative placement for 7WF9-A. Our high local-distance agreement and large global errors are consistent with that observation. Figure 11 does not identify its checkpoint size, seed, or sample selection, so this 100M experiment is not an exact reproduction of that figure. The authors suggest an ESM2 representation limitation; this run does not establish that cause.

The deposited biological assembly is a C2 homodimer. This experiment predicts and compares one isolated chain, and does not test oligomer context.

## Artifacts and reproduction

Remote long-term storage:

```text
/pub/ynkim4/ml-simplefold/artifacts/references/7WF9/
/pub/ynkim4/ml-simplefold/artifacts/runs/57838536/predictions_simplefold_100M/
/pub/ynkim4/ml-simplefold/artifacts/runs/57838536/comparison/
```

Local ignored artifacts:

- `artifacts/remote-runs/57838536/`: original ten PDBs and run provenance/validation.
- `artifacts/analysis/7wf9_a/57838536/`: `metrics.csv`, `comparison.json`, per-residue distances, pairwise RMSDs, aligned PDBs, plots, and interactive superposition HTML. The JSON includes reference, FASTA, prediction, and analysis-script hashes plus analysis package versions.

With the base local SimpleFold environment plus `python -m pip install -r requirements/analysis.txt`, compare downloaded PDBs without loading a protein model:

```bash
python scripts/compare_structures.py \
  --reference artifacts/references/7WF9/7WF9.cif --chain A \
  --fasta examples/7wf9_a.fasta \
  --prediction-dir artifacts/remote-runs/57838536/predictions_simplefold_100M \
  --output-dir artifacts/analysis/7wf9_a/57838536
```

Sampling invocation on HPC3y (already completed; do not rerun merely to reproduce analysis):

```bash
sbatch hpc_sample.slurm examples/7wf9_a.fasta 500 10
```

Analysis verification: four fixture tests cover unresolved-prefix sequence mapping, author residue numbering/insertion codes, alternate CA occupancy selection, rigid transformations/reflection rejection, and end-to-end output/normalization checks. No protein-model sampling was performed locally.
