# 2GOO complex sampling with SimpleFold

> **2026-10-06 correction:** the deposited six-chain asymmetric unit contains two
> separate half-complexes. It must be reconstructed with crystal symmetry to
> represent the native BMP2 homodimer. Earlier whole-assembly scores using
> the asymmetric unit are superseded; original file audits remain historical
> observations. See [ensemble and corrected-reference notes](2goo_ensemble.md).

The requested experiment compares the smallest (`simplefold_100M`) and largest
(`simplefold_3B`) released folding checkpoints on the six-chain BMP2–BMPR1A–ActRIIA
extracellular complex. Begin with one pilot sample per model at the CLI defaults:
500 steps, tau 0.01, seed 42, PyTorch, pLDDT disabled. Both use the ESM2-3B encoder.

Input: `examples/2goo_hexamer.fasta`, six separate chains A–F, totaling 694
residues. Sequences come from the original PDB SEQRES records; provenance and
hashes are in `examples/2goo_hexamer.provenance.json`. Unresolved terminal
residues and the deposited BMPR1A cloning residues are included. Waters and
glycans are excluded. The PDB coordinates are a reference, not an inference
template or initial state. See [reference quality notes](2goo_structure_audit.md).

The inference parser formerly assigned every record to chain A. It now retains
distinct IDs for multi-record inputs while preserving the single-record A
convention. Sequence extraction now separates chains by `asym_id` rather than
`entity_id`, so adjacent identical copies remain distinct in ESM input. Seven
regression tests pass locally without loading any models. These repairs enable
an exploratory assembly experiment; they do not establish model accuracy on
complexes. Assembly geometry and individual-chain quality must be assessed
separately.

Run through Slurm from `/data/homezvol2/ynkim4/ml-simplefold` on `hpc3y`:

```bash
sbatch --gres=gpu:A100:1 --mem=96G --time=01:00:00 \
  scripts/hpc_sample.slurm examples/2goo_hexamer.fasta 500 1 0.01 42 simplefold_100M
sbatch --gres=gpu:A100:1 --mem=96G --time=01:00:00 \
  scripts/hpc_sample.slurm examples/2goo_hexamer.fasta 500 1 0.01 42 simplefold_3B
```

The optional sixth launcher argument selects the model; it defaults to 100M.
Missing weights download within the allocated job, using a resumable `.part`
file and atomic rename after successful download. The 3B folding weights are
additional to the shared ESM encoder cache. Long-term checkpoint and output
paths remain `/pub/ynkim4/ml-simplefold/artifacts/checkpoints/` and `runs/<job>/`.
The launcher checks storage accessibility before downloads or inference.
Prediction validation checks the six chain sequences individually, full
residue coverage, finite coordinates, and one Cα per residue. These are file
integrity checks, not stereochemical validation or evidence of correct binding.

## Current status

2026-10-06, approximately 19:10 PDT: source commit
`885c4f1d85946d17635e7b0bb02240099fed9ec1` was pushed and pulled on HPC3y.
Direct `hpc3` and `hpc3y` SSH work, but login-node checks of the `/pub`
checkpoint directory and cached 100M file timed out.

- **100M job 57873686:** failed with exit `1:0` after 24 seconds on
  `hpc3-gpu-l54-04`. The launcher reported `Model storage is unavailable;
  stopping before inference.` No sampling or new checkpoint download occurred.
- **3B job 57873706:** submitted with `afterok:57873686`, then explicitly
  canceled after the small pilot failed. It did not run or download weights.
- **Transfer-endpoint check:** retrieving the existing 5,372-byte sweep HTML
  from `/pub` through `access-hpc3.rcic.uci.edu` succeeded. This shows the
  existing artifact was accessible through that endpoint; it does not prove
  that checkpoint reads or compute-node mounts work.

The initial failure was model-cache access from the compute node, not an
observed model or multichain inference failure. Preserve the existing weights
and storage paths; do not repeatedly launch GPU jobs to probe a stalled mount.
Logs are `logs/sample-57873686.out` and
`logs/sample-57873686.err` in the remote checkout. The project Obsidian receipt
is `experiments/2026-10-06_191045_2goo-model-pilots.md`.

### Recovery and small-model result

Subsequent login-node metadata checks and one-byte reads of the three required
cache files succeeded. Retry **57873825** then sampled on the A100 successfully.
It wrote all **694 residues in six chains, 5,464 atoms**, with finite coordinates.
Sampling used source `885c4f1`; no input structure coordinates were supplied.
The 500 sampling steps took approximately **15 seconds**; total job time was
61 seconds including model loading and the initial validation failure.

The initial validator rejected the saved chain order, so Slurm reports this
retry as `FAILED 1:0` despite a complete prediction. The schema groups identical
sequences in first-appearance order and its PDB writer relabels the result:

| Saved chain | Input / crystal author chain | Component |
| --- | --- | --- |
| A | A | BMP2 |
| B | D | BMP2 |
| C | B | BMPR1A |
| D | E | BMPR1A |
| E | C | ActRIIA |
| F | F | ActRIIA |

Validator correction `ad2942f` reproduces this grouping and records the mapping.
The existing sample passed this corrected local coordinate-only validation;
it was not resampled. The PDB SHA256 is
`aecf2a395b7a5a5634c4ad378e7d2e296b7b53c3464137866cc0b333cdd9dc36`.
Validation provenance preserves the original failed Slurm status and separate
inference/validation versions. The intermediate dependent 3B job **57873828**
was canceled before running while validation was corrected.

Raw prediction: `/pub/ynkim4/ml-simplefold/artifacts/runs/57873825/predictions_simplefold_100M/2goo_hexamer_sampled_0.pdb`.
Local copy and validation: `artifacts/remote-runs/57873825/`.
Local overlay, metrics, and coordinate-analysis script:
`artifacts/analysis/2goo_models/2026-10-06/`.
These analysis artifacts and corrected validation reports are also archived
under `/pub/ynkim4/ml-simplefold/artifacts/analysis/2goo_models/2026-10-06/`
and the original remote run directory, respectively. The raw prediction hash
matches locally and remotely. The viewer JavaScript passes a syntax check;
interactive browser rendering has not been verified in this session.

The comparison scores only the **565 resolved reference Cα atoms**, using
canonical full-polymer positions from the RCSB mmCIF. All eight assignments of
the equivalent chain copies are checked; the assignment with minimum global
Cα RMSD is used. Individual-chain RMSDs use separate fits. One sample gives:

| Measure | 100M pilot |
| --- | ---: |
| Globally aligned assembly Cα RMSD ↓ | 32.355 Å |
| Assembly Cα lDDT ↑ | 0.6314 |
| Interchain Cα contact recall at 8 Å ↑ | 2.86% |
| BMP2 separately aligned Cα RMSD ↓, two copies | 1.12 / 1.36 Å |
| BMPR1A separately aligned Cα RMSD ↓, two copies | 5.48 / 5.81 Å |
| ActRIIA separately aligned Cα RMSD ↓, two copies | 3.58 / 3.72 Å |

The BMP2 folds are substantially closer than the overall assembly. This pilot
does not recover the experimental binding arrangement. Cα proximity metrics
are not DockQ or all-atom interface contacts; density-quality masking was not
applied beyond excluding unresolved reference residues. The crystal coordinates
were not repaired.

### Large-model run

**57873967** uses source `ad2942f`, the same input/settings, an A100, 96G CPU
memory, and a one-hour limit. Submitted after validating the saved 100M sample,
without a dependency on its failed Slurm status. At approximately 19:15 PDT it
was `PENDING (Resources)`; Slurm estimated a 22:08 PDT start. This estimate can
change. All A100 GPUs were allocated; lowering CPU memory alone would not
provide an available GPU. There is no 3B prediction yet, and the job has not
downloaded its weights. It will populate the 3B checkpoint under `/pub` within
the allocation and validate its saved chains automatically after sampling.

Check it with `squeue -j 57873967` and
`sacct -j 57873967 --format=JobID,State,ExitCode,Elapsed`.
Logs: `logs/sample-57873967.out` / `.err` in the remote checkout.
Expected prediction: `/pub/ynkim4/ml-simplefold/artifacts/runs/57873967/predictions_simplefold_3B/2goo_hexamer_sampled_0.pdb`.
