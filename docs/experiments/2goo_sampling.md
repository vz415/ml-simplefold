# 2GOO complex sampling with SimpleFold

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
`entity_id`, so adjacent identical copies remain distinct in ESM input. Five
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

2026-10-06: direct `hpc3` and `hpc3y` SSH work. Initial login-node checks of the
`/pub` checkpoint directory and cached 100M file timed out. No predictions
exist yet for this experiment. A compute-node pilot will distinguish a
login-node mount problem from a general storage outage. Do not interpret a
storage failure as a model or multichain failure.
