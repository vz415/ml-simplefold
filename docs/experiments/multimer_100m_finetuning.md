# All-complex synthetic fine-tuning capacity pilot

User authorized 2026-10-19: fine-tune the smallest (100M) folding model on all
selected refined multimers, with ESM2 fixed. This is reconstruction of training
complexes, not a held-out generalization experiment.

The six targets are BMP2–BMPR1A–ACVR2B, BMP4–BMPR1A–ACVR2B,
BMP6–ACVR1–ACVR2B, BMP7–BMPR1A–ACVR2A, BMP7–ACVR1–ACVR2A,
and GDF5–BMPR1A–ACVR2B. Each contributes one selected Rosetta-refined synthetic
target, hash pinned in `configs/datasets/multimer_refined_v1.json`. BMP2–BMPR1A–
ACVR2A has no verified relaxation; the exploratory activin A ensemble is not
included as a solved target. All six are assigned to training.

Dataset: `/pub/ynkim4/ml-simplefold/artifacts/datasets/synthetic/multimer-refined-v1`.
The converter writes native structures, records, tokens, manifest, FASTAs and
provenance. It disables biological assembly expansion, retains all six chains,
corrects entity IDs by identical sequence and symmetry IDs within each entity,
and checks every record can be featurized without cropping or silent replacement.

Training config: `configs/multimer_train.yaml`. Start from pretrained 100M folding
weights; 600 optimizer updates, batch/multiplicity 1, full assemblies, AdamW
1e-5, 30-step warmup, original flow-matching plus smooth-lDDT loss, EMA decay .95.
ESM2-3B representations are computed once under no-grad and cached; no encoder
is present in the training module or optimizer. Only the folding model is optimized.
The subclass retains the upstream loss, bypasses its Fairscale encoder wrapper,
and updates EMA after optimizer updates. Both raw and EMA architecture weights
and full Lightning checkpoints are saved under `checkpoints/trained/JOB_ID/`.

Following training, generate three unguided samples per complex from each of
pretrained and fine-tuned EMA weights: 500 steps, tau .3, common seeds and cached
conditioning. These 36 outputs assess reconstruction and initial diversity only.
No Rosetta relaxation is applied to either comparison arm. Analyze geometric
agreement to synthetic training targets separately from available experimental
references. Fine-tuning cannot independently validate synthetic target correctness.

Submit through the existing approved project launcher:

```bash
sbatch --job-name=multimer-100M-finetune --gres=gpu:A100:1 --mem=96G \
  --time=03:00:00 scripts/hpc_sample.slurm --train-multimers
```

The launcher installs only the small pinned Fairscale import dependency when
needed; all conversion, ESM encoding, training and inference occur in Slurm.
The isolated worktree/branch is `ml-simplefold-multimer-train` /
`experiments/multimer-100m-finetune`. Existing sampling experiments and the
upstream training implementation are retained.

## Submitted jobs

- A100 training + matched evaluation: **58274048**.
- CPU-only conversion/featurization preflight: **58274101**. Training depends on
  its successful completion, avoiding consumption of a GPU on malformed data.
- Source PDB hashes were verified after the transfer to `/pub`.
- The DockQ reconstruction benchmark passed identity and equivalent-chain-swap
  controls (~1.0); displacing one receptor by 100 Å reduced its score to .0028.
  The four-interface mean fell to .7507, as expected for three intact interfaces.

Analyze downloaded outputs with DockQ 2.1.3 available:

```bash
python scripts/benchmark_multimer_learning.py \
  --run /path/to/downloaded/JOB_ID \
  --source /path/to/multimer-refined-v1/source \
  --selection configs/datasets/multimer_refined_v1.json \
  --output /path/to/analysis
```

Scores explicitly refer to **synthetic training targets**. They must not be
labeled DockQ to an experimental reference or held-out validation performance.
