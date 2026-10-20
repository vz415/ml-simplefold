# General-data replay mixed with six synthetic complexes

Authorized 2026-10-20: start again from the original released SimpleFold100M
checkpoint, mix general fine-tuning structures with the six refined synthetic
complexes, and train on HPC3y with data and results under `/pub`.

## Sampling and data

At each of 600 updates, choose synthetic with probability 0.1 and general data
with probability 0.9. Synthetic examples are sampled uniformly over the six
complexes. General examples are sampled uniformly from the pooled eligible PDB
and author-filtered SwissProt catalogs. The mixture is a random source/record
schedule generated before training and saved as `training-plan.json`; the
loader replays that schedule without shuffling. Expected counts are 540 general
and 60 synthetic, with realized counts logged. These are update probabilities,
not percentages of disk space, atom counts, or elapsed GPU time.

Only general entries drawn for this finite run are downloaded/prepared. This
is sampling from the full eligible catalog, not mirroring the complete corpus
or choosing examples by prediction quality. Raw and processed structures stay
in `/pub/ynkim4/ml-simplefold/artifacts/datasets/general-finetuning-v1/`.
The plan-specific directory is `mixed-600-seed82301`. The existing synthetic
dataset remains in `datasets/synthetic/multimer-refined-v1`.

The PDB catalog uses a 2020-05-01 release cutoff and resolution no worse than
5 Å. The numeric-resolution query excludes entries without a resolution,
including most NMR entries; the upstream filter can admit these because its
parser defaults missing resolution to zero. SwissProt candidates
come from the authors' filtered list; their version4 predictions must be
available without silently substituting a newer model. Current PDB files can
include revisions, so this is a documented reconstruction, not an assertion of
byte-identical historical training data. Data preparation records source URLs,
hashes, exclusions, and any candidate replacement. The held-out CAMEO22
sequences are screened out during general-data preparation.

## Training

`configs/mixed_train.yaml` controls the mix and training budget. Entry point:
`scripts/train_mixed.py`, via the approved `hpc_sample.slurm --train-mixed`
dispatcher. CPU preparation runs with `phase=prepare`; the dependent A100
training job runs with the default `phase=train`. Both use explicit `/pub`
artifact paths. No sampling or training runs locally or on login nodes.

Keep the first six-only pilot settings fixed to isolate the mixture change:
600 updates, learning rate1e-5, warmup30, EMA0.95, constant smooth-lDDT weight1,
batch/multiplicity1. Start from original checkpoint SHA256
`4cd0b8a0b317a6ab8634444fffd78ce84cfd49c20fe927b83c76c36fda5f54bd`;
do not resume the six-only fine-tuned checkpoint. General examples use native
512-residue crops; the synthetic examples retain their full six-chain
assemblies. ESM2-3B remains frozen and is evaluated on demand for each crop,
avoiding a large cache of all37-layer embeddings. Optimize folding parameters
only. Keep raw/EMA and resumable training checkpoints in
`checkpoints/trained/JOB_ID/`.

Save source-specific losses and actual source/record draws. Require the loader
to return the scheduled record, since upstream fallback to another record would
silently change the requested experiment. Nonfinite losses/gradients fail the
run. The data plan, configuration, checkpoint hashes, source commit and actual
counts are saved alongside outputs.

## Evaluation

Generate the same three unguided samples per training complex for each weight
set as in the six-only pilot: common seeds, 500 steps, tau0.3. These36 structures
measure reconstruction of synthetic training targets. Evaluate broad folding
retention separately on the frozen24-target CAMEO22 subset with its existing
sampling and OpenStructure settings. Do not use the benchmark references as
general training data. Improvements on the six complexes do not establish
generalization to unseen complexes.
