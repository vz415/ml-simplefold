# BMP2 receptor preference fine-tuning plan

Status: research/design only, 2026-10-09. The user wants to revisit a
DPO-modified model after testing frozen-weight FK steering. No preference
trainer, adapter, training dataset or training job has been created.

## Current decision and separate objectives

The configured reward in `configs/sampling/2goo_fk_pilot.yaml` is the adopted
automated objective for this complex: worse-copy receptor hotspot/native
contact recovery and precision, with BMP2/BMPR1A placement and clash penalties.
The weights remain revisable. Record the exact reward/config version when
labeling data; do not relabel old pairs silently after changing coefficients.

FK selects trajectories with the pretrained weights frozen. Preference
fine-tuning would update the folding model toward preferred structures while
comparing against the immutable original SimpleFold-3B reference. The ESM
encoder can remain frozen. A small adapter is a first implementation option;
the repository does not currently contain an adapter path.

## Data to retain from rollouts

Create preferred/rejected pairs for the same FASTA and consistent chain/atom
mapping. Store both structure paths and hashes, FASTA/checkpoint hashes, run
IDs and sampling configuration, raw interface/clash measurements, scalar
rewards and reward version, FK ancestry, and preference source. Keep automated
labels and any later human override separately, including reasons and ties.
The existing viewer can support human decisions about geometry missed by the
scalar reward; no preference annotation UI is implemented yet.

Use clear reward gaps initially rather than forcing ties into confident labels.
Store future pair data under `/pub/ynkim4/ml-simplefold/artifacts/datasets/synthetic/`
and trained checkpoints/adapters under `artifacts/checkpoints/trained/` on the
same remote artifact root. Keep the original pretrained checkpoint immutable.

## Code and objective to inspect before implementation

- `src/simplefold/model/simplefold.py`, `flow_matching_train_step`: upstream
  timestep/noise sampling, masked velocity MSE, rigid alignment and optional
  smooth-lDDT loss. A preference objective needs per-example errors before
  the current batch reduction.
- `src/simplefold/model/flow.py`, `LinearPath`: coordinates interpolate as
  `x_t = (1 - t) * noise + t * structure`, with velocity target
  `structure - noise`; SimpleFold uses noise-to-data time.
- `src/simplefold/model/torch/architecture.py`, `FoldingDiT.forward`: returns
  the `predict_velocity` used to compute those errors.
- `src/simplefold/datasets/train_datamodule.py`: current examples are single
  structures, so paired loading and synchronized preparation need a new path.
- `src/simplefold/train.py` and `configs/base_train.yaml`: existing
  Hydra/Lightning training orchestration to reuse.

[Diffusion-DPO](https://arxiv.org/abs/2311.12908) derives a tractable preference
objective using a diffusion likelihood bound.
[Flow-DPO, section 4.2 and appendix C](https://proceedings.neurips.cc/paper_files/paper/2025/file/76227feb18ea0ee40bd15cf02c33e18e-Paper-Conference.pdf)
adapts preference optimization to velocity regression for rectified flow.
These results motivate a flow-aware objective; they do not establish protein
performance or justify copying diffusion timestep weights unchanged.

A candidate to review is a logistic preference loss comparing preferred and
rejected velocity errors, each relative to the frozen reference. This is a
flow-matching preference surrogate, not an exact final-structure likelihood.
Its timestep weighting, MSE reduction and training coefficient need explicit
choices. The training coefficient is separate from FK's sampling `beta=2`.

Use matched timesteps, noise, masks and whole-complex rigid transforms within
each pair and across trainable/reference evaluations. Upstream optional rigid
alignment depends on each model's prediction; independently applying it can
give the two models different velocity targets. Choose a shared coordinate
gauge or common detached alignment. Independently fitting receptors would
erase the docking error that preferences are meant to correct.

## First evaluation to plan

Compare the original and adapted models on fresh seeds, measuring ordinary
sampling and FK steering separately. Retain receptor hotspot recovery,
placement/orientation, fold accuracy, clashes and ensemble diversity alongside
the scalar reward. Group holdouts by generating run and FK ancestry so related
descendants do not appear on both sides. Improvement on this one FASTA would
establish complex-specific adaptation, not general protein performance.

Revisit once the current FK batch completes and its outputs are inspected.
Then select the paired-loss/alignment design, adapter versus full fine-tuning,
and training resources before implementing and launching a concrete pilot.
