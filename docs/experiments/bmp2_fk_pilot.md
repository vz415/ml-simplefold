# Frozen-weight BMP2 receptor steering pilot

The pilot compares ten baseline particles against ten FK-steered particles
using SimpleFold-3B, 500 EM steps and tau 0.01. Model parameters are frozen;
there is no optimizer or training. The Hydra experiment profile is
`configs/experiment/2goo_fk.yaml`, composed with shared settings in
`configs/sample.yaml` and steering settings in
`configs/sampling/2goo_fk_pilot.yaml`. This is reference-guided reconstruction,
not a binding affinity estimate.

Particles share one new initial Gaussian latent (seed 42), with independent
future draws from Brownian seed 43 and a separate resampling seed 44. Both
conditions use matching streams. This does not replay historical Sample 10.
Actual source states at the requested score checkpoints are saved for future
branching. Checkpoint times 0.60, 0.75, 0.90 and 0.97 select the nearest points
on the existing log time grid. The initial potential is constant one.

The sampler preserves the original EM transition. At late checkpoints it
scores estimated clean coordinates; at the terminal time it scores the actual
finished structure. Log weight increments are `beta * (reward - prior_reward)`
with beta 2. Multinomial resampling occurs when ESS is below 80% of particle
count and stochastic steps remain. Selected coordinates, velocities, rewards
and ancestry travel together. Intermediate proxies are not final measurements.

Positive terms use the worse of two copies for each hotspot/contact recovery
and precision feature. ACVR2A weights are 2, 1, 1 respectively; BMPR1A weights
are 1, 1, 0.5. Penalties are weighted, scaled and capped: BMP2 dimer RMSD
0.25 / 2 Å, worst BMPR1A placement RMSD 0.25 / 5 Å, and clashes per 1,000
heavy atoms 0.5 / 50; each normalized penalty is capped at 4. These are
exploratory coefficients. Undefined precision with explicitly zero predicted
contacts contributes zero for ranking; original metric reporting is retained.

BMPR1A α1 helix-like fraction is a preservation diagnostic. The reference
scores 6/7, including its nonhelical cap; maximizing to 1 is inappropriate.
Calibrated fine-tuning toward a desired distribution is a future possibility,
not part of this frozen-weight pilot.

The corrected symmetry-derived reference must be staged as
`/pub/ynkim4/ml-simplefold/artifacts/datasets/references/2goo-native/reference.cif`
and `reference.pdb`. Reference paths are checked before model loading.
Inspect the composed configuration locally without sampling or loading model
weights:

```bash
python scripts/run_sampling.py experiment=2goo_fk --cfg job --resolve
```

The user authorized the pilot after inspecting the sampler and reward.
Commit/push and pull the remote checkout before submitting future runs:

```bash
sbatch --parsable --gres=gpu:A100:1 --mem=96G --time=01:45:00 --job-name=2goo-3B-fk-pilot scripts/hpc_sample.slurm experiment=2goo_fk
```

The launcher accepts named Hydra overrides rather than positional sampling
arguments. Override `fasta`, `num_steps`, `samples`, `tau`, `seed`, or `model`
after `experiment=2goo_fk`; nested steering settings use names such as
`sampling.beta=1.0`. Ordinary ten-sample 3B inference uses
`scripts/hpc_sample.slurm experiment=2goo_3b` with appropriate `sbatch` resource
flags. Explicit resources override the toy-job defaults and remain separate
from the sampling configuration.

All inference runs through Slurm. Outputs belong in `/pub/.../artifacts/runs/JOBID/`
with `baseline/` and `fk/` conditions, rewards, ESS/parent histories, source
states, parameter immutability checks, final coordinate validation and runtime.
Hydra records `.hydra/config.yaml`, `.hydra/hydra.yaml`, and
`.hydra/overrides.yaml`; `resolved-config.yaml` saves fully resolved task
settings alongside `experiment.json`, Git/package receipts, and the existing
FK summaries. Obsidian notes continue to track decisions and measured outcomes.
Terminal particles retain weights; no terminal resampling is performed.
Unweighted PDB averages describe the returned population, not independent
samples from the reward-tilted distribution. Report weighted summaries too.

Local validation: 14 CPU toy/objective tests passed; beta zero reproduces the
base EM toy trajectory; scoring all existing 3B coordinate files gives Sample
10 the highest reward (2.271). These checks do not establish GPU execution or
steering improvement. No local protein sampling or model loading occurred.

The complete 100M run was retrieved and the comparison viewer updated before
any new pilot submission. A job receipt and measured outcome must be appended
after execution; no successful FK protein rollout is claimed here.

## First A100 submission — 2026-10-09

Job `57994351` was submitted on `hpc3y` as `ynkim4` using the command above,
one A100, 96 GB RAM and a 1h45 limit. It runs ten baseline particles followed
by ten FK particles with the configured seeds and frozen SimpleFold-3B weights.
The submitted source commit is `e206e3c610f2647bfab6c9c3b0779dd79a1b2136`.
All 24 local toy/objective/orchestration checks passed before submission.

The corrected reference CIF/PDB were staged at the explicit reference paths
above and their SHA-256 hashes verified against the local files:

- CIF: `8dd16e85f43060da7cd4a079868826f680f011b36f7c591678374f399276daa8`.
- PDB: `26eb29d9eec1b26c5c7141057e425a9e3503d5df281d31c129e7a217b7906c7b`.

Outputs: `/pub/ynkim4/ml-simplefold/artifacts/runs/57994351/`.
Logs: `/data/homezvol2/ynkim4/ml-simplefold/logs/sample-57994351.{out,err}`.
Initial state was pending for resources; no inference result is claimed in
this submission receipt. Record execution status and outcomes separately.

## A30 memory-test submission — 2026-10-09

At the user's request, job `57994465` tests the same ten-particle 3B
baseline/FK configuration on one A30. A100 job `57994351` remains unchanged.
Submitted source `bfd20e4099844d40c9a4f31ebe973624a74ed749` differs from the
A100 source only in submission documentation, with identical sampler/config.

```bash
sbatch --parsable --gres=gpu:A30:1 --mem=96G --time=01:45:00 --job-name=2goo-3B-fk-A30-test scripts/hpc_sample.slurm experiment=2goo_fk
```

Outputs: `/pub/ynkim4/ml-simplefold/artifacts/runs/57994465/`.
Logs: `/data/homezvol2/ynkim4/ml-simplefold/logs/sample-57994465.{out,err}`.
The 96 GB request is host RAM, not GPU VRAM. This is a fit/runtime test;
submission alone does not establish that encoding or sampling fits on A30.

Job `57994465` failed after 2m26s with CUDA OOM in `batch_to_device`, before
ESM feature computation or sampling. Both models had loaded, but staging
replicated ten-particle inputs requested 2.30 GiB with only 967.44 MiB free
on the 23.60 GiB A30. No baseline/FK structures were produced.

## Model evaluation in chunks of two

Use `sampling.model_batch_size=2` to retain a ten-particle FK population while
evaluating velocities two particles at a time. Rewards, ESS, parent selection
and Brownian draws still span all ten particles at each step. The default
`null` preserves full-batch inference. CUDA kernels can produce small numeric
differences between batch sizes; chunking does not change the intended kernel
or introduce five separate two-particle selection pools.

For chunked runs, input preparation uses one copy. Folding weights are moved
to CPU while ESM features are computed under `no_grad`; the encoder is then
released before folding weights return to GPU. Shared conditioning is expanded
as tensor views, avoiding ten physical copies of dense input features.
Model outputs are collected in the original particle order before selection.
Checkpoint metadata records the configured model batch size and shared storage.

```bash
sbatch --parsable --gres=gpu:A30:1 --mem=96G --time=01:45:00 --job-name=2goo-3B-fk-A30-chunk2 scripts/hpc_sample.slurm experiment=2goo_fk sampling.model_batch_size=2
```

All 26 local toy/objective/orchestration tests pass. The chunking test covers
a final partial chunk, identical baseline/steered CPU trajectories, parent
selection across model chunks and unchanged RNG/history. A storage test checks
that expanded features retain the singleton backing allocation. These tests
do not establish A30 fit; that requires remote execution.

Chunked A30 job `57994634` uses source `0045498` and the override above. It
passed singleton preparation and ESM encoding, reported 11.26 GiB allocated
before sampling, and was running its baseline at the latest check. No finished
FK results are claimed yet. At the user's request, queued A100 job `57994351`
was cancelled at 23:43:15 PDT on 2026-10-09 without running any GPU compute.
A100 remains the launcher's default, with explicit chunked A30 fallback.

The user adopted the current configured interface/clash reward as the
automated objective for subsequent FK sampling. Its coefficients remain
revisable; this decision does not establish measured steering improvement.
A separate preference fine-tuning plan is recorded in
[`bmp2_preference_finetuning.md`](bmp2_preference_finetuning.md).

The earlier positional launch commands in historical experiment receipts
refer to the launcher before this Hydra refactor. Use the named profiles
above with the current checkout.
