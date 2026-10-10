# Frozen-weight BMP2 receptor steering pilot

The pilot compares ten baseline particles against ten FK-steered particles
using SimpleFold-3B, 500 EM steps and tau 0.01. Model parameters are frozen;
there is no optimizer or training. Configuration:
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
Submit only after committing/pushing and pulling the remote checkout:

```bash
sbatch --parsable --gres=gpu:A100:1 --mem=96G --time=01:45:00 --job-name=2goo-3B-fk-pilot scripts/hpc_sample.slurm examples/2goo_hexamer.fasta 500 10 0.01 42 simplefold_3B configs/sampling/2goo_fk_pilot.yaml
```

The seventh launcher argument selects the pilot; existing six-argument
sampling is preserved. Explicit resources override the toy-job defaults.
All inference runs through Slurm. Outputs belong in `/pub/.../artifacts/runs/JOBID/`
with `baseline/` and `fk/` conditions, rewards, ESS/parent histories, source
states, parameter immutability checks, final coordinate validation and runtime.
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
