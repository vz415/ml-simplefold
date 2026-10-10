# 2H62 topology-aware FK pilot

Approved 2026-10-10 after inspection of the tau-0.3 ensemble: the unobserved
side placed receptors at inappropriate interfaces. The original reward's
typed site assignment only assigns labels; its narrow hotspot windows provide
effectively zero discrimination far from a target. A separate reward variant
now adds a broad interface-distance penalty for all four receptor copies.

## Implementation and objective

[TopologyComplexReward](../../scripts/fk_2h62_topology_reward.py) extends
[PartialComplexReward](../../scripts/fk_2h62_reward.py). The wrapper selects
it through `reward_kind: 2h62_partial_topology`; the old `2h62_partial` profile
retains its original objective. The FK sampler, model weights, Brownian kernel,
ESS rule and resampling mechanism are unchanged.

For each receptor type, choose six spatially spread Cα anchors from the observed
native-contact residues. Choose four Cα landmarks from **each** BMP2 monomer:
start at an interface/nearest position, then spread over the monomer. Their
48 receptor–ligand distances describe the measured interface and its context
in the dimer. If too few interface CAs exist, additional observed receptor CAs
are used. Actual anchor identities and target distances are saved in metadata
and per-copy scoring details.

The opposite site's targets exchange **both** BMP2 chain identities. No
mirrored or rotated receptor coordinates are constructed. Site zero uses an
experimental distance pattern; site one uses an inferred ligand-swapped
pattern. Exact symmetry is not enforced, and the inferred interface is not
experimental ground truth.

For each anchor pair j in receptor copy i:

```text
e_ij = max(abs(d_pred_ij - d_ref_j) - 2 Å, 0)
u_i  = RMS_j(e_ij) / 5 Å
h_i  = 0.5 u_i²                       if u_i <= 1
       u_i - 0.5                     otherwise
T    = 0.5 mean_i(h_i) + 0.5 max_i(h_i)
R    = original hotspot contributions - original penalties - T
```

All four receptors contribute to T. The new term has weight 1, scale 1 and
**no cap**, preserving distinctions at long distances. Missing anchor pairs
contribute a fixed 50 Å excess error rather than shrinking the denominator;
coverage is reported. Existing full-sequence/Cα input checks also remain.
The positive maximum remains +5, but the new uncapped penalty removes the old
-8 lower bound. This is a ranking objective with heuristic scales, not energy,
affinity or a guarantee of correct assembly.

All eight coherent type-preserving assignments are scored using the complete
objective. Identical receptor copies may exchange labels; BMPR1A retains wrist
targets and ACVR2B knuckle targets. Per-copy hotspot definitions are unchanged,
although the selected assignment can change when the topology term is added.
Hotspot windows remain active at the established scoring checkpoints; no
time-dependent hotspot ramp was introduced.

## Validation before submission

**51 relevant checks passed:** 11 focused topology tests, nine original partial
reward tests, and 31 existing wrapper/launcher/FK sampler tests. They cover
typed swapped and duplicated-site arrangements, all four copies, increasing
displacement costs, identical-copy and rigid-motion invariance, asymmetry
within the deadband, declared Huber aggregation, missing anchors and routing.
The focused topology tests isolate the new term from chemistry; original
fixed-geometry tests remain responsible for bond and clash behavior.

Coordinate-only rescoring of all 20 downloaded baseline/tau-0.3 PDBs confirmed
**exactly unchanged original rewards and features** after the parent hooks were
factored out. Rescoring with the new objective gives nonzero interface errors
on the misplaced extra copies instead of negligible Gaussian differences.
These checks validate the mathematical behavior; they do not demonstrate
successful guided sampling.

Offline check artifacts are under ignored
`artifacts/analysis/2h62-inpaint/2026-10-10/topology-reward-checks/`:
`settings.json`, `baseline.json` and `fk_tau0.3.json`.
The new objective ranks original baseline sample 4 at -2.690, sample 9 at
-2.751 and sample 8 at -2.804. Sample 8's missing ACVR2B has 22.9 Å RMS excess
distance error; the other candidates have more balanced but imperfect interfaces.
Original FK tau-0.3 sample 1 scores -4.769, including topology penalty 1.398.
These values use the new objective and should not be compared numerically
with rewards from the old objective as though the scales were unchanged.

## Authorized sampling pilot

The separate [experiment profile](../../configs/experiment/2h62-inpaint-fk-topology.yaml)
inherits [topology sampling settings](../../configs/sampling/2h62_fk_topology.yaml).
One A100 job: **SimpleFold-3B, ten particles, 500 steps, tau 0.3, beta 2,
ESS threshold 0.8, checkpoints 0.60/0.75/0.90/0.97 plus terminal scoring**.
Initial/Brownian/resampling seeds remain 42/43/44 with common-start branching.
`model_batch_size=2` retains the original tau-0.3 run's batching while changing
the GPU from A30 to A100; comparisons need this hardware qualification.
Storage preflight uses 600 seconds. No new ordinary baseline is generated.

```bash
sbatch --parsable --gres=gpu:A100:1 --mem=96G --time=01:15:00 \
  --job-name=2h62-fk-topology-tau0.3 scripts/hpc_sample.slurm \
  experiment=2h62-inpaint-fk-topology
```

Submit from `/data/homezvol2/ynkim4/ml-simplefold` on `hpc3y` after publishing
and pulling the tested source. Outputs use
`/pub/ynkim4/ml-simplefold/artifacts/runs/JOB_ID/fk/`; logs remain in the home
checkout's `logs/sample-JOB_ID.{out,err}`. Record the actual job ID and source
below after successful submission. Existing sweep jobs keep their old profiles.

Evaluate receptor-site geometry, original hotspot diagnostics, chemical geometry
and ancestry/diversity after completion. Rescore old and new ensembles with
**both** objectives for a meaningful comparison. Soft penalties cannot guarantee
the topology or recover candidates that the initial population never explores.
Post-processing and broader completion experiments remain separate work.

Related: [original 2H62 sweep and first evaluation](2h62-inpaint.md).
