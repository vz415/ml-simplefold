# Restrained Rosetta refinement: BMP2–BMPR1A–ACVR2B

## Purpose and pilot inputs

Reduce steric clashes and covalent bond-length strain without losing the useful
FK-generated docking geometry. The user authorized starting with this complex;
BMP2–BMPR1A–ACVR2A is the next target after reviewing this pilot.

| Source job | Noise τ | Displayed sample | Raw file suffix |
|---|---:|---:|---|
| 58030762 | 0.3 | 5 | `sampled_4.pdb` |
| 58030763 | 0.8 | 6 | `sampled_5.pdb` |

Both predictions have 592 residues: BMP2 A/B (114 each), trimmed BMPR1A C/D
(84 each), ACVR2B E/F (98 each). BMPR1A retains original sequence positions
34–117. The observed 2H62 reference validates the deposited 2:1:1 portion;
the second receptor copies remain predictions.

## Protocol

Reuse the HPC3 `rosetta/3.13` module with its existing database, through Slurm
CPU jobs. No package installation or model inference is needed.

- Cartesian FastRelax with `ref2015_cart`, one repeat and one output per input.
- Persistent start-coordinate backbone restraints, σ = 0.5 Å.
- Four repacking/minimization stages with repulsive weights 0.02, 0.25, 0.55,
  1.0. Each custom `ramp_repack_min` stage explicitly keeps coordinate weight
  1.0; jump motion is disabled.
- Fixed sequence, no terminal extension. Side chains may repack, and local
  backbone coordinates may move. Rosetta may add hydrogens and terminal atoms.
- All 27 expected disulfides are supplied explicitly from native connectivity,
  including BMP2 A78–B78; they are never inferred from predicted SG proximity.
- Deterministic Rosetta seed 73. Raw FK PDBs remain untouched.

Configuration: `configs/refinement/2h62_rosetta_pilot.yaml`.
Entry point: `scripts/run_rosetta_refinement.py`.
Analysis: `scripts/analyze_rosetta_refinement.py`.

```bash
sbatch --array=0-1 scripts/hpc_refine.slurm
```

Each task writes to `/pub/ynkim4/ml-simplefold/artifacts/runs/<task-job-id>/`:
raw original, prepared input and explicit connectivity, resolved Hydra config,
source FK receipt, exact Rosetta command/executable hash, score file, log,
refined PDB, preparation check, and before/after comparison JSON.

## Improvement tracking

Compare original, prepared and refined structures with identical chemistry
rules and original chain identities. Preparation must preserve the original
coordinates and metrics exactly. Report:

- Heavy-atom clashes: raw, interchain and per 1,000 heavy atoms, using the same
  original fixed bond graph to exclude covalently related pairs.
- Bond-length deviations, bonds beyond the existing tolerance, and all 27
  disulfide lengths. These do not replace bond-angle or chirality validation.
- Whole-assembly Cα drift and per-receptor drift after fitting the BMP2 dimer,
  plus independently fitted chain RMSD to separate docking and fold motion.
- Each receptor's native-contact recall/precision and hotspot geometry under
  the **original** coherent assignment, alongside independently rescored FK
  rewards. Refined rewards are explicitly rescored, not original sampling data.
- Added atoms and exact sequence/chain correspondence; output residue numbering
  may change but does not define correspondence.

Lower Rosetta energy alone is insufficient. Review clash/bond improvement
against contact preservation and docking drift. This is a restrained local
repair pilot, not independent validation of inferred receptor sites or binding
affinity. Full stereochemical validation and any additional relaxation remain
separate decisions.

## Validation before submission

The analysis tests cover exact reward reproduction on both downloaded pilot
inputs, zero changes for a no-op, known chain mapping with changed residue
numbers, sequence/missing-atom rejection, and separation of receptor translation
from fold drift. Native connectivity and coordinate-preserving preparation are
checked on the actual pilot files.

## Submission — 2026-10-11 10:43 PDT

Array **58031292** started both tasks immediately on standard CPU nodes.

| Array task | Numeric task job / artifact directory | Input |
|---|---|---|
| 58031292_0 | 58031293 | τ=0.3, source 58030762, displayed sample 5 |
| 58031292_1 | 58031292 | τ=0.8, source 58030763, displayed sample 6 |

Source commit: `cec470e`. Remote command: `sbatch --array=0-1 scripts/hpc_refine.slurm`
from `/data/homezvol2/ynkim4/ml-simplefold`. Python entry point:
`python scripts/run_rosetta_refinement.py`; case selection comes from
`SLURM_ARRAY_TASK_ID`, with no explicit Hydra overrides. Resource request per
structure: 1 CPU, 8 GB, 1 hour, account `eehui_lab`, QoS `normal`.

Remote outputs: `/pub/ynkim4/ml-simplefold/artifacts/runs/58031293/` and
`/pub/ynkim4/ml-simplefold/artifacts/runs/58031292/`.

Local downloads will be under `artifacts/remote-runs/<numeric-task-job>/`.
Local checkout had unrelated viewer/research-note edits; only the new refinement
workflow, tests, config, allowlist entry and this protocol note were committed.

### Startup I/O correction — 2026-10-11

Initial array 58031292 was cancelled after roughly four minutes without entering
Rosetta: both Python tasks were blocked in BeeGFS metadata I/O while closing
Hydra log/config files under `/pub`. No refinement took place in that attempt.

Replacement array **58031310** (source commit `ac9d7e2`) uses the same two inputs,
protocol and seed; only output staging changes. Array task 0 is numeric job
**58031311** (τ=0.3 sample 5), task 1 is **58031310** (τ=0.8 sample 6).
Both started immediately. Rosetta/Hydra writes use
`$TMPDIR/simplefold-refinement`; a temporary recovery copy is retained at
`artifacts/refinement-stage/<job-id>/` in the remote home checkout before
publication to `/pub/ynkim4/ml-simplefold/artifacts/runs/<job-id>/`.
The launcher supplies `hydra.run.dir` accordingly; model/reference inputs and
canonical long-term storage remain under `/pub`.

Starting independent heavy-atom measurements:

| Source | Total / interchain clashes | Bonds outside 0.1 Å tolerance | Largest bond deviation | FK reward |
|---|---:|---:|---:|---:|
| τ=0.3 sample 5 | 86 / 21 | 122 / 4843 | 0.917 Å | +2.4090 |
| τ=0.8 sample 6 | 166 / 46 | 356 / 4843 | 0.494 Å | +2.8483 |

## Completed pilot: τ=0.3 displayed sample 5

Task **58031311** completed in approximately 9.1 minutes of Rosetta plus
analysis. All 18 output files were downloaded and verified against remote
SHA-256 hashes (see local `download-verification.json`). All 27 final SSBOND
records match the explicit prepared topology. Raw sampling PDB remains intact.

| Measure | Original | Refined |
|---|---:|---:|
| Heavy-atom overlaps counted as clashes | 86 | 48 |
| Interchain clashes | 21 | 5 |
| Bond deviations >0.1 Å | 122 / 4843 | 2 / 4849 |
| Bond-length deviation RMS | 0.0481 Å | 0.0165 Å |
| Maximum bond-length deviation | 0.9169 Å | 0.1313 Å |
| SG–SG range across 27 fixed disulfides | 1.693–2.249 Å | 2.020–2.062 Å |
| Rescored FK reward | +2.4090 | +4.2537 |

Assembly Cα drift is **0.137 Å**; BMP2 dimer drift is 0.130 Å. Receptor
orientation changes are below 0.08°, and centers move under 0.02 Å in the BMP2
frame. The six added heavy atoms are terminal OXT atoms; hydrogens are excluded
from these counts.

### Interface tradeoffs (original coherent assignment held fixed)

| Receptor copy | Site | Native-contact recovery before → after |
|---|---|---:|
| BMPR1A D | observed | 80.4% → 80.4% |
| BMPR1A C | inferred | 84.3% → 76.5% |
| ACVR2B F | observed | 70.0% → 63.3% |
| ACVR2B E | inferred | 66.7% → 53.3% |

W60 and Y42 distance features remain 1.0 for both ACVR2B copies, and L61
geometry remains compatible. However, inferred-side BMPR1A Q86 reorients:
its OE1–BMP2 L51-N distance changes from 2.747 to 5.156 Å. Its inactive
Q86 geometry feature drops from 0.9983 to 0.0010. **The FK reward rises mainly
because the bond-strain penalty is removed, not because every interface
improves.** This is a repaired candidate to inspect, not an automatically
accepted replacement for the raw seed.

Remaining overlap counts use uniform heavy-atom radii and can include close
polar contacts; they are not MolProbity clashscores. The largest remaining
interchain overlaps involve BMP2 Lys101 and the trimmed BMPR1A C-terminal
Asp84 at approximately 2.44 Å. Full stereochemical validation is still separate.

Local comparison:
`artifacts/remote-runs/58031311/comparison.json`.
Refined PDB:
`artifacts/remote-runs/58031311/relaxed/input_0001.pdb`.
Remote canonical directory:
`/pub/ynkim4/ml-simplefold/artifacts/runs/58031311/`.

## Connectivity interruption during the second pilot

Direct laptop HPC3 SSH was working, then timed out. Cisco status reports
**Disconnected** on both the current laptop and documented old-Mac fallback.
No VPN was started, stopped, or reconfigured. Reconnection was requested.

Last confirmed task 58031310 state: running, third of four stages, approximately
15 minutes elapsed. Do not infer its final status until Slurm is reachable again.
Task 58031311 is complete, published under `/pub`, downloaded and verified.

After reconnecting, query array 58031310 with `sacct`/`squeue`, then collect
finished task 58031310 via the dedicated transfer endpoint. The local collector
is `logs/rosetta-refinement/collect_pilot.py` and can be run with:

```bash
/Users/vincentzaballa/anaconda3/envs/simplefold/bin/python \
  logs/rosetta-refinement/collect_pilot.py 58031310
```

It checks completion, downloads the exact run directory, verifies every SHA-256
hash and all 27 final disulfide annotations, and writes a summary under
`artifacts/refinement/2h62/`. Inspect original-assignment per-copy contacts and
Q86 geometry alongside chemistry improvements. No further refinement variant
or BMP2–BMPR1A–ACVR2A job has been launched.
