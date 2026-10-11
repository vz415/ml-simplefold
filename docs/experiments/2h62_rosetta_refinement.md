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
