# BMP6 restrained Rosetta refinement

Vincent requested relaxation before the next scoring iteration on 2026-10-14.
Compare the v3/tau 0.8 sample 1 from job 58134298 with the previous v1/tau 0.1
sample 9 from job 58107636. Sample indices here are zero-based. Preserve both
raw sampling results; refinement is a separate derived artifact, not a new
sampling recipe or evidence that a docking arrangement is correct.

## Protocol and provenance

`scripts/run_bmp6_refinement.py` runs only under Slurm unless `--prepare-only`.
`scripts/hpc_refine_bmp6.slurm` requests one CPU, 8 GB and one hour on
`standard` / `eehui_lab`; one independent job per candidate. The resolved JSON
input bundle, input hashes, code revision, executable hash, commands, Rosetta
log, raw output, canonical output and comparison are copied into the run receipt.
Node-local working directories isolate Rosetta checkpoints and frequent writes;
`/pub/ynkim4/ml-simplefold/artifacts/runs/<job_id>/` is the canonical result path.

Use Rosetta 3.13, `ref2015_cart`, a single four-stage Cartesian FastRelax pass,
repulsive weights 0.02, 0.25, 0.55 and 1.0, final minimizer tolerance 0.00001,
and fixed seed 73. Repack side chains while preserving sequences. Disable jump
minimization and design. Backbone atoms of the 552 reference-observed residues
receive harmonic coordinate restraints with SD 0.5 Å and weight 1 throughout;
the 164 unresolved terminal residues remain unrestrained. Targets are each
candidate's own starting coordinates, never experimental coordinates. Soft
restraints and Cartesian movement can still change interfaces; measure drift.

Retain all 716 residues. Explicitly declare 27 native disulfides: 26 annotated
intrachain bonds from 9N4K and the BMP6 interchain C477–C477 bond annotated by
UniProt P22004, corresponding to input A103–B103. The latter is absent from the
CIF disulfide list and its reference SG distance is 4.088 Å: its native topology
comes from the annotation, not from treating that coordinate geometry as a bond.
Disable automatic disulfide detection. Preparation preserves original ATOM
records and validates sequence, atom completeness, masks and bonded endpoints.

The ignored frozen input bundle is
`artifacts/bmp90/preparation/bmp6_rosetta_20261014/input_bundle/`.
Its topology JSON includes an agent's optional staged design; this runner uses
only the verified pairs and observed-core masks, not that unexecuted suggestion.
The executed protocol is the XML, relax script and configuration in each run.

## Paired evaluation

`scripts/analyze_bmp6_refinement.py` holds the original coherent assignment fixed
for both structures, both guidance recipes and all four official DockQ scores.
It compares native-contact recovery, N60 CA residual after each receptor fit,
N60 local geometry, severe interchain C/N/O overlaps greater than 1 Å, ordinary
heavy-atom overlaps, backbone bonds, 27 disulfide distances, full-construct and
ligand-frame drift. It requires complete standard heavy atoms; terminal OXT is
optional and counted separately. No alternate chain assignment can hide drift.

Prefer a refinement only after checking physical improvement alongside every
interface and N60. Preserve the unrepaired original if docking agreement worsens.
These geometric checks are not MolProbity or calibrated acceptance. DockQ is
recovery of the observed 9N4K receptor-trap interfaces; the reference is engineered
and glycosylated whereas these free protein chains omit the tether and glycans.

Rosetta implementation references:
[FastRelax](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/FastRelaxMover),
[AtomCoordinateCstMover](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/AtomCoordinateCstMover),
[VirtualRoot](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/VirtualRootMover).
