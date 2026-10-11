# 2H62 BMPR1A terminal-trim control

Date: 2026-10-11. Target: BMP2–BMPR1A–ACVR2B.

The user observed a predicted BMPR1A terminal extension clashing with
ACVR2B and requested a separate reference-length experiment with the previous
reward. Keep the concurrent cooperative-score sweep at the original length.

## Construct and score

The observed 2H62 BMPR1A chain spans full-polymer residues **34–117**,
84 residues with no internal gaps. The original input polymer has 129 residues.
Remove **33 N-terminal and 12 C-terminal unresolved residues** from each
BMPR1A copy. Leave BMP2 and ACVR2B unchanged: total complex length changes
from **682 to 592 residues**.

Input FASTA BMPR1A labels are B/E; output PDB BMPR1A chains are C/D.
Use `examples/2h62-inpaint-bmpr1a-trimmed.fasta`. The retained sequence is
the original BMPR1A slice `[33:117]`; this is terminal trimming, not deletion
of internal unresolved residues. The receptor C terminus is membrane-proximal.
Reference-length constructs are a modeling choice, not evidence that unresolved
residues are biologically absent.

Use the previous **L61** objective, with W60/Y42 packing, BMPR1A contact
recovery/precision, the active L61 penalty, and unchanged structural penalty
coefficients. Preserve original native residue numbering: local F52/Q53 in
the shortened prediction score against original BMPR1A F85/Q86. Native
contact sets, observed coordinates, and contact denominators stay fixed.

## Implementation and verification

- `scripts/fk_2h62_reward.py`: optional component-level
  `prediction_sequence_spans`; exact sequence-slice validation and original
  atom-position mapping; observed Cα placement uses the mapped indices.
- `configs/experiment/2h62-inpaint-fk-l61-bmpr1a-trimmed.yaml`.
- `examples/2h62-inpaint-bmpr1a-trimmed.fasta`.
- `tests/test_fk_2h62_trimmed_reward.py`.

Four trim tests and nine original L61 tests passed. Eight cooperative tests
also passed after the additive mapping change. Tests cover preserving internal
unresolved residues, native contact/hotspot/placement equivalence, original
position mapping, incompatible sequence rejection, and disabling incompatible
full-length-baseline rescoring.

A real-coordinate smoke check cropped saved particle index **6** (displayed
**sample 7**) from run **58015740**. All contact/hotspot/L61/dimer/receptor
placement features were exactly unchanged. Total clashes decreased **169→132**,
interchain clashes **62→47**, and clashes per 1,000 heavy atoms **31.401→28.037**.
This is analysis of an existing prediction, not new sampling. Evidence:
`logs/2h62-inpaint-fk/bmpr1a-trim-control-real-smoke.json` and `.pdb`.

## Authorized run

- A100, one GPU, four CPUs, 96 GB host memory, 75-minute wall limit.
- SimpleFold 3B, pretrained weights fixed, ten particles, 500 steps.
- Tau **0.1**, beta 2, ESS threshold 0.8, common start, seeds 42/43/44.
- Checkpoints **[0.30, 0.60, 0.75, 0.90, 0.97]**.
- No new baseline generation; `baseline_prediction_dir=null` avoids attempting
  to score incompatible full-length predictions during launch.
- Compare with the previous full-length five-checkpoint tau=0.1 L61 run
  **58015740**. Shorter chains change conditioning and random-coordinate layout;
  identical numeric seeds do not imply identical paired initial structures.

```bash
sbatch --parsable --job-name=2h62-l61-trim-tau0.1 --gres=gpu:A100:1 \
  --mem=96G --time=01:15:00 scripts/hpc_sample.slurm \
  experiment=2h62-inpaint-fk-l61-bmpr1a-trimmed
```

Launcher: `python scripts/run_sampling.py
experiment=2h62-inpaint-fk-l61-bmpr1a-trimmed`.
Remote output: `/pub/ynkim4/ml-simplefold/artifacts/runs/JOB_ID/`.
Local download: `artifacts/remote-runs/JOB_ID/`.
Submission manifest: `artifacts/sweeps/2h62_bmpr1a_trim/2026-10-11/manifest.json`.

## Follow-through

Collect completed outputs/receipts and verify transfers. Evaluate interface
placement and contacts in the same reference residue frame, and count clashes
on the actual retained atoms. The current persistent page selects the original
full-length FASTA hash: this construct requires explicit sequence-span-aware
analysis and registration as a construct variant before inclusion. Do not
silently equate shortened PDB local numbering with original residue numbering.
