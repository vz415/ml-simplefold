# 2GOO ensemble sampling and steric quality

Date: 2026-10-06. Requested: ten complete six-chain structures per model,
100M and 3B. Input remains `examples/2goo_hexamer.fasta` (694 residues), two
separate chains per protein. Sampling: 500 steps, tau 0.01, one RNG seed 42
with ten samples in a batch, pLDDT disabled. These are not ten distinct seeds.

## Reference correction

The six deposited asymmetric-unit chains ABCDEF are two independently observed
half-complexes, not one contacting biological hexamer. The earlier whole-assembly
scores against that arrangement are superseded. The original BMP2 A/D chains
have a minimum heavy-atom separation of 29.47 Å.

The current reference uses observed chains ABC and their crystal symmetry mate,
relabelled DEF. Fractional operator `x,x-y,-z+5/6`, lattice shift `(0,1,-1)`.
The rigid operation preserves each protein's geometry and restores the BMP2
homodimer. It is a derived reference, not the identity-only downloadable assembly.
Independent native BMP2 dimer 1REW corroborates it: 0.563 Å Cα RMSD over 206
positions. Reference coverage is 560 observed Cα positions of 694 sequence residues.
Cys78–Cys78 sulfur separation is 2.457 Å; this is retained without minimization.
Original deposited files and BoltzGen endpoints remain untouched. Missing termini,
zero-occupancy atoms and stretched local disulfides remain relevant limitations.

Sources: [2GOO coordinates](https://files.rcsb.org/download/2GOO.cif),
[2GOO deposition annotation](https://pdbj.org/emnavi/quick.php?id=pdb-2goo&lang=en),
[1REW coordinates](https://files.rcsb.org/download/1REW.cif).
Reproduction: `scripts/investigate_2goo_assembly.py`; evidence and derived files:
`artifacts/analysis/2goo_assembly_audit/`.
Use `derived/A_half_symmetry/reference.cif` and `reference.pdb` for analysis.
Native BMP2 has two disulfide-linked chains; concatenation would introduce an
artificial peptide fusion. Current sampling does not enforce dimer geometry.

## Submitted batches

Runtime source at submission: `30cb8bd2041feda387115d71f6d6fb63c0d202e1`.
Remote checkout: `/data/homezvol2/ynkim4/ml-simplefold`, account `ynkim4`.
All model weights and outputs: `/pub/ynkim4/ml-simplefold/artifacts/`.

```bash
sbatch --parsable --gres=gpu:A30:1 --mem=32G --time=00:15:00 --job-name=2goo-100M-10 scripts/hpc_sample.slurm examples/2goo_hexamer.fasta 500 10 0.01 42 simplefold_100M
sbatch --parsable --gres=gpu:A100:1 --mem=96G --time=01:00:00 --job-name=2goo-3B-10 scripts/hpc_sample.slurm examples/2goo_hexamer.fasta 500 10 0.01 42 simplefold_3B
```

- 100M **57874796**: FAILED 1:0 after 2:09, model-storage preflight timed out
  before inference. No ten-sample batch exists from this job.
- 3B **57874803**: FAILED 1:0 after 2:03. Started 2026-10-06 19:57:53
  and ended 19:59:56 (cluster clock). Model-storage preflight timed out after
  120 seconds: `Model storage is unavailable; stopping before inference.`
  No 3B sampling occurred and no 3B predictions were produced.
- Prior queued single-sample 3B job **57873967** was canceled and superseded.
- Existing 100M pilot **57873825** remains usable after the chain-order validator
  correction. The current viewer contains that one pilot, not ten samples.

## Clash counting and current pilot

`scripts/structure_clashes.py` counts occupied protein heavy-atom pairs with
van der Waals overlap greater than 0.4 Å. Hydrogens and zero-occupancy atoms are
excluded; chemical bond graph neighbors through three bonds (1–2, 1–3, 1–4)
are excluded. Explicit disulfides and unique plausible SG pairs define topology.
This is a geometric count, not MolProbity clashscore, energy, or a full
stereochemical validation. Geometry is read from raw predictions without repairs.
A malformed, too-short sulfur pair is counted rather than silently inferred as
an acceptable disulfide. JSON records exclusions, component counts and worst pairs.

| Metric | 100M pilot | Symmetry-derived reference |
| --- | ---: | ---: |
| Total clash pairs | 1,510 | 64 |
| Within-chain pairs | 472 | 58 |
| Between-chain pairs | 1,038 | 6 |
| Occupied heavy atoms | 5,464 | 4,392 |
| Clashes / 1,000 atoms | 276.35 | 14.57 |
| Worst overlap (Å) | 3.274 | 0.920 |
| Nonadjacent Cα clashes | 6 | 0 |
| BMP2 Cys78 sulfur distance (Å) | 1.331 | 2.457 |

Raw totals have different coverage because reference atoms are unresolved;
normalized counts aid comparison. The pilot's too-short dimer sulfur distance
is a separate geometry failure even though the sulfur atoms are near each other.
Against the corrected reference: assembly Cα RMSD 26.259 Å, Cα lDDT 0.6288,
interchain Cα contact recall 0.2390 and precision 0.0802 (8 Å contact cutoff).
These replace earlier assembly scores. Separate-fit mean chain RMSDs are BMP2
1.236 Å, BMPR1A 4.957 Å and ActRIIA 3.556 Å; fold quality and assembly quality
remain distinct.

## Analysis and viewer

```bash
python scripts/analyze_complex_ensemble.py --reference artifacts/analysis/2goo_assembly_audit/derived/A_half_symmetry/reference.cif --reference-pdb artifacts/analysis/2goo_assembly_audit/derived/A_half_symmetry/reference.pdb --prediction-dir 100M=artifacts/remote-runs/57873825/predictions_simplefold_100M --expected-samples 1 --output-dir artifacts/analysis/2goo_models/2026-10-06
python scripts/build_complex_ensemble_viewer.py --analysis-dir artifacts/analysis/2goo_models/2026-10-06
```

For completed batches, supply both downloaded run directories and
`--expected-samples 10`. Each model then has 45 pairwise diversity comparisons.
Global fits allow eight identical-copy permutations; diversity uses a common
560-residue mask. Summary tables report mean, sample SD (n−1), median and range.
The viewer provides overlay, reference-only and predictions-only modes, component
visibility, individual sample selection, clash summaries and per-sample counts.
Local artifact: `artifacts/analysis/2goo_models/2026-10-06/index.html`.
Remote archive updates may be delayed while `/pub` is unresponsive; local analysis
is current. No models or sampling are run locally.

Validation: 24 coordinate-only regression tests passed across clash counting,
ensemble statistics and multichain input; generated viewer JavaScript syntax passed.

### Receptor-end markers

The viewer marks the last available Cα residue of each BMPR1A and ActRIIA
chain with a red sphere. Reference ends are B/E:118 and C/F:99; predicted
full-sequence ends are C/D:131 and E/F:102. Unresolved reference termini mean
these are different sequence positions. The structures contain extracellular
fragments, not the membrane-spanning helices; these are endpoint markers,
not an inferred membrane plane. Marker visibility follows model, sample and
component controls, with a separate toggle. Labels appear on the reference
or individual predictions to limit clutter in ensembles.

### Resubmission after storage reads recovered

Login-host metadata and one-byte reads of the 100M folding checkpoint, shared
ESM encoder and CCD cache succeeded. Both batches were resubmitted at the same
settings from source `862f42a1bf9545f44791db93ed3aa1412fdff89c`:
**57882918** (100M, A30) and **57882919** (3B, A100), ten samples each.
The launcher now reports each checked path, host/account and failing stat exit
code. Login checks do not guarantee compute-node access; the job checks it again.
3B folding weights download within its allocation when storage is accessible.

### Completed 3B ensemble and A100 retry for 100M

**3B 57882919 COMPLETED 0:0**, total job time 34:46, 500-step sampling time
21:23. All ten six-chain, 694-residue predictions passed the launcher validator.
Downloaded ten PDBs and run metadata to `artifacts/remote-runs/57882919/`;
all ten SHA-256 hashes match the remote originals. Full coordinate analysis:
`artifacts/analysis/2goo_3B/57882919/`. The existing combined viewer now shows
the 100M pilot (one sample) and the 3B ensemble (ten samples), explicitly labelled.

| 3B metric | Mean ± sample SD | Range |
| --- | --- | --- |
| Assembly Cα RMSD (Å) | 22.298 ± 4.588 | 10.727–26.515 |
| Assembly Cα lDDT | 0.8200 ± 0.0400 | 0.7774–0.9076 |
| Interchain contact recall | 0.4874 ± 0.1323 | 0.2893–0.7421 |
| Heavy-atom clashes | 852.0 ± 933.0 | 166–2,345 |
| Between-chain clashes | 720.1 ± 924.1 | 59–2,193 |
| BMP2 Cys78 sulfur distance (Å) | 2.098 ± 0.101 | 1.965–2.269 |
| Pairwise ensemble Cα RMSD (Å), 45 pairs | 19.250 ± 6.034 | 2.843–26.598 |

Clash distribution is skewed: median total 381.5, median between-chain 227.
3B has improved coordinate agreement relative to the single 100M pilot, but
many samples retain severe assembly errors. One pilot is not a matched ten-sample
small-model baseline. Native dimer-site sulfur distance is more plausible with 3B;
this does not by itself prove the whole receptor assembly is correct.

**100M 57882918 FAILED 1:0**, elapsed 0:48. Storage passed, then CUDA OOM
inside ESM attention softmax during feature preparation. A30 capacity 23.60 GiB,
only 69.44 MiB free when requesting another 104 MiB. The shared ESM2-3B encoder
still has substantial memory requirements with the smallest folding checkpoint.
No new 100M samples from this job. Retried on A100 with 96G host RAM:
**57888889**, pending (Resources). Submission source
`173dfdba7b009a59e46dd104caf758e852d0d582`, sampling code unchanged from862f42a.

```bash
sbatch --parsable --gres=gpu:A100:1 --mem=96G --time=00:15:00 --job-name=2goo-100M-10 scripts/hpc_sample.slurm examples/2goo_hexamer.fasta 500 10 0.01 42 simplefold_100M
```

Output target `/pub/ynkim4/ml-simplefold/artifacts/runs/57888889/`.
Use A100 for subsequent ten-sample 2GOO batches for either model unless memory
optimizations are separately implemented and verified. No local inference.

### 2026-10-09 status: both requested batches completed

Slurm confirms **100M 57888889 COMPLETED 0:0**. Started 2026-10-06
23:53:48 and ended 23:56:03 on `hpc3-gpu-l54-03`; total job 2:15,
500-step sampling 1:24. All ten six-chain structures passed sequence,
residue coverage and finite-coordinate validation. **3B 57882919** remains
COMPLETED 0:0 with ten samples, sampling 21:23 and total job 34:46.
No sampling jobs remain active and no further batches were submitted.

Collection of the completed 100M ensemble is in progress. Initial /pub transfer
and checksum reads stalled; the current viewer still shows the earlier 100M
pilot until the ten files can be retrieved and analyzed. Do not interpret
its one-sample metrics as this completed ensemble's results.

### 2026-10-09 download retry

Retried the completed 100M run57888889 through the dedicated transfer endpoint
with exact SFTP file paths and temporary `.part` names. The50-second limit
expired while opening `2goo_hexamer_sampled_0.pdb`; zero PDBs downloaded.
No partial file was promoted to a final name. The remote checksum attempt
also failed. A login-node check confirmed `/pub` points to `/dfs6b/pub`;
checking the exact prediction through that physical path timed out with137
(after the timeout's forced kill). SSH authentication and home-storage commands
work. These failures are stalled filesystem reads, not a reported permission
error, and do not invalidate the completed sampling jobs.

Viewer contents remain the old single 100M pilot plus all ten 3B samples.
The latest receptor-only labels and numbered sample choices are retained.
Retry collection when file reads respond; no new sampling job is necessary.

### 2026-10-09 interactive 3B viewer

The viewer now labels the complex **BMP2–BMPR1A–ACVR2A**; PDB 2GOO remains
the experimental provenance. A dedicated ten-sample 3B viewer is available at
`artifacts/analysis/2goo_3B/57882919/index.html`, with all ten aligned predictions
embedded. Its Samples selector offers Sample 1–10 and All samples (10).
Overlay, Reference only, Predictions only, protein visibility, receptor end
dots, and per-sample metrics remain available. Rebuild with:

```bash
python scripts/build_complex_ensemble_viewer.py --analysis-dir artifacts/analysis/2goo_3B/57882919
```

The combined viewer at `artifacts/analysis/2goo_models/2026-10-06/index.html`
also places the 3B ensemble first; its 100M card still contains the old pilot.
No new sampling or remote transfers were needed for this viewer update.

### 2026-10-09 wrist/knuckle and receptor placement analysis

Added native interface diagnostics to the ten downloaded 3B predictions and
the old single 100M pilot. Definitions, numbering correspondence, source papers,
and reproduction commands are in [BMP2 receptor interfaces](bmp2_receptor_interfaces.md).
Both viewers now default to BMP2-dimer alignment, with whole-assembly alignment
still selectable. Individual sample selection shows all four receptor copies,
their contact recovery/precision, placement/orientation, interface RMSD, and
expandable hotspot and BMPR1A α1 details. All-sample mode compares samples and
summarizes their new metrics alongside the established clashes and fold scores.

The fixed reference has 53 wrist residue-pair contacts per BMPR1A copy
(36/17 across the BMP2 monomers, reversed in the other copy), and 23 knuckle
contacts per ACVR2A copy, all on its native BMP2 monomer. These are computed
occupied-heavy-atom ≤4 Å contacts on our reference; do not force the paper's
reported contact-residue totals onto this atom mask. Native α1 has six of seven
residues in the chosen helix-like φ/ψ window; Gly82 has a helix-cap conformation.
Its fraction is a diagnostic relative to 0.857, not a scalar to maximize.

| 3B sample | Wrist contact recovery, copy mean ↑ | Knuckle contact recovery, copy mean ↑ | BMPR1A placement RMSD, copy mean (Å) ↓ | ACVR2A placement RMSD, copy mean (Å) ↓ | Raw clashes ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.358 | 0.348 | 29.38 | 31.72 | 364 |
| 2 | 0.453 | 0.000 | 6.96 | 43.95 | 399 |
| 3 | 0.396 | 0.000 | 20.91 | 39.44 | 2,321 |
| 4 | 0.358 | 0.000 | 22.66 | 51.08 | 166 |
| 5 | 0.396 | 0.283 | 28.63 | 32.16 | 471 |
| 6 | 0.340 | 0.457 | 20.77 | 20.05 | 170 |
| 7 | 0.349 | 0.000 | 21.56 | 39.88 | 1,886 |
| 8 | 0.434 | 0.000 | 20.44 | 48.00 | 216 |
| 9 | 0.396 | 0.000 | 19.12 | 40.54 | 2,345 |
| 10 | 0.764 | 0.152 | 1.41 | 18.74 | 182 |

Useful partial successes support the user's qualitative observation that some
predictions work well at particular interfaces:

- Sample 10: both wrist copies have 1.35–1.48 Å placement RMSD and 72–81%
  contact recovery. Both knuckle copies are misplaced by 18.68–18.81 Å and
  recover 13–17%. Its 0.63 Å C₂ deviation shows why near symmetry alone
  cannot establish native receptor topology.
- Sample 6: one ACVR2A copy has 1.20 Å placement RMSD, 0.50 Å interface
  backbone RMSD, and 91.3% contact recovery; its counterpart is displaced
  by 38.89 Å with zero recovery. Component means conceal this asymmetry.

Across 3B samples, wrist contact recovery is 0.4245 ± 0.1248 and knuckle
recovery 0.1239 ± 0.1760 (sample SD). BMP2-dimer fitted RMSD is
1.131 ± 0.493 Å. The one 100M pilot has wrist recovery 0.0283 and knuckle
recovery zero; this remains an unmatched one-sample comparison. Geometry
agreement does not demonstrate functional binding, and favorable contact
scores must be inspected with clashes.

Validation: 33 coordinate-only tests cover rigid transforms, equivalent-copy
assignment, contact masking and missing atoms, displaced but correctly folded
receptors, incorrectly docked symmetric assemblies, hotspot correspondence,
and dihedral conventions. Programmatic viewer checks use the actual 3Dmol
GLModel API and a DOM harness to exercise sample/reference/alignment, protein,
and red-dot toggles for both the dedicated 3B and combined viewers. This is not
a WebGL screenshot check. Independent numerical review on real Sample 10
confirmed invariance to all eight copy renamings and arbitrary rigid transforms.

### Sample 1 interpretation: wrong receptor identity at overlapping sites

The user's Sample 1 screenshot appeared to show favorable receptor overlays
despite zero contact scores for BMPR1A reference B → prediction C and ACVR2A
reference C → prediction F. These labels compare chains in different structures;
they are not receptor–ligand chain pairs. A contact score of zero means zero
recovered native residue pairs, not zero physical predicted contacts.

| Reference → prediction | Native contacts recovered | Precision: recovered / predicted | Placement RMSD (Å) |
| --- | ---: | ---: | ---: |
| BMPR1A B → C | 0/53 | 0/69 | 55.82 |
| BMPR1A E → D | 38/53 | 38/46 | 2.93 |
| ACVR2A C → F | 0/23 | 0/21 | 58.48 |
| ACVR2A F → E | 16/23 | 16/22 | 4.95 |

In the BMP2 frame, prediction BMPR1A C is 8.58 Å by matched-residue centroid
from reference ACVR2A C, and its ligand-contact residues overlap 12 of the
14 BMP2 residues in that knuckle epitope. Prediction ACVR2A F is 4.37 Å by
centroid from reference BMPR1A B, with contacts to 10 of 23 BMP2 residues in
that wrist epitope. This supports a wrong-type receptor occupying each of those
sites, rather than successful native docking of those copies. The other pair
has partial native interface recovery. Shape/projection overlap alone is
insufficient; toggling each protein type separately reveals the distinction.
The viewer now shows recovered/native and recovered/predicted counts beside
the percentages, and explicitly explains cross-structure chain labels.

### Sample selection and ordering audit

The user reported that the lowest-error Sample 10 appeared to be displayed as
Sample 1. No mismatch was reproduced in the generated artifacts: every source
SHA-256 matches, every ligand-aligned Cα equals its source transformed by the
saved BMP2 fit within PDB rounding, and independently recalculated displayed
receptor placement scores match the JSON. Sample 1 is `sampled_0` (global
assembly RMSD 26.52 Å); Sample 10 is `sampled_9` (10.73 Å, lowest in this batch).
An in-app browser connection failed, so the user's older tab state was not
verified. The prior Sample 1 interpretation applies to the actual `sampled_0`
coordinates, not an independently confirmed identity of the screenshot.

The selector now includes assembly RMSD beside each sample label. Buttons in
the **Sample metrics and structure files** table select the same sample and
highlight its row; clicking from reference-only mode returns to overlay.
Programmatic checks of the actual GLModel hidden flags and all displayed Cα
coordinates passed for every sample in both ligand and global alignments,
including the combined viewer's one-sample 100M card. Table RMSD, selected
status, and highlighted row agree with each chosen structure.
