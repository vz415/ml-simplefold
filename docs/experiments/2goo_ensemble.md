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
- 3B **57874803**: PENDING (Resources), latest status check.
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
