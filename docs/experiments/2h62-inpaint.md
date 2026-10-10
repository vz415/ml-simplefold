# 2h62-inpaint: BMP2–BMPR1A–ACVR2B baseline

Requested 2026-10-10. Start with ordinary, sequence-only SimpleFold-3B sampling
of a **2:2:2 six-chain target**, before developing partial-structure constraints
or FK hotspot steering. `2h62-inpaint` is the project label; this first baseline
does not supply template coordinates, fix observed atoms, or implement inpainting.

## Experimental starting structure and target

[PDB 2H62](https://www.rcsb.org/structure/2H62) is a wild-type BMP2 complex with
one BMPR1A and one ACVR2B extracellular domain, determined at 1.85 Å. Its
deposited assembly contains four chains (ligand:type-I:type-II **2:1:1**).
The [primary paper](https://link.springer.com/article/10.1186/1472-6807-7-6)
reports a hexamer in solution and attributes the missing receptor copies in
the crystal to packing: crystal-neighbor proteins block the empty epitopes.

| Input / prepared chain | Component | Native author chain | Full sequence length | Observed Cα positions | Coordinate provenance |
|---|---|---|---:|---:|---|
| A | BMP2 | A | 114 | 104 | Experimental |
| B | BMPR1A / ALK3 | C | 129 | 84 | Experimental |
| C | ACVR2B / ActRIIB | D | 98 | 93 | Experimental |
| D | BMP2 | B | 114 | 103 | Experimental |
| E | BMPR1A / ALK3 | C, second copy | 129 | — | Missing receptor; sequence duplicated for inference |
| F | ACVR2B / ActRIIB | D, second copy | 98 | — | Missing receptor; sequence duplicated for inference |

Input [FASTA](../../examples/2h62-inpaint.fasta) has **682 residues**. Full
deposited canonical sequences include unresolved termini. This uses ACVR2B,
not the ACVR2A receptor in 2GOO, and BMPR1A lacks 2GOO's initial GS tag.
Sequence/provenance mapping is in
[2h62-inpaint.provenance.json](../../examples/2h62-inpaint.provenance.json).
The parser groups identical sequences, so saved prediction chains A/B are
BMP2, C/D BMPR1A, and E/F ACVR2B. Keep input/prepared and saved chain labels
distinct in analysis.

Local raw coordinates are preserved under
`artifacts/structures/2h62-inpaint/2H62.cif` and `2H62-assembly1.cif`.
Prepared artifacts live in that directory's `prepared/` subdirectory:

- `observed.cif` / `observed.pdb`: four experimental protein chains remapped
  as above, retaining observed coordinates and sequence numbering.
- `known_regions.json`: experimentally occupied heavy-atom/residue mappings
  for future constraints; missing receptor copies E/F are excluded.
- `symmetry_completion.cif` / `.pdb`: a **provisional modeled six-chain**
  geometry, formed by fitting the ligand dimer to its copy-swapped counterpart
  and transforming receptor copies. This is a placement hypothesis, not a
  deposited six-chain biological assembly or experimental ground truth.

The dimer copy-swap fit has Cα RMSD **1.625 Å**, despite an exact 180° fitted
rotation. The naive extra receptor copies produce severe overlaps (some atom
separations below 1 Å). Do not use their coordinates or contact maps as native
targets without a separately justified modeling/validation step. The observed
four-chain reference is the primary baseline comparison. No atom repairs or
energy minimization are applied during preparation.

## Baseline sampling

The [Hydra profile](../../configs/experiment/2h62-inpaint.yaml) selects the
established 3B model, **10 samples, 500 steps, tau 0.01, seed 42**, PyTorch and
PDB output. This is one initial baseline, not a tau sweep or FK run. Model
weights remain frozen; pLDDT is disabled, so PDB B-factors are not confidence.

```bash
sbatch --job-name=2h62-inpaint-baseline --gres=gpu:A100:1 --mem=96G \
  --time=01:00:00 scripts/hpc_sample.slurm experiment=2h62-inpaint
```

Run on `hpc3y` from `/data/homezvol2/ynkim4/ml-simplefold`, using the established
conda prefix. Checkpoints and output stay under
`/pub/ynkim4/ml-simplefold/artifacts`. The existing launcher is in the project's
approved script list. Hydra settings, source commit, package versions, and
sequence/finite-coordinate validation are recorded in `runs/JOB_ID/`.
Expected PDB names are
`predictions_simplefold_3B/2h62-inpaint_sampled_0.pdb` through `_9.pdb`.

Local sampling is prohibited. Preparation, configuration inspection, and
analysis of downloaded coordinates are permitted locally.

## Comparison and later steering

For the baseline, align the BMP2 dimer with equivalent copies allowed to swap.
Compare one matched BMPR1A and ACVR2B prediction against the observed receptors,
and explicitly distinguish the additional unobserved receptor copies. Their
native-coordinate error is unavailable from 2H62. Report clashes over all six
predicted chains and preservation/recovery of the observed interfaces; do not
report six-chain native recovery against the naive symmetry completion.

Future hotspot rewards should be receptor-specific. The source paper identifies
ACVR2B Tyr42 (rather than ACVR2A Phe42), Trp60, and surrounding interface
residues; map identities and numbering from this construct before choosing
reward terms. BMPR1A F85/Q86 are candidates to inspect against the actual
observed contacts, rather than assuming the 2GOO contact list transfers intact.

The same study provides [2H64](https://www.rcsb.org/structure/2H64), a complete
assembly with BMP2 **L100K/N102D**. This is a potentially useful external
comparison, with a mutated ligand; it is not wild-type native ground truth
for all interfaces. No 2H64 comparison or guided sampling has been started.

Related: [ligand panel and completion proposal](bmp_partial_complex_completion.md).

## Baseline submission: 2026-10-10

A100 job **57998266**, submitted from source
`dc0946221b5123c48ee5604805467774bed81928`, uses the command above. Initial
status is `PENDING (Resources)`; no prediction is available yet.
Outputs: `/pub/ynkim4/ml-simplefold/artifacts/runs/57998266/`.
Prepared/raw references are archived separately in
`/pub/ynkim4/ml-simplefold/artifacts/datasets/references/2h62-inpaint/`.
Local coordinate comparison is available through
[analyze_2h62_baseline.py](../../scripts/analyze_2h62_baseline.py).

## Hotspot clues and proposed FK reward

These are experimental clues and a proposed objective, not an implemented
2H62 reward or an FK submission.

| Feature | Evidence / local target |
|---|---|
| ACVR2B W60 | W60A binding is below detection in the paper's assay. W60 packs against BMP2 A34/P35/S88/L90/L100. |
| ACVR2B Y42 | Y42A raises apparent Kd 61-fold for BMP2 in the paper's assay. Reward native-like engagement rather than an arbitrary short distance. |
| BMPR1A Q86 | Q86A has approximately 100-fold weaker BMP2 affinity. Its side chain interacts with the Leu51 backbone. |

Sources: [Weber et al. 2007, including Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC1802081/)
and [Kotzsch et al. 2009, discussing BMPR1A Q86](https://pmc.ncbi.nlm.nih.gov/articles/PMC2670865/).
The W60A assay result is censored, not a measured finite fold change. These
affinity changes motivate priorities; they do not determine numerical reward
weights. Weber's F83A variant has a possible folding defect, so do not transfer
the previous ACVR2A F83 reward blindly.

Direct sequence/coordinate audit: paper numbering equals the full-input
one-based positions for BMP2 A34/P35/L51/D53/S88/L90/L100, BMPR1A F85/Q86,
and ACVR2B Y42/W60/L61. Saved prediction chain groups are documented above.
In the actual crystal, native BMPR1A chain C Q86 OE1 is **2.848 Å** from
BMP2 chain B L51 N; Q86 NE2 is **2.974 Å** from L51 O. These are heavy-atom
geometry targets; hydrogen-bond claims additionally require compatible
donor/acceptor orientation. D53 may contribute elsewhere but is not a direct
Q86 hydrogen-bond partner in this coordinate audit.

Proposed first objective, with each local feature normalized to [0,1]:

`R = 2 min(W60 packing across copies) + min(Y42 engagement across copies)`
`    + 2 min(Q86 geometry across copies) - structural penalties`

The 2:1:2 positive weights are an interpretable initial hypothesis, not fitted
binding energies. Positive terms have maximum 5 before penalties. Keep them
configurable and inspect the separate contributions before tuning beta.

- **Local geometry:** use smooth, bounded distance/orientation agreement with
  observed 2H62 interactions, not unlimited contact counts or shorter-is-better
  attraction. For W60, distribute credit across the named pocket residues;
  for Q86, require both complementary backbone partners with plausible geometry.
- **Both copies and topology:** take the worse copy for each feature, with a
  one-to-one assignment of identical receptors to two distinct wrist sites
  and two distinct knuckle sites. The second site's motif is inferred by
  exchanging ligand identities, not by enforcing the clashing copied receptor
  coordinates. Two receptors at one site must not receive two successful scores.
- **Known structure:** preserve the observed subcomplex using a declared
  coordinate restraint or fixed-region mechanism. Penalize its deformation,
  ligand-dimer distortion, severe nonbonded clashes, and strained covalent
  geometry. Keep native disulfides separate from nonbonded overlap counts.
- **Scope:** retain whole observed-interface recovery and precision as reported
  diagnostics initially, rather than hiding them inside a larger objective.
  The wild-type S88–L61 hydrogen bond is not an energetic hotspot; do not elevate
  it to a leading reward merely because it is visible. Mutant 2H64 is an
  external comparison, not the unknown wild-type completion's ground truth.
- **FK mechanics:** a starting choice is the established late clean-estimate
  checkpoints (t=0.60/0.75/0.90/0.97), with terminal scoring and fixed weights.
  The same potential-difference/ESS selection machinery can be reused after
  implementing the new reward and observed-region policy.

**User instruction:** future FK submissions must run FK only and reuse this
baseline; do not regenerate a baseline inside every job. The current
`sample_complex_fk.py` always loops over baseline and FK, so a skip-baseline/FK-only
execution path is required before submitting this experiment. No such change
has been made yet. Reusing an ordinary baseline avoids extra inference, but
does not imply exact paired RNG/initial-latent matching; preserve that distinction
in any comparison.
