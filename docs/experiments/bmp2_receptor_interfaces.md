# BMP2–BMPR1A–ACVR2A interface analysis

## Biological targets and correspondence

The reference is the existing crystal-symmetry reconstruction of the native
2:2:2, C₂-symmetric complex. Its BMP2 chains are A/D, BMPR1A B/E, and ACVR2A
C/F. Prediction chains are BMP2 A/B, BMPR1A C/D, and ACVR2A E/F. BMP2 and
BMPR1A are human; this deposited ACVR2A construct is mouse. Original coordinates
and predictions remain unmodified.

BMPR1A recognizes the **wrist epitope**, a composite BMP2-dimer binding site.
ACVR2A recognizes the **knuckle epitope**, primarily on one BMP2 monomer, with
its concave β-sheet face. β-sheet-mediated docking does not imply formation
of a shared intermolecular β-sheet.

| Feature | Mature/PDB author numbering | Prediction sequence ordinal |
| --- | --- | --- |
| BMPR1A α1 in the β4–β5 loop | Gly82–Lys88 | 84–90 |
| BMPR1A binding hot spots | Phe85, Gln86 | 87, 88 |
| ACVR2A aromatic cluster | Phe42, Trp60, Phe83 | 42, 60, 83 |

The BMPR1A sequence includes an N-terminal GS tag. Mapping uses the mmCIF
full-sequence positions and observed author residue IDs, with residue-identity
checks; missing residues or atoms remain explicit. These mature-protein labels
are not full-length UniProt numbering.

Sources: [Allendorph et al., 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC1456805/),
[Harth et al., 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0013049),
[2GOO record](https://www.rcsb.org/structure/2GOO).

## Coordinate diagnostics

`scripts/complex_interfaces.py` prepares reference interfaces once, then scores
raw predictions without model loading or sampling. The ensemble analyzer
includes its results in `metrics.json`; detailed definitions and coverage
are stored with the reference metadata and each receptor copy.

- **BMP2-dimer fit:** fit the observed ligand Cα positions only, allowing its
  identical chains to exchange. Assign equivalent receptor copies in that fixed
  ligand frame. This assignment is separate from the original global fit.
- **Receptor placement:** Cα RMSD without refitting the receptor, centroid
  displacement, and orientation error. A separate receptor fit is used only
  to measure the rotation error, not to erase placement errors.
- **Native-contact recovery and precision:** receptor–BMP2 residue-pair contacts
  at occupied heavy-atom distance ≤4 Å, using matched reference atom identities.
  Contact counts are also broken down by BMP2 monomer. The BMPR1A partner is
  the whole dimer, preserving its composite wrist interface.
- **Interface backbone RMSD:** independently fit the matched backbone atoms
  of the native contacting residues on both partners. This is a local interface
  diagnostic alongside the fixed-frame placement score.
- **Hotspot diagnostics:** recover reference contacts involving the named
  receptor residues and report their nearest ligand heavy-atom distances.
  Close approach can be a clash; it is not automatically favorable packing.
- **BMPR1A α1 geometry:** separately fitted backbone RMSD for the native α1
  segment and a backbone φ/ψ helix-like fraction. The latter is a dihedral
  diagnostic, not a DSSP secondary-structure assignment.
- **C₂ consistency:** compare corresponding copies under the reference's
  twofold operator in the ligand frame. A wrong but symmetric arrangement
  can score well here; assess native placement and contacts at the same time.

The established global assembly, separate-chain fold, dimer sulfur-distance,
ensemble diversity, and steric-clash metrics are retained. New contact scores
use a different definition from the original <8 Å Cα contact scores. The
interface metrics are not DockQ, binding affinity, or evidence of signaling.
Buried surface area and intermolecular β-sheet assignment are not calculated.

## Viewer and reproduction

The HTML offers **BMP2 dimer** and **Whole assembly** alignment. Select an
individual sample to inspect all four receptor copies and expand hotspot
details; select all samples for ensemble summaries and a comparison table.
The reference-only mode shows the reference interface definitions.

```bash
python scripts/analyze_complex_ensemble.py --reference artifacts/analysis/2goo_assembly_audit/derived/A_half_symmetry/reference.cif --reference-pdb artifacts/analysis/2goo_assembly_audit/derived/A_half_symmetry/reference.pdb --prediction-dir 3B=artifacts/remote-runs/57882919/predictions_simplefold_3B --expected-samples 10 --output-dir artifacts/analysis/2goo_3B/57882919
python scripts/build_complex_ensemble_viewer.py --analysis-dir artifacts/analysis/2goo_3B/57882919
```

Local analysis uses the ten downloaded 3B predictions and the older single
100M pilot. The completed 100M ten-sample run remains uncollected while `/pub`
reads stall; it is not silently substituted by the pilot.
