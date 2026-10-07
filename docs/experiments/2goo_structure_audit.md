# 2GOO extracellular complex: reference quality notes

Date: 2026-10-06. Decision: document the issues; no structure repair requested or performed.

2GOO is the experimental BMP2–BMPR1A/ALK3–ActRIIA extracellular hexamer, with two copies of each component. BMP2 and BMPR1A are human; ActRIIA/ACVR2A is mouse. This is not an ALK2 complex. The X-ray resolution is 2.20 Å. See the [RCSB entry](https://www.rcsb.org/structure/2GOO) and [official wwPDB validation report](https://files.rcsb.org/validation/view/2goo_full_validation.pdf) (report dated 2026-03-08).

## Files examined

- Raw endpoint: `/Users/vincentzaballa/Development/boltzgen/research/bmp_signaling/structures/endpoints/bmp2_bmpr1a_actrii_ecd_complex_2goo_raw.pdb`
- Existing protein-only endpoint: `/Users/vincentzaballa/Development/boltzgen/research/bmp_signaling/structures/normalized/bmp2_bmpr1a_actrii_ecd_2goo/protein_standard/bmp2_bmpr1a_actrii_ecd_2goo_full_hexamer_protein_standard.pdb`
- Reproducible local coordinate audit: `artifacts/analysis/2goo_audit/2026-10-06/audit.py` and `audit.json` in this repository. These generated artifacts are ignored by git.
- Raw PDB SHA256: `ba7f2330a9d076179d1deb85218de21a6b8558c752d943b452905f732d0d3bbf`.

## Independent coordinate checks

The file contains one model, six protein chains, 4,475 protein atoms, four sugar residues (60 atoms), and 327 waters. There are no duplicate protein atom identities or alternate conformations. All observed residues have N/CA/C/O backbone atoms, and the observed chain segments have no internal residue-numbering gaps or large peptide-bond breaks.

| Chain | Component | Deposited residues | Observed residues | Observed author-number range |
| --- | --- | ---: | ---: | --- |
| A | BMP2 | 114 | 103 | 12–114 |
| D | BMP2 | 114 | 104 | 11–114 |
| B | BMPR1A/ALK3 | 131 | 85 | 34–118 |
| E | BMPR1A/ALK3 | 131 | 88 | 33–120 |
| C | ActRIIA/ACVR2A | 102 | 92 | 8–99 |
| F | ActRIIA/ACVR2A | 102 | 93 | 7–99 |

- **Missing coordinates:** 129 of the 694 deposited protein residues are absent, all at chain termini. The 565 modeled residues are not a complete full-sequence reference. The two copies of each component have different resolved coverage.
- **Zero occupancy:** 42 side-chain atoms across 12 residues have occupancy 0, all in ActRIIA chains C/F. Affected residues in each chain are ARG20, ARG22, LYS35, LYS37, ILE73, and LYS94. Presence in the PDB does not establish experimental support for these atom positions.
- **Chirality:** the PDB CAVEAT explicitly flags ILE F:73 Cβ and sugar NDG C:402 C1 as having wrong chirality; the official report corroborates this.
- **Disulfide geometry:** the listed BMPR1A B:CYS40–CYS44 disulfide has an SG–SG distance of 2.952 Å. The other listed disulfides measure approximately 2.03–2.08 Å. This is a local geometry concern, not evidence that the entire assembly is invalid.
- **Existing normalization is not repair:** all 4,475 protein atom identities, coordinates, occupancies, and B-factors are exactly retained in the protein-only endpoint. Removing sugars/waters eliminates the sugar chirality issue but preserves the protein issues above.

## Official validation findings

The wwPDB report lists 121/565 modeled residues as electron-density-fit outliers (RSRZ > 2; approximately 21%). ActRIIA has the highest fractions: 35/92 in chain C (38%) and 29/93 in F (31%). These indicate weak local agreement with the experimental density; they do not prove every flagged residue is incorrect.

The report also lists three Ramachandran outliers (C:34 ASP, C:36 ASP, F:36 ASP), 28 side-chain rotamer outliers, eight bond-length outliers, and 18 bond-angle outliers. Its all-atom clashscore is 8, calculated with added hydrogens; this was not independently recomputed. It reports no chain breaks.

## Implications for experiments

The structure remains useful as an experimental assembly and fold reference, with variable local reliability. For Cα comparisons, align sequences and score only observed matched residues; report coverage and consider a separate density-quality mask. Do not treat absent terminal coordinates as experimental ground truth.

For future all-atom energy calculations or action-minimization endpoints, revisit the zero-occupancy side chains, F:73 stereochemistry, and B:40–44 disulfide before interpreting local energies. Any repair should be a separate derived structure with recorded provenance. No repair or sampling was performed for this audit; the original files remain unchanged.
