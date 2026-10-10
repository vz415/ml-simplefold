# BMP partial-complex completion: ligand panel and experimental references

Recorded 2026-10-10. This is a research direction and reference catalog; no
completion implementation or sampling run has been started.

## User's ligand panel and affinity data

The user studies **BMP4, BMP7, BMP9, BMP10, and GDF5** and has associated
**binding-affinity data** that may be useful for subsequent modeling, steering,
or experimental design. The numerical values, files, assay definitions, and
uncertainties have not been supplied. Integrating these measurements is
deferred until later.

The supplied background describes a possible 30-combination grid: these five
ligands crossed with type-I receptors ACVR1/ALK2 or BMPR1A/ALK3, and type-II
receptors ACVR2A, ACVR2B, or BMPR2. Treat that receptor grid as provisional
until matched to the actual affinity dataset. **ACVRL1/ALK1 is distinct from
ACVR1/ALK2.**

When affinity data are introduced, retain ligand/receptor species, constructs,
mutations, glycosylation, assay conditions, units, replicates, and uncertainty.
Distinguish binding Kd from functional EC50 and multivalent avidity. An affinity
measurement can constrain a hypothesis without uniquely determining docking
coordinates. No affinity values are inferred in this note.

## Experimental reference catalog

Checked against RCSB entry and biological-assembly metadata on 2026-10-10.
Stoichiometry below refers to the deposited biological assembly, rather than
assuming the asymmetric-unit coordinate file is the complete assembly.
Receptor structures here are extracellular constructs. “Complete” means the
expected six protein chains are represented, not that every atom/residue is
resolved or that these are full-length membrane receptors.

### Complete ligand:type-I:type-II 2:2:2 assemblies

| Reference | Ligand | Type-I receptor | Type-II receptor | Method / resolution | Qualification and use |
|---|---|---|---|---|---|
| [2GOO](https://www.rcsb.org/structure/2GOO) | BMP2 | BMPR1A / ALK3 | ACVR2A / ActRIIA | X-ray, 2.20 Å | Current C₂-symmetric six-chain reference. Two letter O's in the identifier. BMP2 is outside the user's ligand panel but useful as a control. |
| [4FAO](https://www.rcsb.org/structure/4FAO) | BMP9 / GDF2 | ACVRL1 / ALK1 | ACVR2B / ActRIIB | X-ray, 3.36 Å | Complete assembly for a panel ligand, but ALK1 differs from both proposed type-I choices. |
| [7PPC](https://www.rcsb.org/structure/7PPC) | BMP10 | ACVRL1 / ALK1 | BMPR2 | X-ray, 3.60 Å | Complete assembly; again ALK1 rather than ALK2 or ALK3. |
| [9N4K](https://www.rcsb.org/structure/9N4K) | BMP6 | ACVR1 / ALK2 | ACVR2B / ActRIIB | Cryo-EM, 3.20 Å | Released September 2025; engineered receptor-trap experiment. Useful ALK2 interface comparator, outside the ligand panel. |
| [9MIR](https://www.rcsb.org/structure/9MIR) | BMP6 | BMPR1A / ALK3 | ACVR2B / ActRIIB | Cryo-EM, 3.30 Å | Released September 2025; same study/construct strategy, enabling ALK2 versus ALK3 comparison. |
| [3KFD](https://www.rcsb.org/structure/3KFD) | TGF-β1 | TGFBR1 / ALK5 | TGFBR2 | X-ray, 3.00 Å | Related-family control, outside the panel. Author assemblies 1/2 are hexamers; a separate PISA assembly is larger. Select explicitly. |
| [2PJY](https://www.rcsb.org/structure/2PJY) | TGF-β3 | TGFBR1 / ALK5 | TGFBR2 | X-ray, 3.00 Å | Related-family control. RCSB lists a mutation in **each receptor construct**, not just type I. Do not silently treat either as wild type. |

The BMP6 constructs were made as receptor-ECD/Fc heterodimeric traps. Fc was
removed for cryo-EM, leaving the receptor ECDs joined through antibody-hinge
disulfides; the flexible linker was not confidently modeled. The study finds
glycan-dependent BMP6 recognition by ALK2. Protein-only predictions therefore
omit experimentally relevant chemistry, and trap geometry must not be treated
as evidence from untethered receptors without qualification.
[Goebel et al., PNAS 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12415261/)

### Partial assemblies / local-interface references

| Reference | Components | Method / resolution | Deposited assembly 1 | What it can validate / missing information |
|---|---|---|---|---|
| [2H62](https://www.rcsb.org/structure/2H62) | BMP2–BMPR1A–ACVR2B | X-ray, 1.85 Å | Ligand:type-I:type-II **2:1:1** | Both receptor interface types, but not the complete six-chain assembly. |
| [1LX5](https://www.rcsb.org/structure/1LX5) | BMP7–ACVR2A | X-ray, 3.30 Å | Ligand:type-II **2:2** | BMP7/type-II interface; no type-I receptor. Biological assembly uses crystallographic symmetry. |
| [7PPA](https://www.rcsb.org/structure/7PPA) | BMP10–BMPR2 | X-ray, 1.48 Å | Ligand:type-II **2:2** | BMP10/type-II interface; no type-I receptor. Pair with 7PPC for a completion benchmark after checking construct correspondence. |
| [3QB4](https://www.rcsb.org/structure/3QB4) | GDF5 **R57A**–BMPR1A | X-ray, 2.28 Å | Ligand:type-I **1:1** | Local type-I interface for a mutant, not a wild-type GDF5/full-assembly reference. |

This is a confirmed reference set, **not an exhaustive search for absence**.
BMP4 being absent from this table does not establish absence of experimental
structures. Likewise, homologous-ligand structures and ALK1 complexes are
useful contextual evidence, not exact experimental matches to the provisional
30 combinations.

RCSB metadata snapshot (ignored local artifact):
`artifacts/research/bmp_complex_completion/2026-10-10/reference-metadata.json`.
Before using coordinates, audit the biological-assembly transformations,
species, chain/entity identities, construct sequences, mutations, observed
residue coverage, numbering, glycans, and disulfides. No new coordinate files
were downloaded for this catalog.

## Proposed SimpleFold + guided-sampling direction

The user's idea is to complete experimentally partial complexes with
SimpleFold and FK steering or other sampling constraints. Existing favorable
2GOO results motivate testing this approach; they do not yet establish accuracy
for unobserved interfaces.

1. Define the task: missing receptor copies/types versus missing residues in
   an otherwise present chain. Choose sequences, ectodomain boundaries, and
   target stoichiometry explicitly; these are different completion problems.
2. Preserve the experimentally observed subcomplex with a mapped coordinate
   mask and/or restraints. Decide whether coordinates are fixed or allowed
   limited movement according to experimental uncertainty. Sample the missing
   components while keeping the measured interface intact.
3. Build rewards from available evidence: observed contact/coordinate
   preservation, ligand-dimer integrity, appropriate wrist/knuckle placement,
   receptor topology, plausible symmetry where supported, and chemical geometry.
   Contact/hotspot rules transferred from BMP2 need ligand/receptor-specific
   evidence and residue mapping; do not assume identical contacts for all pairs.
4. Keep **observed** and **inferred** regions separate in the viewer, scoring,
   and saved provenance. Native-contact recovery for an unknown interface is
   unavailable; missing native data must not be converted to a zero score or
   invented ground truth.
5. Test on complete experimental assemblies with selected receptors withheld.
   Score the withheld structure only after sampling, without exposing its
   coordinates/contacts to the steering reward. Compare ordinary sampling,
   constraints alone, and FK; use independent initial populations and inspect
   diversity, receptor placement, contacts, clashes, and bond geometry.
6. Apply the separately proposed refinement workflow later, keeping raw outputs
   and equivalent baseline controls. Confidence in an inferred interface must
   be calibrated against withheld-reference performance and ideally independent
   evidence; agreement with our chosen reward or resampled siblings is not a
   confidence estimate.

These steps are a proposed design, not an implemented completion algorithm.
The current [FK sampler](../../src/simplefold/model/torch/fk_sampler.py) can
resample particles using a supplied reward, but has no implemented
observed-coordinate inpainting/fixed-region path. The wrapper's
`atom_pad_mask` identifies valid atoms, not experimentally fixed atoms.
The current [complex reward](../../scripts/fk_complex_reward.py) expects the
specific six-chain, 694-residue BMP2–BMPR1A–ACVR2A reference and uses its known
native contacts and placement. It cannot be reused unchanged for unknown
parts of other ligand complexes. Both conditioning and reward generalization
need an explicit design before this becomes a completion experiment.

Related work: [BMP2 FK sweep, limitations, and deferred refinement](bmp2_fk_tau_sweep.md).

## 2026-10-10: first target selected

The user selected **2H62 (BMP2–BMPR1A–ACVR2B)**. A100 baseline job **57998266**
is submitted, initially pending resources, for 10 ordinary SimpleFold-3B
samples of the 682-residue 2:2:2 target. See the
[2h62-inpaint experiment](2h62-inpaint.md) for experimental/modeled chain
provenance, the receipt, and the proposed hotspot reward. Future FK jobs should
reuse this baseline and run FK only. Guided completion remains unimplemented.
