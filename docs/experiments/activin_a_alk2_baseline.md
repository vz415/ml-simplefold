# Activin A–ALK2–ACVR2B first baseline

One target, ten sequence-only SimpleFold-3B samples, 500 steps, tau 0.3, seed 82101. Use A100 through Slurm; no local inference. Input is a 2:2:2 protein-only assembly (676 residues). No reward is used in this first baseline. No noise sweep is authorized by this launch.

ACVR2B was chosen because it has several existing project examples and an experimental activin A interface (1S4Y). Activin A–ALK4–ACVR2B in 7OLY supplies related assembly context, but ALK4 is not ALK2 and the crystal used a receptor-trap/Fab strategy. 7U5P supplies an alternative activin A–ACVR2A partial complex for later comparison.

Input provenance: examples/activin-a-acvr1-acvr2b.provenance.json. Activin A uses the 116-residue mature INHBA sequence deposited in 7OLY; ALK2 (103 residues) and ACVR2B (119 residues) reuse the project's prior ECD constructs. No Fc, antibody, membrane, or intracellular kinase is included; glycans are not modeled.

WT ALK2 can form activin A/type-II non-signaling complexes. Future guidance should treat that binding hypothesis separately from downstream signaling. The 2025 ALK2/BMP6 study describes activin fingertip recognition, so transferring the BMP6 wrist/glycan guidance unchanged is not justified. Develop an activin-specific type-I score after inspecting the baseline and residue evidence. Experimental type-II contacts may anchor that module; ALK4-derived ALK2 geometry remains a transfer hypothesis.

References:
- https://www.rcsb.org/structure/1S4Y
- https://www.rcsb.org/structure/7OLY
- https://www.rcsb.org/structure/7U5P
- https://elifesciences.org/articles/54582 (Aykul et al., 2020)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12415261/ (Goebel et al., 2025)

Launch: sbatch --parsable --gres=gpu:A100:1 --mem=96G --time=01:30:00 --job-name=activinA-alk2-actr2b-baseline scripts/hpc_sample.slurm experiment=activin-a-acvr1-acvr2b

Outputs and receipts: /pub/ynkim4/ml-simplefold/artifacts/runs/JOB_ID. On completion collect PDBs and receipts to local artifacts/remote-runs/JOB_ID and build the standard viewer. No automatic FK, refinement, or follow-up batches are implied by this baseline.
