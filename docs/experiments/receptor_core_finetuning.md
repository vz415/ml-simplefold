# Mixed fine-tuning with shortened receptor targets

User authorized training on the six shortened refined structures on 2026-10-20.
Use `configs/mixed_train_receptor_core.yaml`, starting from the released 100M
checkpoint, frozen ESM2, 600 updates and the previous 90% PDB / 10% synthetic
mixture. Reuse the exact seed-82301 draw plan (540 PDB, 60 synthetic), learning
rate, EMA, loss settings, three reconstruction samples per complex and the
24-target CAMEO retention evaluation. Do not resume the earlier tuned model.

The six targets retain BMPR1A 36–108 (73 aa), ACVR2B 6–98 (93 aa), ACVR2A
8–99 (92 aa) and ACVR1 11–89 (79 aa), in full modeled-construct numbering.
The latter two boundaries were checked against 2GOO and 9N4K. All receptor
cysteines, mapped disulfide endpoints and reference ligand-contact residues
remain. Retained coordinates and complete ligand chains are unchanged.
Terminal chemistry has not been rebuilt or relaxed after deletion.

Dataset: `/pub/ynkim4/ml-simplefold/artifacts/datasets/synthetic/multimer-receptor-core-v1`.
Selection: `configs/datasets/multimer_receptor_core_v1.json`. First prepare and
featurize it on a CPU allocation using the existing approved
`scripts/hpc_sample.slurm --train-multimers prepare_only=true` entry point.
Then run `scripts/hpc_sample.slurm --train-mixed
--config-name=mixed_train_receptor_core` on an A100, dependent on preparation.

General data remain in the immutable existing `pdb-mixed-600-seed82301` cache.
`general_replay_selection` verifies its original preparation receipt, while
training verifies the new synthetic PDB hashes. The selection IDs and chain
groups must agree exactly with the prior schedule. Record both selection
hashes, the unmodified training-plan hash and the converted target receipt.
No bulk general-data download is needed. Checkpoints and outputs remain under
the standard `/pub` checkpoint/run directories.

Reconstruction compares both pretrained and new EMA samples against the
cropped synthetic targets. Those sequence lengths differ from the previous
run, so complex score changes are not a like-for-like whole-structure benchmark.
Use CAMEO retention and geometric diagnostics alongside interface scores.
