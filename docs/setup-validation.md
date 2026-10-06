# Setup verification — 2026-10-05

SimpleFold is installed locally and on HPC3y. The complete 500-step toy
inference pipeline passed on a remote NVIDIA A30. No local models were loaded
and no local sampling was performed.

## Environments

| Location | Environment | Verified behavior |
| --- | --- | --- |
| Laptop | `conda activate simplefold` | Dependency check, CLI imports/help, config resources, ESM registry, CCD cache behavior, notebook kernel |
| HPC3y (`ynkim4`) | `/data/homezvol2/ynkim4/.conda/envs/simplefold` | CUDA PyTorch installation, cached model loading, GPU and CPU Slurm inference, PDB export |

Both use Python 3.10, PyTorch 2.6.0, torchvision 0.21.0, NumPy 1.26.4,
fair-esm 2.0.0, and upstream pinned dependencies. Remote PyTorch uses CUDA
12.4 wheels. Package locks are stored in ignored `artifacts/environment/` and
inside each run. The laptop's Jupyter kernel is `Python (SimpleFold)`.

Remote checkout: `/data/homezvol2/ynkim4/ml-simplefold`, branch
`experiments/hpc3y-setup`, fork `https://github.com/vz415/ml-simplefold`.
Home storage was used after BeeGFS metadata writes stalled. About 29 GiB
remained free after setup; larger model/data downloads need a space check.

## Slurm results

| Job | Purpose | Compute node | Elapsed | Result |
| --- | --- | --- | --- | --- |
| `57819969` | Install environment and download caches | `hpc3-14-13` | 4m 10s | Completed, exit `0:0` |
| `57820072` | GPU crambin inference, 500 steps | `hpc3-gpu-24-06` (A30) | 50s | Completed, exit `0:0`; PDB validated |
| `57820090` | Remote CPU diagnostic, 20 steps | `hpc3-18-00` | 59s | Completed, exit `0:0`; PDB validated |

The first A100-only sampling request (`57819973`) was cancelled while pending.
The replacement GPU job allowed any GPU in the `gpu` partition, four CPUs,
32 GiB host memory, and 15 minutes. It ran with approximately 22.6 GiB peak
host RSS. Its sampling loop took about 7 seconds; the 50-second elapsed time
also includes imports, model loading, features, export, and validation.

The GPU run used execution commit `acfd94379b4d593a877cb5403d0dcbb0c7a4de49`;
the CPU diagnostic used `d442498`. Both use SimpleFold-100M plus ESM2-3B,
tau 0.01, seed 42, and one sample. pLDDT was disabled, so the B-factors are
placeholders. The 20-step diagnostic checks execution, not folding quality.

## Artifacts and local verification

Remote GPU output:

```text
/data/homezvol2/ynkim4/ml-simplefold/artifacts/runs/57820072/
  predictions_simplefold_100M/crambin_sampled_0.pdb
  validation.json
  git-commit.txt
  pip-freeze.txt
```

Downloaded copy: `artifacts/remote-runs/57820072/`. The PDB and original
validation report were retrieved through `access-hpc3.rcic.uci.edu`; SHA-256
checksums match the remote files. The PDB contains the expected 46 residues,
326 atoms, one CA per residue, and finite coordinates. Local revalidation
also passed. These checks verify pipeline execution, not folding accuracy.

```text
PDB SHA-256:
828b123e31022f4ed2cb64aff87448aeeea83511f0f4574d9776425375baf8b9
```

Every cell of `notebooks/inspect_remote.ipynb` executed successfully against
the downloaded GPU result using the local `simplefold` kernel. The executed
copy is in `artifacts/environment/inspect_remote.executed.ipynb`. The sampling
notebook's Slurm/CUDA guard was checked and blocks local execution before
model loading. No weights, generated PDBs, or environment files are tracked
in Git.

Receipts and outcomes are saved in the Obsidian project
`Computational Research/SimpleFold/experiments/`. Use [the setup runbook](hpc3y.md)
for installation, sampling, and retrieval commands.

## Storage update — 2026-10-06

The current artifact root is `/pub/ynkim4/ml-simplefold/artifacts`. Migration
job `57836989` completed on `hpc3-14-01` in 1m18s, exit `0:0`, using execution
commit `d8a46e01714b43bd25010f9b492b3908a297a903`. It verified and moved
`checkpoints/`, `torch/`, `ccd/`, and `runs/` (18 files, 6,410,825,895 bytes),
reusing destination copies from the cancelled earlier attempt. The original
GPU PDB SHA-256 above is unchanged. Component receipts are
`COMPONENT/migration-57836989.json` under the artifact root.

Pretrained SimpleFold weights remain in `checkpoints/simplefold_100M.ckpt`.
Future trained checkpoints use `checkpoints/trained/ITERATION/`, synthetic
datasets use `datasets/synthetic/ITERATION/`, and sampling results use
`runs/JOB_ID/`. Launchers and the remote notebook use the `/pub` root by
default. Installation caches, package locks, the checkout, and conda prefix
remain in home storage. The superseded home paths above describe the
original 2026-10-05 execution.

The post-migration GPU run `57837138` loaded both folding and ESM weights
from `/pub` and completed 500-step crambin sampling on an NVIDIA A30
(`hpc3-gpu-l54-02`) in 47s, exit `0:0`. It produced 46 residues / 326 atoms
with matching sequence and finite coordinates. The PDB SHA-256 matches the
original run above. Output is
`/pub/ynkim4/ml-simplefold/artifacts/runs/57837138/predictions_simplefold_100M/crambin_sampled_0.pdb`.

Training profile `experiment=hpc3y_train` configures `/pub` logs, samples,
new checkpoint directories, and pretrained warm-start loading. The loader
initializes both current and EMA architecture weights and uses a fresh
optimizer rather than treating raw inference weights as a full Lightning
resume checkpoint. Three tiny-layer loader tests and Hydra path composition
checks pass. End-to-end training awaits a prepared synthetic dataset config.
