# SimpleFold on the laptop and HPC3y

The local `simplefold` conda environment is for editing, imports, and result
analysis. Sampling runs only on HPC3y (`ynkim4`) through Slurm.

Remote model weights, trained checkpoints, synthetic datasets, and run outputs
use `/pub/ynkim4/ml-simplefold/artifacts`, the default `SIMPLEFOLD_ARTIFACT_DIR`.
`SIMPLEFOLD_CACHE_DIR` defaults to the same root for checkpoint, Torch, and CCD
caches; set it explicitly to override those caches. Installation/pip caches and
environment locks stay in the home checkout's `artifacts/pip-cache/` and
`artifacts/environment/`. The checkout remains at
`/data/homezvol2/ynkim4/ml-simplefold`, and the conda prefix remains at
`/data/homezvol2/ynkim4/.conda/envs/simplefold`. Use explicit storage paths
without home-to-DFS artifact symlinks. Local downloaded results still go under
the laptop checkout's `artifacts/remote-runs/`.

Within the `/pub` artifact root, use this layout for current and future work:

| Directory | Content |
| --- | --- |
| `checkpoints/` | Pretrained folding weights, including `simplefold_100M.ckpt` |
| `checkpoints/trained/` | Future trained checkpoints |
| `torch/` | Torch/ESM model cache (`TORCH_HOME`) |
| `ccd/` | CCD dictionary cache |
| `datasets/synthetic/` | Future synthetic datasets |
| `runs/JOB_ID/` | PDBs, run outputs, validation, and package-version receipts |

The training and dataset paths establish storage conventions for future active
sampling and retraining work; this setup currently runs inference only.

## Install locally

```bash
bash scripts/setup_environment.sh local
conda activate simplefold
simplefold --help
```

Python 3.10, PyTorch 2.6.0, torchvision 0.21.0, NumPy 1.26.4, and the upstream
dependencies are installed. ESM is installed as `fair-esm==2.0.0`; loading it
does not require fetching executable source through GitHub Torch Hub. Local
installation does not download model weights or run inference.

Select the `Python (SimpleFold)` kernel for notebooks. Use
[`notebooks/inspect_remote.ipynb`](../notebooks/inspect_remote.ipynb) locally to
summarize and view downloaded PDBs. The upstream `sample.ipynb` now requires a
Slurm GPU allocation before any model construction or downloads.

## Set up HPC3y

```bash
ssh hpc3y
cd /data/homezvol2/ynkim4
git clone https://github.com/vz415/ml-simplefold.git
cd ml-simplefold
git switch experiments/hpc3y-setup
mkdir -p logs
sbatch scripts/hpc_setup.slurm
```

Setup runs on a CPU compute node in the `free` partition. It creates
`/data/homezvol2/ynkim4/.conda/envs/simplefold`, installs CUDA 12.4 PyTorch wheels, checks
imports and dependencies, and downloads the 100M folding checkpoint, ESM2-3B
weights, and CCD dictionary. Interrupted downloads resume from `.part` files.
Checkpoint, Torch, and CCD caches live under the `/pub` artifact root in
`checkpoints/`, `torch/`, and `ccd/`. Installation/pip caches remain in the home
checkout's `artifacts/pip-cache/`. Resolved package versions remain in
`/data/homezvol2/ynkim4/ml-simplefold/artifacts/environment/pip-hpc.txt`, with
package-version receipts also saved inside individual run directories.
`TORCH_HOME` points to the cache root's `torch/` directory. The remote-only
sampling notebook uses the same checkpoint cache and writes outputs under
`/pub/ynkim4/ml-simplefold/artifacts/runs/SLURM_JOB_ID/notebook/`.

## Sample a toy protein

After setup has completed successfully:

```bash
cd /data/homezvol2/ynkim4/ml-simplefold
sbatch scripts/hpc_sample.slurm examples/crambin.fasta 500 1
squeue -u ynkim4
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList
```

The default job requests one GPU, four CPU cores, 32 GiB host memory, and
15 minutes on `gpu`, charged to `eehui_lab_gpu`. Even the 100M folding model
uses an ESM2-3B encoder. CPU sampling is refused if CUDA is unavailable.
The arguments are FASTA file/directory, step count, and samples per protein.
Use one protein sequence per FASTA file. For multiple targets, pass a directory
containing separate single-record FASTA files; multichain input is not validated
by this setup.
The supplied crambin sequence has 46 residues (PDB 1CRN). Default sampling is
500 steps, tau 0.01, seed 42, and one PDB output. pLDDT is disabled because it
requires an additional 1.6B model; its absence means the PDB B-factors are
placeholders, not confidence scores.

Results go to
`/pub/ynkim4/ml-simplefold/artifacts/runs/JOB_ID/predictions_simplefold_100M/`.
Each run records its Git commit, installed packages, and `validation.json`, which
checks output count, sequence, CA atoms, and finite coordinates. This verifies
the inference pipeline; it is not a folding-accuracy benchmark.

To run on free GPU resources, override scheduling explicitly:

```bash
sbatch --partition=free-gpu --account=eehui_lab --qos=low scripts/hpc_sample.slurm
```

Free jobs can be preempted. For queued setup, use a dependency:

```bash
setup_id=$(sbatch --parsable scripts/hpc_setup.slurm)
sbatch --dependency=afterok:$setup_id scripts/hpc_sample.slurm
```

For a quick pipeline check while GPUs are queued, `scripts/hpc_cpu_smoke.slurm`
runs 20 steps on a **remote Slurm CPU node**, with 16 cores and 48 GiB host
memory. It forces CUDA off and uses the same cache and PDB validator. This is
a diagnostic run; use the GPU launcher and 500 steps for the folding example.

```bash
sbatch scripts/hpc_cpu_smoke.slurm
```

## Retrieve outputs

Use the dedicated RCIC transfer host; keep Git/Slurm commands on the login host.

```bash
mkdir -p artifacts/remote-runs/JOB_ID/predictions_simplefold_100M
scp -o HostName=access-hpc3.rcic.uci.edu \
  hpc3y:/pub/ynkim4/ml-simplefold/artifacts/runs/JOB_ID/predictions_simplefold_100M/crambin_sampled_0.pdb \
  artifacts/remote-runs/JOB_ID/predictions_simplefold_100M/
scp -o HostName=access-hpc3.rcic.uci.edu \
  hpc3y:/pub/ynkim4/ml-simplefold/artifacts/runs/JOB_ID/validation.json \
  artifacts/remote-runs/JOB_ID/
```

Inspect the result and accompanying validation report locally. Do not launch
`simplefold` inference on the laptop. Git changes go to the `experiments`
remote at `https://github.com/vz415/ml-simplefold`; `origin` remains upstream.

The root `hpc_setup.slurm` and `hpc_sample.slurm` symlinks support the shared
HPC submission helper, which expects launcher filenames at repository root.
The commands above use `sbatch` directly with the files under `scripts/`.
