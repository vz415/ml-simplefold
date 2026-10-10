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
Use iteration-specific subdirectories under `checkpoints/trained/` and
`datasets/synthetic/`. The pretrained checkpoint remains at
`checkpoints/simplefold_100M.ckpt`, so later training can always start from it.

The training profile `experiment=hpc3y_train` loads those pretrained weights
into the folding model and its EMA copy with fresh training state. It directs
logs and samples to `runs/SLURM_JOB_ID/`, new checkpoints to
`checkpoints/trained/SLURM_JOB_ID/`, and ESM loading to the `/pub` Torch cache.
Select a prepared synthetic-data configuration with `data=YOUR_DATA_CONFIG`
when launching training through Slurm; the inherited upstream PDB data paths
are placeholders for this workflow. The profile requires `SLURM_JOB_ID`.
Use `pretrained_folding_ckpt_path=null` and `+load_ckpt_path=CHECKPOINT` to
resume a full Lightning training checkpoint instead of warm-starting.

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
sbatch scripts/hpc_sample.slurm fasta=examples/crambin.fasta num_steps=500 samples=1
squeue -u ynkim4
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList
```

The default job requests one GPU, four CPU cores, 32 GiB host memory, and
15 minutes on `gpu`, charged to `eehui_lab_gpu`. Even the 100M folding model
uses an ESM2-3B encoder. CPU sampling is refused if CUDA is unavailable.
The launcher passes named Hydra overrides to `scripts/run_sampling.py`.
Settings live in `configs/sample.yaml`; override `fasta`, `num_steps`, `samples`,
`tau`, `seed`, and `model` as needed. Tau defaults to 0.01 and seed to 42.
For example, ten samples at tau 0.3 use:

```bash
sbatch scripts/hpc_sample.slurm fasta=examples/7wf9_a.fasta num_steps=500 samples=10 tau=0.3 seed=42
```

For multiple targets, pass a directory containing separate FASTA files.
A multichain complex uses one FASTA file containing its chain records;
validation checks their sequences and the saved chain order.
The supplied crambin sequence has 46 residues (PDB 1CRN). Default sampling is
500 steps, tau 0.01, seed 42, and one PDB output. pLDDT is disabled because it
requires an additional 1.6B model; its absence means the PDB B-factors are
placeholders, not confidence scores.

Results go to
`/pub/ynkim4/ml-simplefold/artifacts/runs/JOB_ID/predictions_simplefold_100M/`.
Hydra saves `.hydra/config.yaml` (composed task settings), `.hydra/hydra.yaml`
(Hydra runtime settings), and `.hydra/overrides.yaml` (explicit overrides) in
the run directory. `resolved-config.yaml` additionally saves the task settings
with environment variables and interpolations resolved. Each run also records
its Git commit, installed packages, sampling parameters in `experiment.json`,
and `validation.json`, which
checks output count, sequence, CA atoms, and finite coordinates. This verifies
the inference pipeline; it is not a folding-accuracy benchmark.

Hydra tracks executable configuration; the existing metadata and Obsidian
experiment receipts retain provenance, results, and research decisions.
The storage callback checks `/pub` with a timeout before Hydra writes its
receipts, and rejects an output directory containing an existing run receipt.
Override `paths.cache_dir` to move all pretrained/CCD caches together;
individual checkpoint/CCD directory overrides are rejected to keep download
and inference destinations consistent.
Inspect settings locally without creating a sampling run or loading weights:

```bash
python scripts/run_sampling.py experiment=2goo_fk --cfg job --resolve
```

Hydra's basic multirun can vary scalar settings inside an existing Slurm
allocation, for example `-m seed=42,43`; each run gets its own numbered
subdirectory and configuration receipts. It does not submit additional Slurm
jobs. Storage paths and experiment/sampling-mode selection must remain shared
within a sweep, using the default numbered subdirectories.
Sweep storage/output paths can reference shared `paths.*` settings and
environment variables, but cannot depend on task parameters such as `seed`.
Steering parameters can also be swept, for example
`experiment=2goo_fk -m sampling.beta=1.0,2.0`.

The named `experiment=2goo_3b` profile selects ordinary ten-sample SimpleFold-3B
inference on the BMP2–BMPR1A–ACVR2A complex. The `experiment=2goo_fk` profile
selects the matched baseline/FK pilot described in
[`bmp2_fk_pilot.md`](experiments/bmp2_fk_pilot.md). Resource requests remain
`sbatch` flags, separate from Hydra sampling settings:

```bash
sbatch --gres=gpu:A100:1 --mem=96G --time=01:00:00 scripts/hpc_sample.slurm experiment=2goo_3b
```

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
