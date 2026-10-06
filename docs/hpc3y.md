# SimpleFold on the laptop and HPC3y

The local `simplefold` conda environment is for editing, imports, and result
analysis. Sampling runs only on HPC3y (`ynkim4`) through Slurm.

The initial setup uses the home volume because BeeGFS `/pub/ynkim4` metadata
writes stalled on 2026-10-05. The home volume had 41 GiB available before
installation. Keep an eye on `df -h ~` before downloading additional models;
the ESM2-3B checkpoint alone is approximately 11 GiB. Large experiment data
should move to project storage once its performance has recovered.

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
Caches live in `artifacts/checkpoints`, `artifacts/torch`, and `artifacts/ccd`.
Resolved package versions are saved in `artifacts/environment/pip-hpc.txt`.

## Sample a toy protein

After setup has completed successfully:

```bash
cd /data/homezvol2/ynkim4/ml-simplefold
sbatch scripts/hpc_sample.slurm examples/crambin.fasta 500 1
squeue -u ynkim4
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList
```

The default job requests one A100, eight CPU cores, 64 GiB host memory, and
45 minutes on `gpu`, charged to `eehui_lab_gpu`. Even the 100M folding model
uses an ESM2-3B encoder. CPU sampling is refused if CUDA is unavailable.
The arguments are FASTA file/directory, step count, and samples per protein.
The supplied crambin sequence has 46 residues (PDB 1CRN). Default sampling is
500 steps, tau 0.01, seed 42, and one PDB output. pLDDT is disabled because it
requires an additional 1.6B model; its absence means the PDB B-factors are
placeholders, not confidence scores.

Results go to `artifacts/runs/JOB_ID/predictions_simplefold_100M/`. Each run
records its Git commit, installed packages, and `validation.json`, which
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

## Retrieve outputs

Use the dedicated RCIC transfer host; keep Git/Slurm commands on the login host.

```bash
scp -o HostName=access-hpc3.rcic.uci.edu \
  hpc3y:/data/homezvol2/ynkim4/ml-simplefold/artifacts/runs/JOB_ID/predictions_simplefold_100M/crambin_sampled_0.pdb \
  artifacts/
```

Inspect the result and accompanying validation report locally. Do not launch
`simplefold` inference on the laptop. Git changes go to the `experiments`
remote at `https://github.com/vz415/ml-simplefold`; `origin` remains upstream.
