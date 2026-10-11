#!/usr/bin/env bash
#SBATCH --job-name=2h62-rosetta-refine
#SBATCH --partition=standard
#SBATCH --account=eehui_lab
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/refine-%A_%a.out
#SBATCH --error=logs/refine-%A_%a.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit refinement through Slurm}"
cd "${SLURM_SUBMIT_DIR:?}"
module load rosetta/3.13
module load miniconda3/24.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SIMPLEFOLD_ENV_PREFIX:-/data/homezvol2/ynkim4/.conda/envs/simplefold}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec python scripts/run_rosetta_refinement.py "$@"
