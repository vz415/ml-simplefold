#!/usr/bin/env bash
#SBATCH --job-name=simplefold-sample
#SBATCH --partition=gpu
#SBATCH --account=eehui_lab_gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/sample-%j.out
#SBATCH --error=logs/sample-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load miniconda3/24.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SIMPLEFOLD_ENV_PREFIX:-/data/homezvol2/ynkim4/.conda/envs/simplefold}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
exec python scripts/run_sampling.py "$@"
