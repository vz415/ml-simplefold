#!/usr/bin/env bash
#SBATCH --job-name=simplefold-setup
#SBATCH --partition=free
#SBATCH --account=eehui_lab
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load miniconda3/24.9.2
bash scripts/setup_environment.sh hpc
/data/homezvol2/ynkim4/.conda/envs/simplefold/bin/python scripts/download_weights.py
echo "Setup completed on $(hostname) at $(date -Is)"
