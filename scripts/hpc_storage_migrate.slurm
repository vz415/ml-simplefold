#!/usr/bin/env bash
#SBATCH --job-name=simplefold-storage
#SBATCH --partition=free
#SBATCH --account=eehui_lab
#SBATCH --qos=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/storage-%j.out
#SBATCH --error=logs/storage-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
git rev-parse HEAD
export PATH="${SIMPLEFOLD_ENV_PREFIX:-/data/homezvol2/ynkim4/.conda/envs/simplefold}/bin:$PATH"
python \
    scripts/migrate_artifacts.py --source "$PWD/artifacts" \
    --destination "${SIMPLEFOLD_ARTIFACT_DIR:-/pub/ynkim4/ml-simplefold/artifacts}"
