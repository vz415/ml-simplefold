#!/usr/bin/env bash
#SBATCH --job-name=simplefold-sample
#SBATCH --partition=gpu
#SBATCH --account=eehui_lab_gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
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
if [[ "${1:-}" == --geometry-setup ]]; then
    shift
    exec python scripts/setup_geometry_validation.py "$@"
fi
if [[ "${1:-}" == --train-mixed ]]; then
    shift
    module load apptainer/1.4.5
    exec python scripts/train_mixed.py "$@"
fi
if [[ "${1:-}" == --benchmark-cameo ]]; then
    shift
    module load apptainer/1.4.5
    exec python scripts/run_cameo_benchmark.py "$@"
fi
# The approved project launcher also supports the explicitly requested training pilot.
if [[ "${1:-}" == --train-multimers ]]; then
    shift
    export PIP_CACHE_DIR="$SLURM_SUBMIT_DIR/artifacts/pip-cache"
    python -m pip install --no-deps fairscale==0.4.13
    exec python scripts/train_multimers.py "$@"
fi
exec python scripts/run_sampling.py "$@"
