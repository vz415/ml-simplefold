#!/usr/bin/env bash
#SBATCH --job-name=bmp6-rosetta
#SBATCH --partition=standard
#SBATCH --account=eehui_lab
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bmp6-refine-%j.out
#SBATCH --error=logs/bmp6-refine-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
cd "${SLURM_SUBMIT_DIR:?}"
module load rosetta/3.13
module load miniconda3/24.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SIMPLEFOLD_ENV_PREFIX:-/data/homezvol2/ynkim4/.conda/envs/simplefold}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
work_dir="${TMPDIR:?Slurm must provide node-local TMPDIR}/bmp6-refinement-${SLURM_JOB_ID}"
stage_dir="$PWD/artifacts/refinement-stage/${SLURM_JOB_ID}"
publish_dir="${SIMPLEFOLD_ARTIFACT_DIR:-/pub/ynkim4/ml-simplefold/artifacts}/runs/${SLURM_JOB_ID}"
status=0
python scripts/run_bmp6_refinement.py --output "$work_dir" "$@" || status=$?
if [[ -d "$work_dir" ]]; then
    mkdir -p "$stage_dir"
    cp -a "$work_dir/." "$stage_dir/"
    printf 'Recovery copy: %s\n' "$stage_dir"
    mkdir -p "$publish_dir"
    cp -a "$stage_dir/." "$publish_dir/"
    printf 'Published BMP6 refinement: %s\n' "$publish_dir"
fi
exit "$status"
