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
# Keep Rosetta's frequent writes off the shared filesystem. The home copy is
# temporary recovery storage; /pub remains the canonical output destination.
work_dir="${TMPDIR:?Slurm must provide node-local TMPDIR}/simplefold-refinement"
stage_dir="${SIMPLEFOLD_REFINEMENT_STAGE_DIR:-$PWD/artifacts/refinement-stage}/${SLURM_JOB_ID}"
publish_dir="${SIMPLEFOLD_ARTIFACT_DIR:-/pub/ynkim4/ml-simplefold/artifacts}/runs/${SLURM_JOB_ID}"
status=0
if [[ "${SIMPLEFOLD_REFINEMENT_MODE:-reference}" == reference-free ]]; then
    python scripts/run_reference_free_refinement.py "hydra.run.dir=$work_dir" "$@" || status=$?
else
    python scripts/run_rosetta_refinement.py "hydra.run.dir=$work_dir" "$@" || status=$?
fi
mkdir -p "$stage_dir"
cp -a "$work_dir/." "$stage_dir/"
printf 'Recovery copy: %s\n' "$stage_dir"
mkdir -p "$publish_dir"
cp -a "$stage_dir/." "$publish_dir/"
printf 'Published refinement: %s\n' "$publish_dir"
exit "$status"
