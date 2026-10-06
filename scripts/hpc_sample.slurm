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
artifact_dir="${SIMPLEFOLD_ARTIFACT_DIR:-/pub/ynkim4/ml-simplefold/artifacts}"
export TORCH_HOME="$artifact_dir/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
fasta_path="${1:-examples/crambin.fasta}"
num_steps="${2:-500}"
num_samples="${3:-1}"
output_dir="$artifact_dir/runs/${SLURM_JOB_ID}"
mkdir -p "$output_dir"
git rev-parse HEAD > "$output_dir/git-commit.txt"
python -m pip freeze > "$output_dir/pip-freeze.txt"
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; refusing CPU sampling"; print(torch.__version__, torch.cuda.get_device_name(0))'
simplefold --backend torch --simplefold_model simplefold_100M \
    --ckpt_dir "$artifact_dir/checkpoints" --cache_dir "$artifact_dir/ccd" \
    --fasta_path "$fasta_path" --output_dir "$output_dir" \
    --num_steps "$num_steps" --tau 0.01 --nsample_per_protein "$num_samples" \
    --seed 42 --output_format pdb
python scripts/validate_predictions.py --fasta_path "$fasta_path" \
    --prediction_dir "$output_dir/predictions_simplefold_100M" --num_samples "$num_samples"
echo "Sampling completed on $(hostname) at $(date -Is)"
