#!/usr/bin/env bash
#SBATCH --job-name=simplefold-cpu-smoke
#SBATCH --partition=standard
#SBATCH --account=eehui_lab
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=logs/cpu-smoke-%j.out
#SBATCH --error=logs/cpu-smoke-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Run this remote CPU smoke test through Slurm}"
cd "${SLURM_SUBMIT_DIR:?}"
module load miniconda3/24.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SIMPLEFOLD_ENV_PREFIX:-/data/homezvol2/ynkim4/.conda/envs/simplefold}"
export CUDA_VISIBLE_DEVICES=''
export TORCH_HOME="$PWD/artifacts/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
fasta_path="examples/crambin.fasta"
output_dir="$PWD/artifacts/runs/${SLURM_JOB_ID}"
mkdir -p "$output_dir"
git rev-parse HEAD > "$output_dir/git-commit.txt"
python -m pip freeze > "$output_dir/pip-freeze.txt"
python -c 'import os, torch; assert os.environ.get("SLURM_JOB_ID"), "Slurm is required"; assert not torch.cuda.is_available(), "CPU smoke test requires CUDA to be disabled"; print(torch.__version__, "remote CPU smoke test", flush=True)'
simplefold --backend torch --simplefold_model simplefold_100M \
    --ckpt_dir "$PWD/artifacts/checkpoints" --cache_dir "$PWD/artifacts/ccd" \
    --fasta_path "$fasta_path" --output_dir "$output_dir" \
    --num_steps 20 --tau 0.01 --nsample_per_protein 1 \
    --seed 42 --output_format pdb
python scripts/validate_predictions.py --fasta_path "$fasta_path" \
    --prediction_dir "$output_dir/predictions_simplefold_100M" --num_samples 1
echo "CPU smoke test completed on $(hostname) at $(date -Is)"
