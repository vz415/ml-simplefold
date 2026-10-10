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
cache_dir="${SIMPLEFOLD_CACHE_DIR:-$artifact_dir}"
export TORCH_HOME="$cache_dir/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
fasta_path="${1:-examples/crambin.fasta}"
num_steps="${2:-500}"
num_samples="${3:-1}"
tau="${4:-0.01}"
seed="${5:-42}"
folding_model="${6:-simplefold_100M}"
sampling_config="${7:-}"
case "$folding_model" in
    simplefold_100M|simplefold_360M|simplefold_700M|simplefold_1.1B|simplefold_1.6B|simplefold_3B) ;;
    *) echo "Unknown folding model: $folding_model" >&2; exit 1 ;;
esac
output_dir="$artifact_dir/runs/${SLURM_JOB_ID}"
storage_timeout="${SIMPLEFOLD_STORAGE_TIMEOUT_SECONDS:-120}"
if [[ ! "$storage_timeout" =~ ^[1-9][0-9]*$ ]]; then
    echo "SIMPLEFOLD_STORAGE_TIMEOUT_SECONDS must be a positive integer." >&2
    exit 1
fi
echo "Checking model storage (timeout ${storage_timeout}s)."
echo "Storage host: $(hostname); account: $(id -un); cache: $cache_dir"
for storage_path in \
    "$cache_dir/checkpoints" \
    "$cache_dir/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt" \
    "$cache_dir/ccd/ccd.pkl"; do
    echo "Checking: $storage_path"
    if timeout -k 5s "$storage_timeout" stat -- "$storage_path" >/dev/null; then
        echo "Accessible: $storage_path"
    else
        storage_status=$?
        echo "Storage check failed (exit $storage_status) for $storage_path on $(hostname); stopping before inference." >&2
        exit 1
    fi
done
if [[ -n "$sampling_config" ]]; then
    reference_dir="${SIMPLEFOLD_FK_REFERENCE_DIR:-$artifact_dir/datasets/references/2goo-native}"
    for reference_path in "$sampling_config" "$reference_dir/reference.cif" "$reference_dir/reference.pdb"; do
        timeout -k 5s "$storage_timeout" test -s "$reference_path"
    done
fi
python scripts/download_weights.py --artifact_dir "$cache_dir" --simplefold_model "$folding_model"
mkdir -p "$output_dir"
git rev-parse HEAD > "$output_dir/git-commit.txt"
python -m pip freeze > "$output_dir/pip-freeze.txt"
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; refusing CPU sampling"; print(torch.__version__, torch.cuda.get_device_name(0))'
python - "$fasta_path" "$output_dir" "$num_steps" "$num_samples" "$tau" "$seed" "$folding_model" "$sampling_config" <<'PY'
import hashlib
import json
import math
import os
import sys
from pathlib import Path

fasta, output, steps, samples, tau, seed, model, config = sys.argv[1:]
assert int(steps) > 0 and int(samples) > 0
assert math.isfinite(float(tau)) and float(tau) >= 0 and int(seed) >= 0
settings = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "model": model,
    "sampling_mode": "paired_frozen_weight_fk" if config else "standard",
    "sampling_config": config or None,
    "encoder": "ESM2-3B",
    "backend": "torch",
    "fasta_path": fasta,
    "fasta_sha256": hashlib.sha256(Path(fasta).read_bytes()).hexdigest() if Path(fasta).is_file() else None,
    "num_steps": int(steps),
    "nsample_per_protein": int(samples),
    "tau": float(tau),
    "seed": int(seed),
    "log_timesteps": True,
    "compute_plddt": False,
    "output_format": "pdb",
    "output_dir": output,
    "source_commit": (Path(output) / "git-commit.txt").read_text().strip(),
}
(Path(output) / "experiment.json").write_text(json.dumps(settings, indent=2) + "\n")
print(json.dumps(settings, indent=2))
PY
if [[ -n "$sampling_config" ]]; then
    reference_dir="${SIMPLEFOLD_FK_REFERENCE_DIR:-$artifact_dir/datasets/references/2goo-native}"
    python scripts/sample_complex_fk.py --config "$sampling_config" \
        --model "$folding_model" --ckpt-dir "$cache_dir/checkpoints" \
        --cache-dir "$cache_dir/ccd" --fasta-path "$fasta_path" \
        --output-dir "$output_dir" --num-steps "$num_steps" \
        --tau "$tau" --particles "$num_samples" --seed "$seed" \
        --reference-cif "$reference_dir/reference.cif" \
        --reference-pdb "$reference_dir/reference.pdb"
    for condition in baseline fk; do
        python scripts/validate_predictions.py --fasta_path "$fasta_path" \
            --prediction_dir "$output_dir/$condition/predictions_$folding_model" \
            --num_samples "$num_samples"
    done
else
    simplefold --backend torch --simplefold_model "$folding_model" \
        --ckpt_dir "$cache_dir/checkpoints" --cache_dir "$cache_dir/ccd" \
        --fasta_path "$fasta_path" --output_dir "$output_dir" \
        --num_steps "$num_steps" --tau "$tau" --nsample_per_protein "$num_samples" \
        --seed "$seed" --output_format pdb
    python scripts/validate_predictions.py --fasta_path "$fasta_path" \
        --prediction_dir "$output_dir/predictions_$folding_model" --num_samples "$num_samples"
fi
echo "Sampling completed on $(hostname) at $(date -Is)"
