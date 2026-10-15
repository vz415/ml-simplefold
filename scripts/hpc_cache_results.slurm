#!/usr/bin/env bash
#SBATCH --job-name=simplefold-cache-results
#SBATCH --partition=standard
#SBATCH --account=eehui_lab
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/cache-results-%j.out
#SBATCH --error=logs/cache-results-%j.err
# Attach with afterok to the last evaluation writer of a frozen historical run.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm with an afterok dependency}"
cd "${SLURM_SUBMIT_DIR:?}"
source_job="${1:?Pass the completed sampling job ID}"
[[ "$source_job" =~ ^[0-9]+$ ]] || exit 2
cache_root="${SIMPLEFOLD_RESULT_CACHE_ROOT:-$PWD/artifacts/transfer-stage}"
canonical="${SIMPLEFOLD_ARTIFACT_DIR:-/pub/ynkim4/ml-simplefold/artifacts}/runs/$source_job"
cache_helper="${SIMPLEFOLD_RESULT_CACHE_HELPER:-$PWD/scripts/result_cache.py}"
cache_python="${SIMPLEFOLD_CACHE_PYTHON:-/data/homezvol2/ynkim4/.conda/envs/simplefold/bin/python}"
timeout -k 5s 300s "$cache_python" "$cache_helper" stage \
    --source "$canonical" --canonical "$canonical" \
    --cache-root "$cache_root" --job-id "$source_job" --completed
manifest_sha=$("$cache_python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
    "$cache_root/runs/$source_job/CACHE_MANIFEST.json")
timeout -k 5s 120s "$cache_python" "$cache_helper" publish \
    --cache-root "$cache_root" --job-id "$source_job" \
    --expected-manifest-sha256 "$manifest_sha"
printf 'Verified temporary result copy: %s/runs/%s\n' "$cache_root" "$source_job"
