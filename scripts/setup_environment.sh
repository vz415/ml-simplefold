#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-local}"
artifact_dir="$repo_dir/artifacts"
if [[ "$mode" == hpc ]]; then
    : "${SLURM_JOB_ID:?Run the HPC installation through Slurm}"
    env_prefix="${SIMPLEFOLD_ENV_PREFIX:-/data/homezvol2/ynkim4/.conda/envs/simplefold}"
    export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/data/homezvol2/ynkim4/.conda/pkgs}"
    artifact_dir="${SIMPLEFOLD_ARTIFACT_DIR:-/pub/ynkim4/ml-simplefold/artifacts}"
    mkdir -p "$artifact_dir"
    export PIP_CACHE_DIR="$artifact_dir/pip-cache"
    if [[ ! -x "$env_prefix/bin/python" ]]; then
        conda create --prefix "$env_prefix" --override-channels -c conda-forge python=3.10 pip -y
    fi
    python_bin="$env_prefix/bin/python"
elif [[ "$mode" == local ]]; then
    if [[ "$(uname -s)" != Darwin ]]; then
        echo 'The local setup is for the Mac; use hpc mode inside Slurm on HPC3.' >&2
        exit 1
    fi
    conda_base="$(conda info --base)"
    python_bin="$conda_base/envs/simplefold/bin/python"
    if [[ ! -x "$python_bin" ]]; then
        conda env create --file "$repo_dir/environment.yml"
    fi
else
    echo 'Usage: setup_environment.sh [local|hpc]' >&2
    exit 2
fi
"$python_bin" -m pip install --upgrade pip wheel build
if [[ "$mode" == hpc ]]; then
    "$python_bin" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
fi
"$python_bin" -m pip install -c "$repo_dir/requirements/runtime.txt" -e "$repo_dir"
"$python_bin" -m pip check
"$(dirname "$python_bin")/simplefold" --help
mkdir -p "$artifact_dir/environment"
"$python_bin" -m pip freeze > "$artifact_dir/environment/pip-$mode.txt"
echo "Environment ready: $python_bin"
