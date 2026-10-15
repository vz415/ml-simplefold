#!/usr/bin/env python3
"""Run Slurm sampling into experiment-local home storage, then mirror to /pub.

The run stays unsealed until downstream evaluation writers have finished.
Retrieval seals, verifies both copies and removes only the temporary run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import result_cache as cache

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = "/pub/ynkim4/ml-simplefold/artifacts"
INSPECTION_FLAGS = {"--cfg", "-c", "--help", "-h", "--hydra-help", "--info", "-i", "--version"}


def launch_paths(arguments, environment, root=ROOT):
    job = environment.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("Sampling requires a numeric SLURM_JOB_ID")
    artifact_root = environment.get("SIMPLEFOLD_ARTIFACT_DIR", DEFAULT_ARTIFACT_ROOT)
    for argument in arguments:
        key, _, value = argument.partition("=")
        key = key.lstrip("+")
        if key in {"hydra.run.dir", "hydra.sweep.dir"}:
            raise ValueError("Use SIMPLEFOLD_RESULT_CACHE_ROOT for cached launcher output paths")
        if key == "paths.artifact_dir":
            artifact_root = value
    if not Path(artifact_root).is_absolute() or "${" in artifact_root or "," in artifact_root:
        raise ValueError("Canonical artifact root must be one explicit absolute path")
    cache_root = Path(environment.get("SIMPLEFOLD_RESULT_CACHE_ROOT", str(root / "artifacts/transfer-stage")))
    if not cache_root.is_absolute():
        raise ValueError("SIMPLEFOLD_RESULT_CACHE_ROOT must be absolute")
    cache_root, run = cache._cache_path(cache_root, job)
    canonical = cache._run_path(Path(artifact_root) / "runs" / job, job, check_filesystem=False)
    cache._separate(cache_root, Path(artifact_root))
    return run, canonical


def mirror_run(run, canonical, timeout):
    """Keep the home copy if /pub is unavailable; never delete remote data."""
    subprocess.run(["mkdir", "-p", str(canonical.parent)], check=True, timeout=timeout)
    subprocess.run(["rsync", "-a", "--checksum", f"--timeout={timeout}",
                    str(run) + "/", str(canonical) + "/"], check=True, timeout=timeout)


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    command = [sys.executable, str(ROOT / "scripts/run_sampling.py")]
    if any(arg.split("=", 1)[0] in INSPECTION_FLAGS for arg in arguments):
        return subprocess.call(command + arguments)
    run, canonical = launch_paths(arguments, os.environ)
    if run.exists() or run.is_symlink():
        raise ValueError(f"Temporary run already exists; refusing reuse: {run}")
    timeout = int(os.environ.get("SIMPLEFOLD_STORAGE_TIMEOUT_SECONDS", "120"))
    if timeout <= 0:
        raise ValueError("Storage timeout must be positive")
    # Change output only. Weights, references and canonical paths remain /pub.
    overrides = [f"hydra.run.dir={run}", f"hydra.sweep.dir={run}"]
    print(f"Temporary results: {run}\nCanonical archive: {canonical}", flush=True)
    status = subprocess.call(command + arguments + overrides)
    if not run.is_dir():
        return status or 1
    receipt = dict(job_id=os.environ["SLURM_JOB_ID"], temporary_run=str(run),
                   canonical_run=str(canonical), sampling_exit_code=status,
                   state="unsealed_pending_downstream_completion",
                   cleanup="Use retrieve_results.py after all sampling/evaluation writers finish.")
    with (run / "RESULT_STORAGE.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    if status:
        print(f"Sampling failed ({status}); temporary output retained at {run}", flush=True)
        return status
    try:
        mirror_run(run, canonical, timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"Sampling completed; archive copy pending: {error}\nRecover from {run}", flush=True)
        # Evaluators for new campaigns must read the home run first. A /pub
        # outage should not cause a scientifically completed run to be retried.
    else:
        print("Sampling archive copy completed; home run retained for evaluation/download.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
