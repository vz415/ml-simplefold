#!/usr/bin/env python3
"""Retrieve a completed run through its temporary home copy, then clean it up.

Pass every downstream evaluation job with --evaluation-job. The script checks
Slurm completion, preserves /pub, and retains the cache on any verification error.
SSH uses the direct login host for status and the RCIC transfer host for files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import result_cache as cache

ROOT = Path(__file__).resolve().parents[1]
REMOTE_DRIVER = """
import json, pathlib, sys
request = json.load(sys.stdin)
ns = {'__name__': 'result_cache'}
exec(request['module'], ns)
root, job = request['cache_root'], request['job_id']
run = pathlib.Path(root) / 'runs' / job
action = request['action']
if action == 'prepare':
    canonical = request['canonical']
    source_tree = run if run.exists() else pathlib.Path(canonical)
    for evaluation_job in request['evaluation_jobs']:
        evaluation = ns['_read_json'](source_tree / 'evaluation' / ('paid_analysis_' + evaluation_job) / 'receipt.json')
        if str(evaluation['analysis_slurm_job']) != evaluation_job or str(evaluation['source_sampling_job']) != job:
            raise ValueError('Evaluation receipt does not match requested completed writers')
    if (run / ns['MANIFEST']).is_file():
        source = ns['_read_json'](run / ns['MANIFEST'])['source']
        manifest = ns['seal'](root, job, source=source, canonical=canonical, completed=True)
    elif run.exists():
        receipt = ns['_read_json'](run / 'RESULT_STORAGE.json')
        if (receipt['job_id'] != job or receipt['sampling_exit_code'] != 0
                or receipt['temporary_run'] != str(run) or receipt['canonical_run'] != canonical):
            raise ValueError('Home run completion receipt does not match this run')
        manifest = ns['seal'](root, job, source=str(run), canonical=canonical, completed=True)
    else:
        manifest = ns['stage'](canonical, root, job, canonical=canonical, completed=True)
    result = {'manifest': manifest, 'manifest_sha256': ns['sha256'](run / ns['MANIFEST'])}
elif action == 'publish':
    result = ns['publish'](root, job, expected_manifest_sha256=request['manifest_sha256'])
elif action == 'cleanup':
    receipt_path = run / ns['DOWNLOAD']
    receipt = request['download_receipt']
    if receipt_path.exists():
        if ns['_read_json'](receipt_path) != receipt:
            raise ValueError('Existing download receipt differs')
    else:
        ns['_write_new'](receipt_path, receipt)
    result = ns['cleanup'](root, job, expected_manifest_sha256=request['manifest_sha256'], download_receipt=str(receipt_path))
else:
    raise ValueError('Unknown operation')
print(json.dumps(result, sort_keys=True))
"""


def numeric_job(value):
    if not re.fullmatch(r"[0-9]+", value):
        raise argparse.ArgumentTypeError("Job IDs must be numeric")
    return value


def check_completed(output, jobs):
    rows = {}
    for line in output.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3:
            rows[parts[0]] = (parts[1], parts[2])
    for job in jobs:
        if rows.get(job) != ("COMPLETED", "0:0"):
            raise ValueError(f"Job {job} is not confirmed COMPLETED/0:0: {rows.get(job)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", type=numeric_job)
    parser.add_argument("--evaluation-job", action="append", type=numeric_job, default=[])
    parser.add_argument("--host", default="hpc3y")
    parser.add_argument("--remote-repo", default="/data/homezvol2/ynkim4/ml-simplefold")
    parser.add_argument("--cache-root")
    parser.add_argument("--artifact-root", default="/pub/ynkim4/ml-simplefold/artifacts")
    parser.add_argument("--remote-python", default="/data/homezvol2/ynkim4/.conda/envs/simplefold/bin/python")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    cache_root = args.cache_root or str(Path(args.remote_repo) / "artifacts/transfer-stage")
    canonical = str(Path(args.artifact_root) / "runs" / args.job_id)
    if not Path(cache_root).is_absolute() or not Path(args.artifact_root).is_absolute():
        parser.error("Remote storage paths must be absolute")
    destination = (args.destination or ROOT / "artifacts/remote-runs" / args.job_id).absolute()
    cache._path(destination)  # Reject symlink/traversal destinations before rsync.
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    jobs = [args.job_id, *args.evaluation_job]
    command = ["sacct", "-X", "-n", "-P", "-j", ",".join(jobs), "--format=JobIDRaw,State,ExitCode"]
    status = subprocess.check_output(ssh + [args.host, shlex.join(command)], text=True, timeout=40)
    check_completed(status, jobs)
    transfer_ssh = ssh + ["-o", "HostName=access-hpc3.rcic.uci.edu"]
    module = Path(cache.__file__).read_text()

    def remote(action, **extra):
        request = dict(action=action, module=module, cache_root=cache_root,
                       canonical=canonical, job_id=args.job_id, evaluation_jobs=args.evaluation_job, **extra)
        command = ["timeout", "-k", "5s", str(args.timeout) + "s", args.remote_python, "-c", REMOTE_DRIVER]
        result = subprocess.run(ssh + [args.host, shlex.join(command)],
                                input=json.dumps(request), text=True, capture_output=True,
                                timeout=args.timeout + 25)
        if result.returncode:
            raise RuntimeError(f"Remote {action} failed; cache retained: {result.stderr[-3000:]}")
        return json.loads(result.stdout)

    prepared = remote("prepare")
    manifest_sha = prepared["manifest_sha256"]
    destination.mkdir(parents=True, exist_ok=True)
    source = f"{args.host}:{shlex.quote(str(Path(cache_root) / 'runs' / args.job_id))}/"
    # Existing local bytes are never replaced. A conflict fails verification.
    subprocess.run(["rsync", "-a", "--checksum", "--ignore-existing", f"--timeout={args.timeout}",
                    "-e", shlex.join(transfer_ssh), source, str(destination) + "/"],
                   check=True, timeout=args.timeout + 25)
    cache.verify_payload(destination, expected_manifest_sha256=manifest_sha)
    print(f"Downloaded and SHA-256 verified: {destination}", flush=True)
    # Publication can be retried later if /pub is unavailable. Downloaded data
    # are already usable, and the home recovery copy remains until both pass.
    publication = remote("publish", manifest_sha256=manifest_sha)
    publication_path = destination / cache.PUBLICATION
    if publication_path.exists():
        if cache._read_json(publication_path) != publication:
            raise ValueError("Local publication receipt differs; cache retained")
    else:
        cache._write_new(publication_path, publication)
    downloaded = cache.verify_download(destination, expected_manifest_sha256=manifest_sha)
    removed = remote("cleanup", manifest_sha256=manifest_sha, download_receipt=downloaded)
    receipt_root = ROOT / "artifacts/transfer-receipts"
    receipt_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with (receipt_root / f"{args.job_id}_{stamp}.json").open("x") as stream:
        json.dump(dict(slurm_jobs_checked=jobs, local_directory=str(destination), **removed), stream, indent=2)
        stream.write("\n")
    print(f"Removed verified temporary run: {cache_root}/runs/{args.job_id}; /pub retained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
