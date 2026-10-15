"""Checksum-bound, temporary run-result copies; standard library only.

The caller asserts that a run is complete and stops writers before sealing it.
This module does not query Slurm or infer completion from filenames. Only result
data belong in a cache; pretrained/training weights do not. Known weight paths
are rejected. Receipts are integrity records, not signed authentication tokens.

Workflow: stage/seal -> publish -> copy the entire cache run to the laptop ->
verify_download -> transfer CACHE_DOWNLOAD.json back -> cleanup. Publication
preserves unrelated canonical files. Cleanup removes only the sealed cache run,
never its parent directories or canonical results. Failed/incomplete operations
retain the cache for inspection. No generic recursive deletion is used.
"""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import tempfile


DEFAULT_CACHE_ROOT = "/data/homezvol2/ynkim4/ml-simplefold/artifacts/transfer-stage"
MANIFEST = "CACHE_MANIFEST.json"
PUBLICATION = "CACHE_PUBLICATION.json"
DOWNLOAD = "CACHE_DOWNLOAD.json"
CONTROL_FILES = {MANIFEST, PUBLICATION, DOWNLOAD}
# HPC3's system mount alias, not a user-configurable symlink allowance. Tests
# replace this table with an isolated temporary alias and verify owner/target.
_TRUSTED_SYSTEM_ALIASES = {Path("/pub"): Path("/dfs6b/pub")}


def _fail(message):
    raise ValueError(message)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _lexical(value):
    path = Path(value)
    if ".." in path.parts:
        _fail(f"Path traversal is forbidden: {path}")
    return Path(os.path.abspath(path))


def _physical_spelling(value):
    """Normalize the pinned mount alias lexically, without touching storage."""
    path = _lexical(value)
    for alias, target in _TRUSTED_SYSTEM_ALIASES.items():
        if path == alias or alias in path.parents:
            return target / path.relative_to(alias)
    return path


def _path(value):
    path = _lexical(value)
    for part in reversed((path, *path.parents)):
        if part.is_symlink():
            target = _TRUSTED_SYSTEM_ALIASES.get(part)
            if (target is None or part.lstat().st_uid != 0
                    or os.readlink(part) != str(target)):
                _fail(f"Symlink path is forbidden: {part}")
            # The trusted target itself must not introduce another alias.
            for target_part in reversed((target, *target.parents)):
                if target_part.is_symlink():
                    _fail(f"Symlink behind system mount alias: {target_part}")
    return path


def _job(value):
    value = str(value)
    if not re.fullmatch(r"[0-9]+", value):
        _fail("job_id must contain ASCII digits only")
    return value


def _cache_path(cache_root, job_id, *, check_filesystem=True):
    validate = _path if check_filesystem else _lexical
    root = validate(cache_root)
    resolved = _physical_spelling(root)
    forbidden = (Path("/pub"), Path("/dfs6b/pub"))
    if root == Path("/") or any(p == root or p in root.parents or p == resolved or p in resolved.parents for p in forbidden):
        _fail("Cache root must not be / or under /pub")
    return root, validate(root / "runs" / _job(job_id))


def _run_path(value, job_id, *, check_filesystem=True):
    path = (_path if check_filesystem else _lexical)(value)
    if path.name != _job(job_id) or path.parent.name != "runs":
        _fail(f"Expected explicit runs/{job_id} directory: {path}")
    return path


def _separate(left, right):
    # I/O operations validate the real alias separately through _path. Metadata
    # checks must work on the laptop and while canonical storage is unavailable.
    left, right = _physical_spelling(left), _physical_spelling(right)
    if left == right or left in right.parents or right in left.parents:
        _fail(f"Paths must be separate trees: {left}, {right}")


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"Invalid inventory path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in (".", "..", "") for p in value.split("/")):
        _fail(f"Invalid inventory path: {value!r}")
    if path.as_posix() != value or path.parts[0] in CONTROL_FILES:
        _fail(f"Reserved or non-normalized inventory path: {value!r}")
    return path


def _data_path(relative):
    path = _relative(relative)
    if (any(p.lower() in {"weights", "checkpoints", "pretrained"} for p in path.parts)
            or path.suffix.lower() in {".ckpt", ".safetensors", ".pth"}
            or path.name.lower() in {"pytorch_model.bin", "model.pt", "weights.pt", "checkpoint.pt"}):
        _fail(f"Model weights are not run-result cache data: {relative}")


def _record(path):
    path = _path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            _fail(f"Expected regular file: {path}")
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    current = path.stat(follow_symlinks=False)
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)):
        _fail(f"File changed during verification: {path}")
    return {"size": after.st_size, "sha256": digest.hexdigest()}


def sha256(path):
    return _record(path)["sha256"]


def _inventory(root):
    root = _path(root)
    if not root.is_dir():
        _fail(f"Expected directory: {root}")
    files, directories = {}, []
    def visit(directory):
        for entry in sorted(directory.iterdir()):
            entry = _path(entry)
            mode = entry.stat(follow_symlinks=False).st_mode
            relative = entry.relative_to(root).as_posix()
            if entry.parent == root and entry.name in CONTROL_FILES:
                if not stat.S_ISREG(mode):
                    _fail(f"Invalid receipt file: {entry}")
                continue
            _data_path(relative)
            if stat.S_ISDIR(mode):
                directories.append(relative)
                visit(entry)
            elif stat.S_ISREG(mode):
                files[relative] = _record(entry)
            else:
                _fail(f"Non-regular cache entry: {entry}")
    visit(root)
    return {"files": files, "directories": sorted(directories)}


def _read_json(path):
    _record(path)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    def constant(value):
        _fail(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=constant)


def _write_new(path, value):
    """Publish a complete JSON file exclusively, without replacing any file."""
    path = _path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".cache-receipt-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic exclusive creation, including on NFS.
    finally:
        os.unlink(temporary)


def _check_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        _fail("An expected lowercase SHA256 is required")


def _load_manifest(directory, expected_manifest_sha256):
    _check_sha(expected_manifest_sha256)
    directory = _path(directory)
    if sha256(directory / MANIFEST) != expected_manifest_sha256:
        _fail("Manifest SHA256 mismatch")
    manifest = _read_json(directory / MANIFEST)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "result_cache_manifest" or manifest.get("completed") is not True:
        _fail("Invalid or incomplete cache manifest")
    job_id = _job(manifest["job_id"])
    root, run = _cache_path(manifest["cache_root"], job_id, check_filesystem=False)
    if manifest["cache_run"] != str(run):
        _fail("Manifest cache path/job mismatch")
    canonical = _run_path(manifest["canonical"], job_id, check_filesystem=False)
    _run_path(manifest["source"], job_id, check_filesystem=False)
    _separate(root, canonical)
    files, directories = manifest["files"], manifest["directories"]
    if not isinstance(files, dict) or not files or not isinstance(directories, list):
        _fail("Invalid or empty manifest inventory")
    if directories != sorted(set(directories)):
        _fail("Invalid manifest directory list")
    for relative in directories:
        _data_path(relative)
    for relative, record in files.items():
        _data_path(relative)
        if set(record) != {"size", "sha256"} or type(record["size"]) is not int or record["size"] < 0:
            _fail(f"Invalid file inventory record: {relative}")
        _check_sha(record["sha256"])
    if manifest["inventory_sha256"] != _digest({"files": files, "directories": directories}):
        _fail("Inventory digest mismatch")
    return manifest


def _bound(cache_root, job_id, expected_manifest_sha256):
    root, run = _cache_path(cache_root, job_id)
    manifest = _load_manifest(run, expected_manifest_sha256)
    if manifest["cache_root"] != str(root) or manifest["cache_run"] != str(run) or manifest["job_id"] != _job(job_id):
        _fail("Manifest is not bound to this cache root and job")
    _verify_payload(run, manifest, exact=True)
    return root, run, manifest


def _verify_payload(directory, manifest, *, exact):
    directory = _path(directory)
    expected = {"files": manifest["files"], "directories": manifest["directories"]}
    if exact:
        if _inventory(directory) != expected:
            _fail(f"Payload mismatch (changed, missing, or additional entries): {directory}")
    else:
        for relative in manifest["directories"]:
            if not _path(directory / relative).is_dir():
                _fail(f"Missing published directory: {relative}")
        for relative, record in manifest["files"].items():
            if _record(directory / relative) != record:
                _fail(f"Published file mismatch: {relative}")


def seal(cache_root, job_id, *, source, canonical, completed=False):
    """Seal already-completed root/runs/JOB data, or verify an existing seal."""
    if completed is not True:
        _fail("Caller must assert completed=True after all run writers stop")
    root, run = _cache_path(cache_root, job_id)
    source = _run_path(source, job_id, check_filesystem=False)
    canonical = _run_path(canonical, job_id, check_filesystem=False)
    _separate(root, canonical)
    if (run / MANIFEST).exists():
        _, _, manifest = _bound(root, job_id, sha256(run / MANIFEST))
        if manifest["source"] != str(source) or manifest["canonical"] != str(canonical):
            _fail("Existing seal has different source/canonical metadata")
        return manifest
    if any((run / name).exists() for name in CONTROL_FILES):
        _fail("Unsealed cache has pre-existing control files")
    inventory = _inventory(run)
    if not inventory["files"]:
        _fail("Cannot seal an empty result cache")
    manifest = {"schema_version": 1, "kind": "result_cache_manifest", "created_at": _now(),
                "completed": True, "job_id": _job(job_id), "cache_root": str(root),
                "cache_run": str(run), "source": str(source), "canonical": str(canonical),
                "inventory_sha256": _digest(inventory), **inventory}
    _write_new(run / MANIFEST, manifest)
    _verify_payload(run, manifest, exact=True)
    return manifest


def _copy_file(source, target, record):
    target = _path(target)
    if target.exists():
        if _record(target) != record:
            _fail(f"Destination conflict: {target}")
        return
    # On failure leave the incomplete destination for explicit inspection; never
    # overwrite a partial or conflicting file on retry.
    source_fd = os.open(_path(source), os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(source_fd, "rb") as original, open(target, "xb") as destination:
        if not stat.S_ISREG(os.fstat(original.fileno()).st_mode):
            _fail(f"Expected regular copy source: {source}")
        for chunk in iter(lambda: original.read(1024 * 1024), b""):
            destination.write(chunk)
        destination.flush()
        os.fsync(destination.fileno())
    if _record(target) != record or _record(source) != record:
        _fail(f"Copy checksum failed: {target}")


def stage(source, cache_root, job_id, *, canonical=None, completed=False):
    """Copy a completed explicit run into a NEW, exclusively-created cache."""
    if completed is not True:
        _fail("Caller must assert completed=True after all run writers stop")
    source = _run_path(source, job_id)
    canonical = _run_path(canonical if canonical is not None else source, job_id, check_filesystem=False)
    root, run = _cache_path(cache_root, job_id)
    _separate(root, source)
    _separate(root, canonical)
    if any((source / name).exists() for name in CONTROL_FILES):
        _fail("Source contains reserved cache-control files")
    inventory = _inventory(source)
    if not inventory["files"]:
        _fail("Cannot stage an empty run")
    run.parent.mkdir(parents=True, exist_ok=True)
    _path(run.parent)
    run.mkdir()  # Exclusive: never merge with a legacy or interrupted cache.
    for relative in inventory["directories"]:
        _path(run / relative).mkdir()
    for relative, record in inventory["files"].items():
        _copy_file(source / relative, run / relative, record)
    if _inventory(source) != inventory or _inventory(run) != inventory:
        _fail("Source or staged results changed while copying")
    return seal(root, job_id, source=source, canonical=canonical, completed=True)


def _receipt_binding(manifest, manifest_sha256):
    return {"job_id": manifest["job_id"], "cache_root": manifest["cache_root"],
            "cache_run": manifest["cache_run"], "canonical": manifest["canonical"],
            "manifest_sha256": manifest_sha256, "inventory_sha256": manifest["inventory_sha256"]}


def _validate_receipt(receipt, kind, manifest, manifest_sha256):
    if receipt.get("schema_version") != 1 or receipt.get("kind") != kind or receipt.get("state") != "verified":
        _fail(f"Unconfirmed {kind} receipt")
    for key, value in _receipt_binding(manifest, manifest_sha256).items():
        if receipt.get(key) != value:
            _fail(f"Receipt {key} does not match sealed inventory")


def publish(cache_root, job_id, *, expected_manifest_sha256):
    """Verify/copy payload to canonical storage; refuse any conflicting file."""
    _, run, manifest = _bound(cache_root, job_id, expected_manifest_sha256)
    canonical = _path(manifest["canonical"])
    canonical.mkdir(parents=True, exist_ok=True)
    for relative in manifest["directories"]:
        _path(canonical / relative).mkdir(parents=True, exist_ok=True)
    # Preflight all existing files before copying anything new.
    for relative, record in manifest["files"].items():
        target = _path(canonical / relative)
        if target.exists() and _record(target) != record:
            _fail(f"Destination conflict: {target}")
    for relative, record in manifest["files"].items():
        _copy_file(run / relative, canonical / relative, record)
    _verify_payload(canonical, manifest, exact=False)
    _bound(cache_root, job_id, expected_manifest_sha256)
    if (run / PUBLICATION).exists():
        receipt = _read_json(run / PUBLICATION)
        _validate_receipt(receipt, "result_cache_publication", manifest, expected_manifest_sha256)
        return receipt
    receipt = {"schema_version": 1, "kind": "result_cache_publication", "state": "verified",
               "verified_at": _now(), "verification_host": socket.gethostname(),
               **_receipt_binding(manifest, expected_manifest_sha256)}
    _write_new(run / PUBLICATION, receipt)
    return receipt


def verify_payload(download_dir, *, expected_manifest_sha256):
    """Check local bytes without remote I/O; this cannot authorize cleanup."""
    directory = _path(download_dir)
    manifest = _load_manifest(directory, expected_manifest_sha256)
    _separate(directory, _lexical(manifest["cache_root"]))
    _separate(directory, _lexical(manifest["canonical"]))
    _verify_payload(directory, manifest, exact=True)
    return {"state": "payload_verified_publication_unassessed", "download_dir": str(directory),
            **_receipt_binding(manifest, expected_manifest_sha256)}


def verify_download(download_dir, *, expected_manifest_sha256):
    """Verify the complete local downloaded cache; issue a bound local receipt."""
    verify_payload(download_dir, expected_manifest_sha256=expected_manifest_sha256)
    directory = _path(download_dir)
    manifest = _load_manifest(directory, expected_manifest_sha256)
    publication = _read_json(directory / PUBLICATION)
    _validate_receipt(publication, "result_cache_publication", manifest, expected_manifest_sha256)
    _verify_payload(directory, manifest, exact=True)
    publication_sha256 = sha256(directory / PUBLICATION)
    if (directory / DOWNLOAD).exists():
        receipt = _read_json(directory / DOWNLOAD)
        _validate_receipt(receipt, "result_cache_download", manifest, expected_manifest_sha256)
        if receipt.get("download_dir") != str(directory) or receipt.get("publication_sha256") != publication_sha256:
            _fail("Existing download receipt has different provenance")
        return receipt
    receipt = {"schema_version": 1, "kind": "result_cache_download", "state": "verified",
               "verified_at": _now(), "verification_host": socket.gethostname(),
               "verification_account": getpass.getuser(), "download_dir": str(directory),
               "publication_sha256": publication_sha256,
               **_receipt_binding(manifest, expected_manifest_sha256)}
    _write_new(directory / DOWNLOAD, receipt)
    return receipt


def _remove_exact(run, expected):
    """Unlink only pre-verified names, using no-follow directory descriptors."""
    descriptors = {}
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptors[""] = os.open(run, flags)
        for relative in sorted(expected["directories"], key=lambda p: (p.count("/"), p)):
            path = PurePosixPath(relative)
            parent = "" if str(path.parent) == "." else str(path.parent)
            descriptors[relative] = os.open(path.name, flags, dir_fd=descriptors[parent])
        # Ensure every directory still has exactly the prechecked entries.
        for relative, descriptor in descriptors.items():
            names = {PurePosixPath(p).name for p in (*expected["files"], *expected["directories"])
                     if ("" if str(PurePosixPath(p).parent) == "." else str(PurePosixPath(p).parent)) == relative}
            if set(os.listdir(descriptor)) != names:
                _fail("Cache entries changed before cleanup")
        for relative, record in expected["files"].items():
            if _record(run / relative) != record:
                _fail("Cache bytes changed before cleanup")
        for relative in expected["files"]:
            path = PurePosixPath(relative)
            parent = "" if str(path.parent) == "." else str(path.parent)
            os.unlink(path.name, dir_fd=descriptors[parent])
        for relative in sorted(expected["directories"], key=lambda p: (p.count("/"), p), reverse=True):
            path = PurePosixPath(relative)
            parent = "" if str(path.parent) == "." else str(path.parent)
            os.rmdir(path.name, dir_fd=descriptors[parent])
        # The only final removed directory is the exact numeric run itself.
        if (os.stat(run, follow_symlinks=False).st_dev, os.stat(run, follow_symlinks=False).st_ino) != (os.fstat(descriptors[""]).st_dev, os.fstat(descriptors[""]).st_ino):
            _fail("Cache run directory changed during cleanup")
        os.rmdir(run)
    finally:
        for descriptor in reversed(list(descriptors.values())):
            os.close(descriptor)


def cleanup(cache_root, job_id, *, expected_manifest_sha256, download_receipt):
    """Remove only this cache after publication AND local download verification."""
    _, run, manifest = _bound(cache_root, job_id, expected_manifest_sha256)
    publication = _read_json(run / PUBLICATION)
    _validate_receipt(publication, "result_cache_publication", manifest, expected_manifest_sha256)
    receipt_path = _path(download_receipt)
    downloaded = _read_json(receipt_path)
    download_receipt_sha256 = sha256(receipt_path)
    _validate_receipt(downloaded, "result_cache_download", manifest, expected_manifest_sha256)
    download_dir = _lexical(downloaded["download_dir"])
    _separate(download_dir, _lexical(manifest["cache_root"]))
    _separate(download_dir, _lexical(manifest["canonical"]))
    if downloaded.get("publication_sha256") != sha256(run / PUBLICATION):
        _fail("Download receipt does not bind this verified publication")
    if not downloaded.get("verification_host") or not downloaded.get("verification_account") or not downloaded.get("verified_at"):
        _fail("Missing local download-verification provenance")
    _verify_payload(_path(manifest["canonical"]), manifest, exact=False)
    _bound(cache_root, job_id, expected_manifest_sha256)
    expected = {"files": dict(manifest["files"]), "directories": manifest["directories"]}
    for name in CONTROL_FILES:
        path = run / name
        if path.exists():
            if name == DOWNLOAD:
                cached_receipt = _read_json(path)
                if cached_receipt != downloaded:
                    _fail("Cached download receipt conflicts with supplied receipt")
            expected["files"][name] = _record(path)
    _remove_exact(run, expected)
    return {"state": "cache_removed", **_receipt_binding(manifest, expected_manifest_sha256),
            "download_receipt_sha256": download_receipt_sha256}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("seal", "stage", "publish", "cleanup", "verify-download", "verify-payload"):
        command = commands.add_parser(name)
        if name in ("verify-download", "verify-payload"):
            command.add_argument("--download-dir", required=True)
        else:
            command.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
            command.add_argument("--job-id", required=True)
        if name in ("seal", "stage"):
            command.add_argument("--source", required=True)
            command.add_argument("--canonical", required=True)
            command.add_argument("--completed", action="store_true", required=True)
        else:
            command.add_argument("--expected-manifest-sha256", required=True)
        if name == "cleanup":
            command.add_argument("--download-receipt", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = globals()[command.replace("-", "_")](**args)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
