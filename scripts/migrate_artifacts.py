"""Move an artifact tree to project storage after verifying every file."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root):
    files = {}
    directories = []
    def walk_error(error):
        raise error

    for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
        directory = Path(directory)
        for name in sorted(dirs):
            if (directory / name).is_symlink():
                raise RuntimeError(f"Refusing a symlink inside artifacts: {directory / name}")
        directories.append(directory.relative_to(root).as_posix())
        for name in sorted(names):
            path = directory / name
            if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
                raise RuntimeError(f"Refusing a non-regular artifact: {path}")
            files[path.relative_to(root).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
    return files, sorted(directories)


def migrate(source, destination, job_id):
    source = Path(source).absolute()
    destination = Path(destination).absolute()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("Invalid migration job ID")
    print(f"Migrating {source} -> {destination}", flush=True)
    resolved_destination = destination.resolve()
    if source.is_symlink():
        raise RuntimeError(f"Unexpected artifact symlink: {source}")
    if not source.exists() and destination.is_dir():
        for receipt in sorted(destination.glob("migration-*.json")):
            previous = json.loads(receipt.read_text())
            if (previous.get("state") != "migrated"
                    or previous.get("source") != str(source)
                    or previous.get("destination") != str(destination)):
                continue
            for relative, expected in previous["files"].items():
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise RuntimeError(f"Invalid path in migration receipt: {relative}")
                target = destination / relative
                if (target.is_symlink() or not target.is_file()
                        or target.stat().st_size != expected["size"]
                        or sha256(target) != expected["sha256"]):
                    raise RuntimeError(f"Destination verification failed: {target}")
            return {"state": "already_migrated", "source": str(source),
                    "destination": str(destination), "verified_receipt": str(receipt)}
    if not source.is_dir():
        raise RuntimeError(f"Artifact source is not a directory: {source}")
    resolved_source = source.resolve()
    if (resolved_source == resolved_destination
            or resolved_source in resolved_destination.parents
            or resolved_destination in resolved_source.parents):
        raise RuntimeError("Source and destination must be separate directories")
    backup = source.with_name(f".{source.name}.migration-backup-{job_id}")
    if backup.exists() or backup.is_symlink():
        raise RuntimeError(f"Migration backup already exists: {backup}")

    files, directories = inventory(source)
    print(f"Source verified: {len(files)} files, {sum(item['size'] for item in files.values())} bytes", flush=True)
    report = destination / f"migration-{job_id}.json"
    if report.name in files or report.exists() or report.is_symlink():
        raise RuntimeError(f"Migration report would conflict with existing data: {report}")
    if destination.is_symlink():
        raise RuntimeError(f"Refusing a symlink destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for relative in directories:
        print(f"Preparing directory: {relative}", flush=True)
        target_directory = destination / relative
        if target_directory.is_symlink():
            raise RuntimeError(f"Refusing a symlink destination directory: {target_directory}")
        target_directory.mkdir(parents=True, exist_ok=True)
    for relative, expected in files.items():
        original = source / relative
        target = destination / relative
        if target.exists() or target.is_symlink():
            if (target.is_symlink() or not target.is_file()
                    or target.stat().st_size != expected["size"]
                    or sha256(target) != expected["sha256"]):
                raise RuntimeError(f"Destination conflict; original preserved: {target}")
            continue
        partial = target.with_name(f".{target.name}.migration-{job_id}.part")
        if partial.exists() or partial.is_symlink():
            raise RuntimeError(f"Partial migration file already exists: {partial}")
        try:
            # Fresh files inherit the destination's group; do not copy permissions.
            shutil.copyfile(original, partial)
            if partial.stat().st_size != expected["size"] or sha256(partial) != expected["sha256"]:
                raise RuntimeError(f"Copy checksum failed; original preserved: {original}")
            # Publish without overwriting a file that appeared during copying.
            os.link(partial, target)
        finally:
            if partial.exists():
                partial.unlink()
        print(f"Verified: {relative}", flush=True)

    for relative, expected in files.items():
        target = destination / relative
        if target.stat().st_size != expected["size"] or sha256(target) != expected["sha256"]:
            raise RuntimeError(f"Destination verification failed: {target}")
    if inventory(source) != (files, directories):
        raise RuntimeError("Source changed during migration; original preserved")

    manifest = {
        "state": "migrated_cleanup_pending", "job_id": job_id,
        "source": str(source), "destination": str(destination),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files), "total_bytes": sum(item["size"] for item in files.values()),
        "files": files, "retained_backup": str(backup),
    }
    source.rename(backup)
    try:
        with report.open("x") as handle:
            handle.write(json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        backup.rename(source)
        raise
    try:
        shutil.rmtree(backup)
    except OSError as error:
        manifest["cleanup_error"] = str(error)
    else:
        manifest["state"] = "migrated"
        del manifest["retained_backup"]
    report.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise SystemExit("Run bulk artifact migration through Slurm.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = migrate(args.source, args.destination, job_id)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))
    if result["state"] == "migrated_cleanup_pending":
        raise SystemExit("Artifacts moved; the verified home backup still needs cleanup.")


if __name__ == "__main__":
    main()
