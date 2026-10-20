"""Cache the official AFDB v4 Swiss-Prot CIF archive and extract exact members.

Downloads require Slurm. Archive integrity is checked once per process; member
extraction uses separate file handles, so one accessor can serve worker threads.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request


ARCHIVE_URL = "https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/swissprot_cif_v4.tar"
ARCHIVE_BYTES = 39_471_090_688
_CHUNK = 8 * 1024 * 1024
_ACCESSORS = {}
_LOCK = threading.Lock()


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _download(path, url, timeout, attempts):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Download the Swiss-Prot archive through Slurm")
    partial = path.with_name(path.name + ".partial")
    metadata_path = partial.with_name(partial.name + ".json")
    receipt_path = path.with_name(path.name + ".download.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if partial.exists() and metadata.get("url") != url:
        raise ValueError(f"Unverified partial archive URL: {partial}")
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > ARCHIVE_BYTES:
                raise ValueError("Partial Swiss-Prot archive exceeds the pinned size")
            checksum = hashlib.sha256()
            if offset:
                with partial.open("rb") as stream:
                    for block in iter(lambda: stream.read(_CHUNK), b""):
                        checksum.update(block)
            if offset < ARCHIVE_BYTES:
                headers = {"User-Agent": "SimpleFold-finetuning/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                    validator = metadata.get("etag") or metadata.get("last_modified")
                    if validator:
                        headers["If-Range"] = validator
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    if response.status == 200:
                        # An ignored Range or changed If-Range means a fresh file.
                        offset = 0
                        checksum = hashlib.sha256()
                    elif response.status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                             response.headers.get("Content-Range", ""))
                        if not match or tuple(map(int, match.groups())) != (
                                offset, ARCHIVE_BYTES - 1, ARCHIVE_BYTES):
                            raise ValueError("Unexpected archive Content-Range")
                        old_etag = metadata.get("etag")
                        if old_etag and response.headers.get("ETag") != old_etag:
                            raise ValueError("Archive ETag changed during resume")
                    else:
                        raise ValueError(f"Unexpected archive HTTP status {response.status}")
                    if int(response.headers.get("Content-Length", -1)) != ARCHIVE_BYTES - offset:
                        raise ValueError("Archive Content-Length differs from the pinned v4 size")
                    metadata = dict(url=url, expected_bytes=ARCHIVE_BYTES,
                                    etag=response.headers.get("ETag"),
                                    last_modified=response.headers.get("Last-Modified"))
                    with partial.open("ab" if offset else "wb") as stream:
                        # Truncate an obsolete prefix before recording a new
                        # validator, so interruption cannot mix two versions.
                        _write_json(metadata_path, metadata)
                        print(f"Swiss-Prot v4 archive: {offset}/{ARCHIVE_BYTES} bytes", flush=True)
                        for block in iter(lambda: response.read(_CHUNK), b""):
                            if offset + len(block) > ARCHIVE_BYTES:
                                raise ValueError("Archive response exceeded Content-Length")
                            stream.write(block)
                            checksum.update(block)
                            previous_gib = offset // (1024 ** 3)
                            offset += len(block)
                            if offset // (1024 ** 3) != previous_gib:
                                print(f"Swiss-Prot v4 archive: {offset}/{ARCHIVE_BYTES} bytes", flush=True)
                if offset != ARCHIVE_BYTES:
                    raise OSError(f"Incomplete Swiss-Prot archive: {offset}/{ARCHIVE_BYTES}")
            receipt = dict(metadata, sha256=checksum.hexdigest(), bytes=offset,
                           downloaded_at=datetime.now(timezone.utc).isoformat())
            # Persist provenance before promotion, so interruption cannot leave
            # a completed archive without its checksum receipt.
            _write_json(receipt_path, receipt)
            partial.replace(path)
            metadata_path.unlink(missing_ok=True)
            return receipt
        except urllib.error.HTTPError as error:
            if error.code in (400, 401, 403, 404, 410) or attempt + 1 == attempts:
                raise
        except (OSError, ValueError):
            if attempt + 1 == attempts:
                raise
        time.sleep(min(2 ** attempt, 8))


def _member_index(path, archive_sha256):
    index_path = path.with_name(path.name + ".index.json")
    if index_path.exists():
        saved = json.loads(index_path.read_text())
        if saved.get("archive_sha256") == archive_sha256 and saved.get("schema_version") == 1:
            return saved["members"]
    members = {}
    print("Indexing Swiss-Prot v4 TAR members", flush=True)
    # The archive is uncompressed TAR containing independently gzipped CIFs.
    # Stream through it once rather than seeking on the shared filesystem for
    # each header. Subsequent extraction reads only the selected compressed CIF.
    with path.open("rb", buffering=_CHUNK) as stream:
        with tarfile.open(fileobj=stream, mode="r|", bufsize=_CHUNK) as archive:
            for member in archive:
                name = PurePosixPath(member.name).name
                if not member.isfile() or not name.endswith("-model_v4.cif.gz"):
                    continue
                catalog_id = name.removesuffix(".cif.gz")
                if catalog_id in members:
                    raise ValueError(f"Duplicate archive member for {catalog_id}")
                members[catalog_id] = dict(name=member.name, offset=member.offset_data,
                                           size=member.size)
    if not members:
        raise ValueError("Swiss-Prot archive has no v4 CIF members")
    _write_json(index_path, dict(schema_version=1, archive_sha256=archive_sha256,
                                 members=members))
    print(f"Indexed {len(members)} Swiss-Prot v4 CIF members", flush=True)
    return members


class SwissProtArchive:
    def __init__(self, path, receipt, members):
        self.path = path
        self.receipt = receipt
        self.archive_sha256 = receipt["sha256"]
        self.archive_url = receipt["url"]
        self.members = members
        self.signature = _signature(path)

    def extract(self, catalog_id, destination):
        """Write the exact member as plain CIF; missing IDs raise KeyError."""
        member = self.members[catalog_id]
        if _signature(self.path) != self.signature:
            raise ValueError("Swiss-Prot archive changed after integrity verification")
        with self.path.open("rb") as stream:
            stream.seek(member["offset"])
            compressed = stream.read(member["size"])
        if len(compressed) != member["size"]:
            raise OSError(f"Truncated archive member: {member['name']}")
        content = gzip.decompress(compressed)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent,
                    prefix=destination.name + ".", suffix=".partial", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return dict(url=f"{self.archive_url}#{member['name']}",
                    sha256=hashlib.sha256(content).hexdigest(), bytes=len(content),
                    archive_sha256=self.archive_sha256,
                    archive_bytes=self.receipt["bytes"],
                    member=member["name"], member_offset=member["offset"],
                    member_bytes=member["size"],
                    member_sha256=hashlib.sha256(compressed).hexdigest(),
                    extracted_at=datetime.now(timezone.utc).isoformat())


def ensure_archive(cache, archive_url, timeout=90, attempts=3):
    """Download/verify/index exact official v4 once, returning a shared accessor."""
    if archive_url != ARCHIVE_URL:
        raise ValueError("Use the pinned official Swiss-Prot CIF v4 archive URL")
    if attempts < 1 or timeout <= 0:
        raise ValueError("Archive attempts and timeout must be positive")
    path = Path(cache).resolve() / "archives" / "swissprot_cif_v4.tar"
    key = (str(path), archive_url)
    with _LOCK:
        cached = _ACCESSORS.get(key)
        if cached is not None:
            if _signature(path) != cached.signature:
                raise ValueError("Swiss-Prot archive changed after integrity verification")
            return cached
        path.parent.mkdir(parents=True, exist_ok=True)
        # Also serialize separate preparation jobs sharing this cache.
        with path.with_name(path.name + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                receipt_path = path.with_name(path.name + ".download.json")
                receipt = json.loads(receipt_path.read_text())
                if (receipt["url"] != archive_url or receipt["bytes"] != ARCHIVE_BYTES
                        or path.stat().st_size != ARCHIVE_BYTES
                        or _digest(path) != receipt["sha256"]):
                    raise ValueError("Swiss-Prot archive cache integrity mismatch")
            else:
                receipt = _download(path, archive_url, timeout, attempts)
            accessor = SwissProtArchive(path, receipt, _member_index(path, receipt["sha256"]))
            _ACCESSORS[key] = accessor
            return accessor
