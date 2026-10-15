"""Verify result-cache safety with isolated tiny trees, including failed transfers."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "result_cache.py"
SPEC = importlib.util.spec_from_file_location("result_cache", SCRIPT)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


class ResultCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.job = "58123456"
        self.source = self.root / "source" / "runs" / self.job
        self.canonical = self.root / "pub" / "runs" / self.job
        self.cache_root = self.root / "experiment" / "artifacts" / "transfer-stage"
        self.run = self.cache_root / "runs" / self.job
        self.local = self.root / "laptop" / self.job
        self.payload = {"experiment.json": b'{"status":"complete"}\n',
                        "fk/predictions/sample.pdb": b"ATOM  payload\nEND\n",
                        "metrics.bin": b"\x00\xff\x03\x18"}
        for relative, content in self.payload.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (self.source / "empty" / "nested").mkdir(parents=True)

    def stage(self):
        manifest = cache.stage(self.source, self.cache_root, self.job,
                               canonical=self.canonical, completed=True)
        self.digest = cache.sha256(self.run / cache.MANIFEST)
        return manifest

    def publish(self):
        return cache.publish(self.cache_root, self.job, expected_manifest_sha256=self.digest)

    def download(self):
        shutil.copytree(self.run, self.local)
        cache.verify_download(self.local, expected_manifest_sha256=self.digest)
        return self.local / cache.DOWNLOAD

    def cleanup(self, receipt=None):
        return cache.cleanup(self.cache_root, self.job, expected_manifest_sha256=self.digest,
                             download_receipt=receipt or self.local / cache.DOWNLOAD)

    def assert_payload(self, directory):
        for relative, content in self.payload.items():
            self.assertEqual((directory / relative).read_bytes(), content)
        self.assertTrue((directory / "empty" / "nested").is_dir())

    def test_complete_roundtrip_removes_only_exact_cache_run(self):
        legacy = self.cache_root / "58139638"
        legacy.mkdir(parents=True)
        (legacy / "keep.txt").write_bytes(b"legacy must survive")
        sibling = self.cache_root / "runs" / "99999"
        sibling.mkdir(parents=True)
        (sibling / "keep.txt").write_bytes(b"other run")
        manifest = self.stage()
        self.assertEqual(manifest["files"]["metrics.bin"]["sha256"], hashlib.sha256(self.payload["metrics.bin"]).hexdigest())
        original_manifest = (self.run / cache.MANIFEST).read_bytes()
        self.publish()
        publication = (self.run / cache.PUBLICATION).read_bytes()
        self.publish()
        self.assertEqual((self.run / cache.PUBLICATION).read_bytes(), publication)
        self.assertEqual((self.run / cache.MANIFEST).read_bytes(), original_manifest)
        self.download()
        receipt = (self.local / cache.DOWNLOAD).read_bytes()
        cache.verify_download(self.local, expected_manifest_sha256=self.digest)
        self.assertEqual((self.local / cache.DOWNLOAD).read_bytes(), receipt)
        # This is the actual cross-host integration: the local receipt is copied back.
        shutil.copyfile(self.local / cache.DOWNLOAD, self.run / cache.DOWNLOAD)
        result = self.cleanup(self.run / cache.DOWNLOAD)
        self.assertEqual(result["state"], "cache_removed")
        self.assertEqual(result["download_receipt_sha256"], hashlib.sha256(receipt).hexdigest())
        self.assertFalse(self.run.exists())
        self.assertTrue(self.run.parent.is_dir())
        self.assertEqual((legacy / "keep.txt").read_bytes(), b"legacy must survive")
        self.assertEqual((sibling / "keep.txt").read_bytes(), b"other run")
        for directory in (self.source, self.canonical, self.local):
            self.assert_payload(directory)

    def test_stage_existing_canonical_run_never_changes_source_bytes(self):
        cache.stage(self.source, self.cache_root, self.job, completed=True)
        self.digest = cache.sha256(self.run / cache.MANIFEST)
        self.publish()
        self.download()
        self.cleanup()
        self.assert_payload(self.source)
        self.assertEqual(set(p.name for p in self.source.iterdir()), {"experiment.json", "fk", "metrics.bin", "empty"})

    def test_seal_direct_completed_cache_is_immutable_and_idempotent(self):
        shutil.copytree(self.source, self.run)
        with self.assertRaisesRegex(ValueError, "completed"):
            cache.seal(self.cache_root, self.job, source=self.run, canonical=self.canonical)
        manifest = cache.seal(self.cache_root, self.job, source=self.run, canonical=self.canonical, completed=True)
        before = (self.run / cache.MANIFEST).read_bytes()
        self.assertEqual(manifest, cache.seal(self.cache_root, self.job, source=self.run,
                                             canonical=self.canonical, completed=True))
        self.assertEqual(before, (self.run / cache.MANIFEST).read_bytes())
        with self.assertRaisesRegex(ValueError, "different source"):
            cache.seal(self.cache_root, self.job, source=self.source, canonical=self.canonical, completed=True)

    def test_home_seal_and_local_manifest_validation_do_not_access_pub(self):
        shutil.copytree(self.source, self.run)
        canonical = Path("/pub/ynkim4/ml-simplefold/artifacts/runs") / self.job
        original_path = cache._path
        def refuse_pub_io(path):
            path = Path(path)
            if path == Path("/pub") or Path("/pub") in path.parents:
                raise AssertionError("canonical storage must not be accessed")
            return original_path(path)
        with patch.object(cache, "_path", side_effect=refuse_pub_io):
            manifest = cache.seal(self.cache_root, self.job, source=self.run,
                                  canonical=canonical, completed=True)
            digest = cache.sha256(self.run / cache.MANIFEST)
            shutil.copytree(self.run, self.local)
            self.assertEqual(cache._load_manifest(self.local, digest), manifest)
            result = cache.verify_payload(self.local, expected_manifest_sha256=digest)
            self.assertEqual(result["state"], "payload_verified_publication_unassessed")
            self.assertFalse((self.local / cache.DOWNLOAD).exists())
            # A missing publication fails promptly without accessing /pub.
            with self.assertRaises(FileNotFoundError):
                cache.verify_download(self.local, expected_manifest_sha256=digest)

    def test_bad_paths_job_mismatch_and_pub_cache_are_refused(self):
        for job in ("../1", "123/x", "", "123_4", "１２３"):
            with self.subTest(job=job), self.assertRaises(ValueError):
                cache.stage(self.source, self.cache_root, job, completed=True)
        with self.assertRaisesRegex(ValueError, "traversal"):
            cache.stage(self.source, self.cache_root / ".." / "other", self.job, completed=True)
        with self.assertRaisesRegex(ValueError, "Expected explicit"):
            cache.stage(self.source.parent, self.cache_root, self.job, completed=True)
        with self.assertRaisesRegex(ValueError, "under /pub"):
            cache.stage(self.source, "/pub/not-a-cache", self.job, completed=True)
        with self.assertRaisesRegex(ValueError, "separate trees"):
            cache.stage(self.source, self.source.parent.parent, self.job, completed=True)

    def test_exclusive_stage_never_overwrites_legacy_or_existing_run(self):
        self.run.mkdir(parents=True)
        (self.run / "legacy.txt").write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            self.stage()
        self.assertEqual((self.run / "legacy.txt").read_bytes(), b"keep")
        self.assertFalse((self.run / cache.MANIFEST).exists())

    def test_symlinks_in_source_or_ancestors_are_refused(self):
        (self.source / "link").symlink_to(self.source / "experiment.json")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.stage()
        (self.source / "link").unlink()
        link = self.root / "linked"
        link.symlink_to(self.root / "source", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            cache.stage(link / "runs" / self.job, self.cache_root, self.job, completed=True)
        self.assertFalse(self.run.exists())

    def test_only_exact_root_owned_system_alias_is_allowed(self):
        target = self.root / "system-storage"
        target.mkdir()
        alias = self.root / "system-alias"
        alias.symlink_to(target, target_is_directory=True)
        alias_source = alias / "runs" / self.job
        shutil.copytree(self.source, target / "runs" / self.job)
        real_lstat = Path.lstat
        def root_owned_alias(path):
            result = real_lstat(path)
            if Path(path) == alias:
                fields = list(result)
                fields[4] = 0  # Only the fixed system alias receives root ownership.
                return os.stat_result(fields)
            return result
        def user_owned_alias(path):
            result = real_lstat(path)
            if Path(path) == alias:
                fields = list(result)
                fields[4] = 1001
                return os.stat_result(fields)
            return result
        with patch.object(cache, "_TRUSTED_SYSTEM_ALIASES", {alias: target}):
            with patch.object(Path, "lstat", new=user_owned_alias):
                with self.assertRaisesRegex(ValueError, "Symlink"):
                    cache.stage(alias_source, self.cache_root, self.job, completed=True)
            with patch.object(Path, "lstat", new=root_owned_alias):
                manifest = cache.stage(alias_source, self.cache_root, self.job, completed=True)
                self.assertEqual(manifest["canonical"], str(alias_source))
                with self.assertRaisesRegex(ValueError, "separate trees"):
                    cache._separate(alias_source, target / "runs" / self.job)
                # Even a root-owned alias must match the single pinned target.
                with patch.object(cache, "_TRUSTED_SYSTEM_ALIASES", {alias: self.root / "different"}):
                    with self.assertRaisesRegex(ValueError, "Symlink"):
                        cache._path(alias_source)

    def test_weight_paths_and_reserved_source_receipts_are_refused(self):
        for name in ("weights.pt", "large.ckpt", cache.MANIFEST):
            path = self.source / name
            path.write_bytes(b"not result data")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.stage()
            path.unlink()
        self.assertFalse(self.run.exists())

    def test_publish_conflicts_never_overwrite_any_canonical_file(self):
        self.stage()
        self.canonical.mkdir(parents=True)
        (self.canonical / "metrics.bin").write_bytes(b"conflicting bytes")
        with self.assertRaisesRegex(ValueError, "Destination conflict"):
            self.publish()
        self.assertEqual((self.canonical / "metrics.bin").read_bytes(), b"conflicting bytes")
        self.assertFalse((self.canonical / "experiment.json").exists())
        self.assertFalse((self.run / cache.PUBLICATION).exists())
        self.assert_payload(self.run)

    def test_publish_matching_existing_and_extra_canonical_file_preserved(self):
        self.stage()
        shutil.copytree(self.source, self.canonical)
        (self.canonical / "later-evaluation.json").write_bytes(b"independent later result")
        self.publish()
        self.download()
        self.cleanup()
        self.assertEqual((self.canonical / "later-evaluation.json").read_bytes(), b"independent later result")

    def test_partial_or_corrupt_download_never_produces_receipt(self):
        self.stage()
        self.publish()
        shutil.copytree(self.run, self.local)
        path = self.local / "metrics.bin"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Payload mismatch"):
            cache.verify_download(self.local, expected_manifest_sha256=self.digest)
        path.write_bytes(b"same length but bad bytes")
        with self.assertRaisesRegex(ValueError, "Payload mismatch"):
            cache.verify_download(self.local, expected_manifest_sha256=self.digest)
        self.assertFalse((self.local / cache.DOWNLOAD).exists())
        with self.assertRaises(FileNotFoundError):
            self.cleanup()
        self.assert_payload(self.run)

    def test_no_publication_or_local_verification_cannot_cleanup(self):
        self.stage()
        with self.assertRaises(FileNotFoundError):
            self.cleanup()
        shutil.copytree(self.run, self.local)
        with self.assertRaises(FileNotFoundError):
            cache.verify_download(self.local, expected_manifest_sha256=self.digest)
        self.publish()
        with self.assertRaises(FileNotFoundError):
            self.cleanup()
        self.assert_payload(self.run)

    def test_changed_additional_or_missing_cache_entries_prevent_cleanup(self):
        self.stage()
        self.publish()
        self.download()
        target = self.run / "metrics.bin"
        target.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Payload mismatch"):
            self.cleanup()
        target.write_bytes(self.payload["metrics.bin"])
        extra = self.run / "unexpected"
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, "Payload mismatch"):
            self.cleanup()
        extra.rmdir()
        target.unlink()
        with self.assertRaisesRegex(ValueError, "Payload mismatch"):
            self.cleanup()
        target.write_bytes(self.payload["metrics.bin"])
        link = self.run / "link"
        link.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.cleanup()
        self.assertTrue(self.run.exists())

    def test_corrupted_canonical_or_publication_retains_cache(self):
        self.stage()
        self.publish()
        self.download()
        (self.canonical / "metrics.bin").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Published file mismatch"):
            self.cleanup()
        (self.canonical / "metrics.bin").write_bytes(self.payload["metrics.bin"])
        receipt = json.loads((self.run / cache.PUBLICATION).read_text())
        receipt["state"] = "pending"
        (self.run / cache.PUBLICATION).write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "Unconfirmed"):
            self.cleanup()
        self.assert_payload(self.run)

    def test_receipt_for_other_inventory_or_wrong_manifest_hash_cannot_cleanup(self):
        self.stage()
        self.publish()
        self.download()
        with self.assertRaisesRegex(ValueError, "Manifest SHA256 mismatch"):
            cache.cleanup(self.cache_root, self.job, expected_manifest_sha256="0" * 64,
                          download_receipt=self.local / cache.DOWNLOAD)
        receipt = json.loads((self.local / cache.DOWNLOAD).read_text())
        receipt["inventory_sha256"] = "0" * 64
        (self.local / cache.DOWNLOAD).write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "Receipt inventory_sha256"):
            self.cleanup()
        self.assert_payload(self.run)

    def test_manifest_traversal_is_rejected_even_with_matching_external_hash(self):
        self.stage()
        manifest = json.loads((self.run / cache.MANIFEST).read_text())
        manifest["files"]["../escape"] = manifest["files"].pop("metrics.bin")
        manifest["inventory_sha256"] = cache._digest({"files": manifest["files"], "directories": manifest["directories"]})
        (self.run / cache.MANIFEST).write_text(json.dumps(manifest))
        self.digest = cache.sha256(self.run / cache.MANIFEST)
        with self.assertRaisesRegex(ValueError, "inventory path"):
            self.publish()

    def test_copied_manifest_cannot_authorize_different_cache_root_or_job(self):
        self.stage()
        new_root = self.root / "other-cache"
        shutil.copytree(self.run, new_root / "runs" / self.job)
        with self.assertRaisesRegex(ValueError, "not bound"):
            cache.publish(new_root, self.job, expected_manifest_sha256=self.digest)
        with self.assertRaises(ValueError):
            cache.publish(self.cache_root, "../" + self.job, expected_manifest_sha256=self.digest)

    def test_download_must_be_separate_from_remote_cache(self):
        self.stage()
        self.publish()
        with self.assertRaisesRegex(ValueError, "separate trees"):
            cache.verify_download(self.run, expected_manifest_sha256=self.digest)
        self.assertFalse((self.run / cache.DOWNLOAD).exists())

    def test_interrupted_copy_leaves_unsealed_cache_and_source_intact(self):
        original = cache._copy_file
        def fail_after_first(source, target, record):
            original(source, target, record)
            raise OSError("simulated interrupted transfer")
        with patch.object(cache, "_copy_file", side_effect=fail_after_first):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.stage()
        self.assertTrue(self.run.is_dir())
        self.assertFalse((self.run / cache.MANIFEST).exists())
        self.assert_payload(self.source)

    def test_cli_roundtrip_uses_per_command_options(self):
        def command(*args):
            result = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], check=True,
                                    capture_output=True, text=True)
            return json.loads(result.stdout)
        command("stage", "--source", self.source, "--canonical", self.canonical,
                "--cache-root", self.cache_root, "--job-id", self.job, "--completed")
        self.digest = cache.sha256(self.run / cache.MANIFEST)
        common = ("--cache-root", self.cache_root, "--job-id", self.job,
                  "--expected-manifest-sha256", self.digest)
        command("publish", *common)
        shutil.copytree(self.run, self.local)
        command("verify-download", "--download-dir", self.local,
                "--expected-manifest-sha256", self.digest)
        self.assertEqual(command("cleanup", *common, "--download-receipt", self.local / cache.DOWNLOAD)["state"], "cache_removed")
        self.assert_payload(self.canonical)


if __name__ == "__main__":
    unittest.main()
