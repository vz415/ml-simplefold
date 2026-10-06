"""Exercise artifact migration safety using tiny, isolated directory trees."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "migrate_artifacts.py"
SPEC = importlib.util.spec_from_file_location("storage_migration", SCRIPT_PATH)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


class StorageMigrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "home" / "artifacts"
        self.destination = self.root / "project" / "artifacts"
        self.job_id = "test_123"
        self.backup = self.source.with_name(f".artifacts.migration-backup-{self.job_id}")
        self.payloads = {
            "checkpoints/toy.ckpt": b"toy weights\x00\x01\xff",
            "runs/123/prediction.pdb": b"ATOM toy coordinates\nEND\n",
        }
        for relative, payload in self.payloads.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        (self.source / "empty" / "nested").mkdir(parents=True)

    def assert_source_preserved(self):
        self.assertTrue(self.source.is_dir())
        self.assertFalse(self.source.is_symlink())
        self.assertFalse(self.backup.exists())
        for relative, payload in self.payloads.items():
            self.assertEqual((self.source / relative).read_bytes(), payload)
        self.assertTrue((self.source / "empty" / "nested").is_dir())

    def test_successful_cutover_preserves_bytes_and_checksums(self):
        manifest = migration.migrate(self.source, self.destination, self.job_id)

        self.assertEqual(manifest["state"], "migrated")
        self.assertFalse(self.source.exists())
        self.assertFalse(self.source.is_symlink())
        self.assertFalse(self.backup.exists())
        self.assertTrue((self.destination / "empty" / "nested").is_dir())
        self.assertEqual(manifest["file_count"], len(self.payloads))
        self.assertEqual(manifest["total_bytes"], sum(map(len, self.payloads.values())))
        for relative, payload in self.payloads.items():
            self.assertEqual((self.destination / relative).read_bytes(), payload)
            self.assertEqual(manifest["files"][relative], {
                "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            })
        report = self.destination / f"migration-{self.job_id}.json"
        self.assertEqual(json.loads(report.read_text()), manifest)

    def test_conflicting_destination_preserves_both_copies(self):
        relative = "checkpoints/toy.ckpt"
        target = self.destination / relative
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing destination must survive")

        with self.assertRaisesRegex(RuntimeError, "Destination conflict"):
            migration.migrate(self.source, self.destination, self.job_id)

        self.assert_source_preserved()
        self.assertEqual(target.read_bytes(), b"existing destination must survive")

    def test_copy_failure_preserves_source_and_removes_partial_file(self):
        def interrupted_copy(original, partial):
            Path(partial).write_bytes(b"interrupted transfer")
            raise OSError("simulated copy failure")

        with patch.object(migration.shutil, "copyfile", side_effect=interrupted_copy):
            with self.assertRaisesRegex(OSError, "simulated copy failure"):
                migration.migrate(self.source, self.destination, self.job_id)

        self.assert_source_preserved()
        self.assertEqual(list(self.destination.rglob("*.part")), [])

    def test_corrupted_copy_is_rejected_before_cutover(self):
        real_copy = migration.shutil.copyfile

        def corrupt_copy(original, partial):
            real_copy(original, partial)
            with Path(partial).open("ab") as handle:
                handle.write(b"corruption")

        with patch.object(migration.shutil, "copyfile", side_effect=corrupt_copy):
            with self.assertRaisesRegex(RuntimeError, "Copy checksum failed"):
                migration.migrate(self.source, self.destination, self.job_id)

        self.assert_source_preserved()
        self.assertEqual(list(self.destination.rglob("*.part")), [])

    def test_rename_failure_preserves_original_source_directory(self):
        with patch.object(Path, "rename", side_effect=OSError("simulated rename failure")):
            with self.assertRaisesRegex(OSError, "simulated rename failure"):
                migration.migrate(self.source, self.destination, self.job_id)

        self.assert_source_preserved()
        for relative, payload in self.payloads.items():
            self.assertEqual((self.destination / relative).read_bytes(), payload)

    def test_already_migrated_receipt_is_idempotent_and_rechecks_hashes(self):
        migration.migrate(self.source, self.destination, self.job_id)
        report = self.destination / f"migration-{self.job_id}.json"
        original_report = report.read_bytes()

        with patch.object(migration.shutil, "copyfile", side_effect=AssertionError("must not copy")):
            with patch.object(migration, "inventory", side_effect=AssertionError("must not inventory")):
                with patch.object(migration, "sha256", wraps=migration.sha256) as rehash:
                    result = migration.migrate(self.source, self.destination, "retry_456")

        self.assertEqual(result["state"], "already_migrated")
        self.assertFalse(self.source.exists())
        self.assertFalse(self.source.is_symlink())
        self.assertEqual(report.read_bytes(), original_report)
        self.assertFalse((self.destination / "migration-retry_456.json").exists())
        self.assertEqual(
            {call.args[0] for call in rehash.call_args_list},
            {self.destination / relative for relative in self.payloads},
        )
        for relative, payload in self.payloads.items():
            self.assertEqual((self.destination / relative).read_bytes(), payload)

    def test_already_migrated_retry_refuses_corrupted_destination(self):
        migration.migrate(self.source, self.destination, self.job_id)
        target = self.destination / "checkpoints/toy.ckpt"
        payload = target.read_bytes()
        corrupted = bytes([payload[0] ^ 1]) + payload[1:]
        target.write_bytes(corrupted)

        with self.assertRaises(RuntimeError):
            migration.migrate(self.source, self.destination, "retry_456")

        self.assertFalse(self.source.exists())
        self.assertEqual(target.read_bytes(), corrupted)

    def test_source_symlink_is_rejected(self):
        source_link = self.root / "source-link"
        source_link.symlink_to(self.source, target_is_directory=True)

        with self.assertRaisesRegex(RuntimeError, "symlink"):
            migration.migrate(source_link, self.destination, self.job_id)

        self.assert_source_preserved()
        self.assertTrue(source_link.is_symlink())
        self.assertFalse(self.destination.exists())

    def test_source_modified_during_copy_is_not_removed(self):
        real_copy = migration.shutil.copyfile

        def copy_then_modify_source(original, partial):
            real_copy(original, partial)
            relative = Path(original).relative_to(self.source).as_posix()
            changed = self.payloads[relative] + b"new source data"
            Path(original).write_bytes(changed)
            self.payloads[relative] = changed

        with patch.object(migration.shutil, "copyfile", side_effect=copy_then_modify_source):
            with self.assertRaisesRegex(RuntimeError, "Source changed during migration"):
                migration.migrate(self.source, self.destination, self.job_id)

        self.assert_source_preserved()

    def test_report_name_conflicting_with_source_preserves_artifact(self):
        relative = f"migration-{self.job_id}.json"
        self.payloads[relative] = b"original report-named artifact"
        (self.source / relative).write_bytes(self.payloads[relative])

        with self.assertRaisesRegex(RuntimeError, "Migration report would conflict"):
            migration.migrate(self.source, self.destination, self.job_id)

        self.assert_source_preserved()
        self.assertFalse(self.destination.exists())

    def test_report_name_conflicting_with_destination_preserves_existing_file(self):
        self.destination.mkdir(parents=True)
        report = self.destination / f"migration-{self.job_id}.json"
        report.write_bytes(b"existing destination report")

        with self.assertRaisesRegex(RuntimeError, "Migration report would conflict"):
            migration.migrate(self.source, self.destination, self.job_id)

        self.assert_source_preserved()
        self.assertEqual(report.read_bytes(), b"existing destination report")
        self.assertEqual(list(self.destination.iterdir()), [report])

    def test_symlink_destination_directory_is_not_followed(self):
        self.destination.mkdir(parents=True)
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "preserve.txt"
        sentinel.write_bytes(b"unrelated outside data")
        (self.destination / "checkpoints").symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(RuntimeError, "Refusing a symlink destination directory"):
            migration.migrate(self.source, self.destination, self.job_id)

        self.assert_source_preserved()
        self.assertTrue((self.destination / "checkpoints").is_symlink())
        self.assertEqual(sentinel.read_bytes(), b"unrelated outside data")
        self.assertEqual(list(outside.iterdir()), [sentinel])

    def test_report_write_failure_restores_original_source(self):
        report = self.destination / f"migration-{self.job_id}.json"
        real_open = Path.open

        class InterruptedReportWriter:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                return self

            def __exit__(self, *exception):
                self.handle.close()

            def write(self, data):
                self.handle.write(data[:10])
                self.handle.flush()
                raise OSError("simulated report write failure")

        def fail_report_write(path, mode="r", *args, **kwargs):
            handle = real_open(path, mode, *args, **kwargs)
            if path == report and mode == "x":
                return InterruptedReportWriter(handle)
            return handle

        with patch.object(Path, "open", new=fail_report_write):
            with self.assertRaisesRegex(OSError, "simulated report write failure"):
                migration.migrate(self.source, self.destination, self.job_id)

        self.assert_source_preserved()
        for relative, payload in self.payloads.items():
            self.assertEqual((self.destination / relative).read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
