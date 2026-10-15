"""Exercise storage boundaries without sampling, model imports or SSH."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_cached_sampling as launcher
import retrieve_results as retrieval


class StorageWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.environment = {"SLURM_JOB_ID": "12345", "SIMPLEFOLD_RESULT_CACHE_ROOT": str(self.root / "stage"),
                            "SIMPLEFOLD_ARTIFACT_DIR": str(self.root / "archive")}

    def fake_sampling(self, command):
        self.assertIn("tau=0.3", command)
        self.assertNotIn("paths.cache_dir=", " ".join(command))
        run = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("hydra.run.dir=")))
        run.mkdir(parents=True)
        (run / "experiment.json").write_text('{"job_id":"12345"}\n')
        return 0

    def test_pub_unavailable_retains_complete_home_results(self):
        with patch.dict(launcher.os.environ, self.environment, clear=True), \
             patch.object(launcher.subprocess, "call", side_effect=self.fake_sampling), \
             patch.object(launcher, "mirror_run", side_effect=subprocess.TimeoutExpired("rsync", 120)):
            self.assertEqual(launcher.main(["tau=0.3"]), 0)
        run = self.root / "stage/runs/12345"
        self.assertTrue((run / "experiment.json").is_file())
        receipt = json.loads((run / "RESULT_STORAGE.json").read_text())
        self.assertEqual(receipt["sampling_exit_code"], 0)
        self.assertFalse((run / "CACHE_MANIFEST.json").exists())
        # Later evaluation is allowed before final sealing.
        (run / "evaluation").mkdir()

    def test_failed_sampling_retains_partial_output_without_publish(self):
        def fail(command):
            self.fake_sampling(command)
            return 2
        with patch.dict(launcher.os.environ, self.environment, clear=True), \
             patch.object(launcher.subprocess, "call", side_effect=fail), \
             patch.object(launcher, "mirror_run") as mirror:
            self.assertEqual(launcher.main(["tau=0.3"]), 2)
            mirror.assert_not_called()

    def test_paths_keep_weights_canonical_and_reject_ambiguous_overrides(self):
        run, canonical = launcher.launch_paths([], self.environment)
        self.assertEqual(run, self.root / "stage/runs/12345")
        self.assertEqual(canonical, self.root / "archive/runs/12345")
        for arguments in (["hydra.run.dir=/tmp/custom"], ["hydra.sweep.dir=/tmp/custom"],
                          ["paths.artifact_dir=relative"]):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                launcher.launch_paths(arguments, self.environment)
        for root in ("/pub/cache", "/dfs6b/pub/cache", str(self.root / "archive")):
            with self.subTest(root=root), self.assertRaises(ValueError):
                launcher.launch_paths([], dict(self.environment, SIMPLEFOLD_RESULT_CACHE_ROOT=root))

    def test_inspection_does_not_create_storage_or_require_slurm(self):
        with patch.dict(launcher.os.environ, {}, clear=True), \
             patch.object(launcher.subprocess, "call", return_value=0) as invoke:
            self.assertEqual(launcher.main(["--cfg", "job"]), 0)
            self.assertEqual(invoke.call_args.args[0][-2:], ["--cfg", "job"])

    def test_pending_missing_failed_or_unknown_writers_block_retrieval(self):
        retrieval.check_completed("12345|COMPLETED|0:0\n12346|COMPLETED|0:0\n", ["12345", "12346"])
        for output in ("12345|RUNNING|0:0", "12345|FAILED|1:0", "", "12345|COMPLETED|0:0"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                retrieval.check_completed(output, ["12345", "12346"])

    def test_missing_home_evaluation_or_wrong_archive_prevents_sealing(self):
        run = self.root / "stage/runs/12345"
        run.mkdir(parents=True)
        canonical = str(self.root / "archive/runs/12345")
        receipt = dict(job_id="12345", sampling_exit_code=0,
                       temporary_run=str(run), canonical_run=canonical + "-wrong")
        (run / "RESULT_STORAGE.json").write_text(json.dumps(receipt))
        request = dict(module=Path(retrieval.cache.__file__).read_text(), action="prepare",
                       cache_root=str(self.root / "stage"), job_id="12345", canonical=canonical,
                       evaluation_jobs=[])
        with patch.object(sys, "stdin", io.StringIO(json.dumps(request))), self.assertRaises(ValueError):
            exec(retrieval.REMOTE_DRIVER, {})
        receipt["canonical_run"] = canonical
        (run / "RESULT_STORAGE.json").write_text(json.dumps(receipt))
        request["evaluation_jobs"] = ["12346"]
        with patch.object(sys, "stdin", io.StringIO(json.dumps(request))), self.assertRaises(FileNotFoundError):
            exec(retrieval.REMOTE_DRIVER, {})
        self.assertFalse((run / "CACHE_MANIFEST.json").exists())


if __name__ == "__main__":
    unittest.main()
