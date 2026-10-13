"""Behavior tests use synthetic files and temporary registries, never real jobs."""

import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts import bmp90_registry as registry


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec_path = self.root / "spec.json"
        # Independent of Vincent's downloaded plan or any real job artifacts.
        self.plan_path = self.root / "test-plan.json"
        jobs = [{"target_id": f"{ligand}_{type_i}_{type_ii}", "ligand": ligand,
                 "type_I": type_i, "type_II": type_ii, "status": "planned_not_sampled"}
                for ligand in ("BMP2", "BMP3", "BMP4", "BMP5", "BMP6", "BMP7", "BMP8A",
                               "BMP8B", "BMP9", "BMP10", "BMP15", "GDF5", "GDF6", "GDF7", "GDF10")
                for type_i in ("ACVR1", "BMPR1A") for type_ii in ("ACVR2A", "ACVR2B", "BMPR2")]
        self.plan_path.write_text(json.dumps({"stage": "synthetic_unit_test",
            "panel_status": "synthetic_unit_test", "jobs": jobs, "protocol": {"paired_conditions": [
                "unguided", "type_I_interface_guidance_only", "type_I_and_type_II_interface_guidance",
                "full_interface_guidance_plus_target_specific_geometry"]}}))
        genes = dict(zip("ABCDEF", ["BMP4", "BMP4", "BMPR1A", "BMPR1A", "ACVR2B", "ACVR2B"]))
        (self.root / "input.fasta").write_text("".join(">" + c + "|protein\nAC\n" for c in genes))
        mapping = {"chains": {c: {"gene": gene, "residues": [
            {"input_residue": p, "aa": aa, "source": "synthetic-test-fixture",
             "source_residue": p + 100} for p, aa in enumerate("AC", 1)]}
            for c, gene in genes.items()}}
        (self.root / "mapping.json").write_text(json.dumps(mapping))
        (self.root / "score.json").write_text(json.dumps({"version": "test-v1", "rules": []}))
        (self.root / "runner.py").write_text("# Synthetic fixture, never executed.\n")
        (self.root / "checkpoint.bin").write_bytes(b"synthetic fixture, not model weights")
        self.spec = {
            "target_id": "BMP4_BMPR1A_ACVR2B", "condition": "unguided",
            "scoring_version": "test-v1", "seed": 17, "replicate": 0,
            "sequences_fasta": "input.fasta", "residue_mapping_json": "mapping.json",
            "scoring_json": "score.json", "runner": "runner.py", "checkpoint": "checkpoint.bin",
            "command_argv": ["python", "runner.py"], "repository": "synthetic-test-repo",
            "code_revision": "a" * 40, "chain_genes": genes,
            "construct_handling": "synthetic fixture", "glycan_handling": "synthetic fixture",
            "disulfide_handling": "synthetic fixture",
            "resources": {"account": "fixture", "partition": "fixture", "gpu_type": "fixture",
                          "gpu_count": 1, "cpus": 2, "memory_gb": 4, "walltime_minutes": 10},
            "sampling": {"samples": 2, "particles": 2},
            "budget": {"max_gpu_hours": 0.2, "max_samples": 2, "max_particles": 2,
                       "scope": "synthetic unit test; never submit"},
        }
        self.save()

    def save(self):
        self.spec_path.write_text(json.dumps(self.spec))

    def test_fixture_passes_and_hashes_change_with_content(self):
        report = registry.preflight(self.spec_path, self.plan_path)
        self.assertTrue(report["ready"], report["issues"])
        old_hash = report["artifacts"]["checkpoint"]["sha256"]
        (self.root / "checkpoint.bin").write_bytes(b"changed synthetic fixture")
        self.assertNotEqual(old_hash, registry.preflight(self.spec_path, self.plan_path)["artifacts"]["checkpoint"]["sha256"])

    def test_recorded_checkpoint_identity_must_match_actual_bytes(self):
        self.spec["checkpoint_expected_sha256"] = registry.sha((self.root / "checkpoint.bin").read_bytes())
        self.save()
        self.assertTrue(registry.preflight(self.spec_path, self.plan_path)["ready"])
        (self.root / "checkpoint.bin").write_bytes(b"different synthetic checkpoint")
        report = registry.preflight(self.spec_path, self.plan_path)
        self.assertFalse(report["ready"])
        self.assertIn("checkpoint SHA256 differs from checkpoint_expected_sha256", report["issues"])

    def test_named_hydra_profile_requires_frozen_config_and_hashes_inputs(self):
        self.spec["hydra_experiment"] = "synthetic-profile"
        self.save()
        self.assertFalse(registry.preflight(self.spec_path, self.plan_path)["ready"])
        (self.root / "resolved.yaml").write_text("name: synthetic-profile\n")
        self.spec["resolved_config"] = "resolved.yaml"
        self.spec["auxiliary_inputs"] = {"synthetic_reference": "runner.py"}
        self.save()
        report = registry.preflight(self.spec_path, self.plan_path)
        self.assertTrue(report["ready"], report["issues"])
        self.assertIn("resolved_config", report["snapshots"])
        self.assertIn("auxiliary_inputs.synthetic_reference", report["artifacts"])

    def test_missing_prerequisites_reject_readiness(self):
        for key in ("sequences_fasta", "residue_mapping_json", "runner", "checkpoint", "budget"):
            with self.subTest(key=key):
                data = copy.deepcopy(self.spec)
                del data[key]
                self.spec_path.write_text(json.dumps(data))
                report = registry.preflight(self.spec_path, self.plan_path)
                self.assertFalse(report["ready"])
                self.assertTrue(any(key.split("_")[0] in issue for issue in report["issues"]), report["issues"])

    def test_active_unmapped_or_mismatched_residue_rejected(self):
        self.spec["condition"] = "type_I_interface_guidance_only"
        self.save()
        for ref in ({"chain": "Z", "input_residue": 1, "aa": "A"},
                    {"chain": "A", "input_residue": 99, "aa": "A"},
                    {"chain": "A", "input_residue": 1, "aa": "W"}):
            with self.subTest(ref=ref):
                score = {"version": "test-v1", "rules": [{"rule_id": "test-rule", "active": True,
                    "residue_based": True, "evidence": "synthetic fixture", "residues": [ref],
                    "weight": 1, "kind": "residue_contact", "interface": "type_I"}]}
                (self.root / "score.json").write_text(json.dumps(score))
                report = registry.preflight(self.spec_path, self.plan_path)
                self.assertFalse(report["ready"])
                self.assertIn("active residue rule references an unmapped/mismatched residue", report["issues"])

    def guided_rule(self):
        return {"rule_id": "test-rule", "active": True, "weight": 1, "kind": "residue_contact",
                "interface": "type_I", "residue_based": True, "evidence": "synthetic fixture",
                "residues": [{"chain": "A", "input_residue": 1, "aa": "A"},
                             {"chain": "C", "input_residue": 1, "aa": "A"}]}

    def check_guided(self, rules):
        self.spec["condition"] = "type_I_interface_guidance_only"
        self.save()
        (self.root / "score.json").write_text(json.dumps({"version": "test-v1", "rules": rules}))
        return registry.preflight(self.spec_path, self.plan_path)

    def test_guided_mapping_passes_but_dummy_inactive_or_zero_weight_rules_do_not(self):
        rule = self.guided_rule()
        report = self.check_guided([rule])
        self.assertTrue(report["ready"], report["issues"])
        mutations = [{"active": False}, {"weight": 0}, {"weight": -1}, {"weight": float("nan")},
                     {"weight": float("inf")}, {"mapping_status": "proposed"},
                     {"activation_status": "disabled"}, {"residues": []},
                     {"kind": "global_geometry", "residue_based": False, "residues": []},
                     {"kind": "unknown_kind"}]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                report = self.check_guided([{**rule, **mutation}])
                self.assertFalse(report["ready"], report["issues"])
        self.assertFalse(self.check_guided([rule, rule])["ready"])

    def test_condition_requires_both_interfaces_when_declared(self):
        self.check_guided([self.guided_rule()])
        self.spec["condition"] = "type_I_and_type_II_interface_guidance"
        self.save()
        report = registry.preflight(self.spec_path, self.plan_path)
        self.assertFalse(report["ready"])
        self.assertIn("guided condition lacks active rules for: type_II", report["issues"])

    def test_duplicate_gene_copies_must_have_equal_sequences(self):
        path = self.root / "input.fasta"
        path.write_text(path.read_text().replace(">B|protein\nAC", ">B|protein\nAD"))
        mapping_path = self.root / "mapping.json"
        mapping = json.loads(mapping_path.read_text())
        mapping["chains"]["B"]["residues"][1]["aa"] = "D"
        mapping_path.write_text(json.dumps(mapping))
        report = registry.preflight(self.spec_path, self.plan_path)
        self.assertFalse(report["ready"])
        self.assertIn("duplicate chains for BMP4 must have identical sequences", report["issues"])

    def test_incomplete_mapping_and_budget_overrun_rejected(self):
        mapping_path = self.root / "mapping.json"
        mapping = json.loads(mapping_path.read_text())
        mapping["chains"]["A"]["residues"].pop()
        mapping_path.write_text(json.dumps(mapping))
        self.spec["resources"]["walltime_minutes"] = 60
        self.save()
        report = registry.preflight(self.spec_path, self.plan_path)
        self.assertFalse(report["ready"])
        self.assertIn("incomplete residue mapping for chain A", report["issues"])
        self.assertIn("requested GPU walltime exceeds the recorded budget", report["issues"])

    def test_manifest_overwrite_protection_events_and_retry(self):
        first = registry.create_run(self.spec_path, self.root, self.plan_path)
        path = registry.run_dir(self.root, first["run_id"]) / "manifest.json"
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            registry.immutable_write(path, b"replacement")
        self.assertEqual(original, path.read_bytes())
        registry.append_event(self.root, first["run_id"], "submitted", "12345", "Synthetic test only",
                              "/synthetic/artifacts/runs/12345", "/synthetic/artifacts/remote-runs/12345")
        registry.append_event(self.root, first["run_id"], "completed", "12345", "Synthetic test only")
        with self.assertRaises(ValueError):
            registry.append_event(self.root, first["run_id"], "submitted", "99999")
        second = registry.create_run(self.spec_path, self.root, self.plan_path, retry_of=first["run_id"])
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["run_id"], second["retry_of"])
        self.assertEqual(original, path.read_bytes())
        rows = list(csv.DictReader(io.StringIO(registry.index_csv(self.root).decode())))
        self.assertEqual(["completed", "ready_for_submission"], [row["status"] for row in rows])
        self.assertEqual("12345", rows[0]["job_id"])
        self.assertEqual("/synthetic/artifacts/runs/12345", rows[0]["output_dir"])
        self.assertEqual("/synthetic/artifacts/remote-runs/12345", rows[0]["local_mirror_dir"])
        self.assertEqual("", rows[1]["job_id"])
        self.assertEqual(3, len(list((path.parent / "events").glob("*.json"))))

    def test_blocked_manifest_cannot_be_marked_launched(self):
        self.spec["checkpoint"] = None
        self.save()
        manifest = registry.create_run(self.spec_path, self.root, self.plan_path)
        self.assertEqual("prepared_blocked", manifest["status"])
        self.assertIsNone(manifest["job_id"])
        for status in ("ready_for_submission", "submitted", "running", "completed"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                registry.append_event(self.root, manifest["run_id"], status, "12345")

    def test_superseded_preparation_preserves_manifest_and_is_terminal(self):
        self.spec["checkpoint"] = None
        self.save()
        manifest = registry.create_run(self.spec_path, self.root, self.plan_path)
        path = registry.run_dir(self.root, manifest["run_id"]) / "manifest.json"
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            registry.append_event(self.root, manifest["run_id"], "superseded")
        registry.append_event(self.root, manifest["run_id"], "superseded", note="New target construct replaces this preparation")
        self.assertEqual(path.read_bytes(), original)
        rows = list(csv.DictReader(io.StringIO(registry.index_csv(self.root).decode())))
        self.assertEqual(rows[0]["status"], "superseded")
        self.assertEqual(rows[0]["job_id"], "")
        with self.assertRaises(ValueError):
            registry.append_event(self.root, manifest["run_id"], "prepared_blocked")

    def test_submitted_run_cannot_be_marked_superseded(self):
        manifest = registry.create_run(self.spec_path, self.root, self.plan_path)
        registry.append_event(self.root, manifest["run_id"], "submitted", "12345")
        with self.assertRaises(ValueError):
            registry.append_event(self.root, manifest["run_id"], "superseded", note="Replacement planned")

    def test_inventory_preserves_all_source_statuses_and_refuses_overwrite(self):
        registry.inventory(self.plan_path, self.root)
        data = registry.read_json(self.root / "target_inventory.json")
        plan, identity = registry.load_plan(self.plan_path)
        self.assertEqual(plan["jobs"], data["targets"])
        self.assertEqual(90, len(data["targets"]))
        self.assertEqual(identity, data["plan"])
        with self.assertRaises(FileExistsError):
            registry.inventory(self.plan_path, self.root)
        self.assertFalse((self.root / "runs").exists())


if __name__ == "__main__":
    unittest.main()
