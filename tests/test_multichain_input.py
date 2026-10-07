"""Check assembly parsing and homomer chain boundaries without loading models."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "simplefold"))
from utils.fasta_utils import parse_fasta
from utils.datamodule_utils import extract_sequence_from_tokens
from boltz_data_pipeline.types import Token
from boltz_data_pipeline import const


class MultichainInputTests(unittest.TestCase):
    def parse_schema(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.fasta"
            path.write_text(text)
            with patch("utils.fasta_utils.parse_boltz_schema", side_effect=lambda name, data, ccd: data):
                return parse_fasta(path, {})

    def test_single_record_keeps_chain_a(self):
        schema = self.parse_schema(">protein_description\nACG\n")
        self.assertEqual(schema["sequences"][0]["protein"]["id"], "A")

    def test_multiple_records_keep_distinct_chain_ids(self):
        schema = self.parse_schema(">A|protein\nACG\n>B|protein\nACG\n>C|protein\nWW\n")
        self.assertEqual([s["protein"]["id"] for s in schema["sequences"]], ["A", "B", "C"])

    def test_duplicate_chain_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.parse_schema(">A|protein\nACG\n>A|protein\nWW\n")

    def test_empty_fasta_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Empty FASTA"):
            self.parse_schema("")

    def test_identical_adjacent_entities_still_have_chain_boundary(self):
        tokens = np.zeros(6, dtype=Token)
        tokens["asym_id"] = [0, 0, 1, 1, 2, 2]
        tokens["entity_id"] = [0, 0, 0, 0, 1, 1]
        tokens["res_type"] = [const.tokens.index(name) for name in ["ALA", "CYS", "ALA", "CYS", "TRP", "TRP"]]
        self.assertEqual(extract_sequence_from_tokens(SimpleNamespace(tokens=tokens)), "AC:AC:WW")

    def validate_toy_prediction(self, chains):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "toy.fasta"
            fasta.write_text(">A|protein\nAC\n>B|protein\nWW\n>C|protein\nAC\n")
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            lines = []
            names = {"A": "ALA", "C": "CYS", "W": "TRP"}
            serial = 1
            for chain_id, sequence in chains:
                for i, letter in enumerate(sequence, 1):
                    lines.append(f"ATOM  {serial:5d}  CA  {names[letter]} {chain_id}{i:4d}    {float(serial):8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{0.:6.2f}           C\n")
                    serial += 1
                lines.append("TER\n")
            lines.append("END\n")
            (prediction_dir / "toy_sampled_0.pdb").write_text("".join(lines))
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "validate_predictions.py"),
                       "--fasta_path", str(fasta), "--prediction_dir", str(prediction_dir)]
            result = subprocess.run(command, capture_output=True, text=True)
            report = json.loads((root / "validation.json").read_text()) if result.returncode == 0 else None
            return result, report

    def test_validator_accepts_schema_grouping_and_records_mapping(self):
        result, report = self.validate_toy_prediction([("A", "AC"), ("B", "AC"), ("C", "WW")])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report[0]["saved_chain_to_input_chain"], {"A": "A", "B": "C", "C": "B"})

    def test_validator_rejects_fused_chains(self):
        result, _ = self.validate_toy_prediction([("A", "ACACWW")])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Chain count/order/sequence mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
