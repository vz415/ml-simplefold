"""Check assembly parsing and homomer chain boundaries without loading models."""
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
