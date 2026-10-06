"""Validate warm-start loading with tiny linear layers, without protein models."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "simplefold" / "utils" / "training_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("training_checkpoint", MODULE_PATH)
checkpoint_loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint_loader)


class TrainingCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.checkpoint = Path(temporary.name) / "toy.ckpt"
        architecture = nn.Linear(3, 2)
        with torch.no_grad():
            architecture.weight.zero_()
            architecture.bias.zero_()
        self.model = SimpleNamespace(
            model=architecture,
            model_ema=AveragedModel(architecture),
        )

    def test_raw_weights_initialize_current_and_ema_without_training_state(self):
        saved_architecture = nn.Linear(3, 2)
        with torch.no_grad():
            saved_architecture.weight.copy_(torch.arange(6).reshape(2, 3))
            saved_architecture.bias.copy_(torch.tensor([7.0, 8.0]))
        expected = saved_architecture.state_dict()
        torch.save(expected, self.checkpoint)
        optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=0.001)

        checkpoint_loader.load_pretrained_folding_weights(self.model, self.checkpoint)

        for key, value in expected.items():
            torch.testing.assert_close(self.model.model.state_dict()[key], value)
            torch.testing.assert_close(self.model.model_ema.module.state_dict()[key], value)
        self.assertEqual(self.model.model_ema.n_averaged.item(), 0)
        self.assertEqual(optimizer.state_dict()["state"], {})
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)

    def test_lightning_checkpoint_is_rejected_with_resume_guidance(self):
        torch.save({"state_dict": nn.Linear(3, 2).state_dict(), "epoch": 2}, self.checkpoint)

        with self.assertRaisesRegex(ValueError, "load_ckpt_path"):
            checkpoint_loader.load_pretrained_folding_weights(self.model, self.checkpoint)

        self.assertEqual(self.model.model_ema.n_averaged.item(), 0)
        for module in (self.model.model, self.model.model_ema.module):
            self.assertEqual(torch.count_nonzero(module.weight).item(), 0)
            self.assertEqual(torch.count_nonzero(module.bias).item(), 0)

    def test_architecture_shape_mismatch_is_rejected(self):
        torch.save(nn.Linear(4, 2).state_dict(), self.checkpoint)

        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            checkpoint_loader.load_pretrained_folding_weights(self.model, self.checkpoint)

        self.assertEqual(self.model.model_ema.n_averaged.item(), 0)
        self.assertEqual(torch.count_nonzero(self.model.model_ema.module.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(self.model.model_ema.module.bias).item(), 0)


if __name__ == "__main__":
    unittest.main()
