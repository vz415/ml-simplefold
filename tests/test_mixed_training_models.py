"""Model-size propagation and recomputation must preserve the intended experiment."""
import copy
from pathlib import Path
import sys
import unittest

import hydra
from omegaconf import OmegaConf
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src/simplefold'), str(ROOT / 'scripts')]
from model.torch.blocks import HomogenTrunk
from train_mixed import retention_config


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.drop = nn.Dropout(0.2)

    def forward(self, latents, c, layer_idx=0):
        return self.drop(self.linear(latents)) + c[:, None, :] + layer_idx


class MixedModelTests(unittest.TestCase):
    def test_3b_benchmark_uses_3b_weights_and_architecture(self):
        with hydra.initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base='1.3'):
            small = hydra.compose(config_name='mixed_train', overrides=['job_id=test'])
            large = hydra.compose(config_name='mixed_train_3b', overrides=['job_id=test'])
            for key in ['seed', 'max_steps', 'synthetic_probability', 'dataset_dir', 'general_dataset_dir', 'selection']:
                self.assertEqual(small[key], large[key])
            self.assertEqual(OmegaConf.to_container(small.evaluation), OmegaConf.to_container(large.evaluation))
            self.assertEqual(large.model.architecture.hidden_size, 2048)
            self.assertEqual(large.model.architecture.trunk.depth, 36)
            benchmark = retention_config(large, 'pinned-3b')
            self.assertEqual(benchmark.model_name, '3B')
            self.assertEqual(benchmark.model.architecture.hidden_size, 2048)
            self.assertEqual(benchmark.model.architecture.trunk.depth, 36)
            self.assertTrue(benchmark.checkpoints.pretrained.path.endswith('simplefold_3B.ckpt'))
            self.assertEqual(benchmark.checkpoints.pretrained.sha256, 'pinned-3b')
            self.assertTrue(benchmark.checkpoints.fine_tuned.path.endswith('simplefold_3B_ema.ckpt'))
            self.assertFalse(large.export_raw)
            self.assertFalse(large.save_full_final)
            self.assertEqual(large.checkpoint_every, 0)

    def test_activation_checkpointing_preserves_outputs_gradients_and_state_keys(self):
        torch.manual_seed(4)
        ordinary = HomogenTrunk(TinyBlock, 3).train()
        checkpointed = copy.deepcopy(ordinary)
        checkpointed.activation_checkpointing = True
        x = torch.randn(2, 5, 4, requires_grad=True)
        c = torch.randn(2, 4, requires_grad=True)
        torch.manual_seed(7)
        expected = ordinary(x, c)
        expected.sum().backward()
        gradient = x.grad.clone()
        x2, c2 = x.detach().clone().requires_grad_(), c.detach().clone().requires_grad_()
        torch.manual_seed(7)
        actual = checkpointed(x2, c2)
        actual.sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x2.grad, gradient)
        torch.testing.assert_close(c2.grad, c.grad)
        for a, b in zip(ordinary.parameters(), checkpointed.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        self.assertEqual(ordinary.state_dict().keys(), checkpointed.state_dict().keys())
        ordinary.load_state_dict(checkpointed.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
