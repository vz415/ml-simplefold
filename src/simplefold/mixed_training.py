"""Replay a seeded general/synthetic training mixture with the upstream loss."""
from collections import Counter
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule

from multimer_training import MultimerFineTune
from processor.protein_processor import ProteinDataProcessor
from utils.datamodule_utils import collate


SOURCES = ('pdb', 'swissprot', 'synthetic')


class PlannedMixture(Dataset):
    """A random source/record schedule frozen before training, without shuffling."""

    def __init__(self, datasets, draws):
        self.datasets, self.draws = datasets, draws
        self.indices = {}
        for source, dataset in datasets.items():
            self.indices[source] = {
                str(record_id): index
                for index, (_, record_id) in enumerate(dataset.samples)
            }

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, index):
        draw = self.draws[index]
        source, record_id = draw['source'], draw['record_id']
        features = self.datasets[source][self.indices[source][record_id]]
        # The upstream loader may substitute after an error. That would change
        # our intended mixture, so require the exact prepared record here.
        assert features['record']['id'] == record_id, (draw, features['record']['id'])
        features['training_source'] = torch.tensor(SOURCES.index(source))
        features['training_plan_index'] = torch.tensor(index)
        return features


class MixtureDataModule(LightningDataModule):
    def __init__(self, datasets, draws):
        super().__init__()
        self.dataset = PlannedMixture(datasets, draws)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0,
                          collate_fn=collate, pin_memory=False)


class FixedESMProcessor(ProteinDataProcessor):
    """Compute fixed ESM2 features on demand, including newly drawn crops."""

    def __init__(self, device, **kwargs):
        super().__init__(device=device, **kwargs)
        from utils.esm_utils import esm_registry, _af2_to_esm
        self.encoder, self.alphabet = esm_registry['esm2_3B']()
        self.encoder = self.encoder.eval().requires_grad_(False).to(device)
        self.mapping = _af2_to_esm(self.alphabet).to(device)

    def preprocess_training(self, batch, **kwargs):
        batch = super().preprocess_training(batch)
        with torch.no_grad():
            self.process_esm(batch, self.encoder, self.alphabet, self.mapping)
        return batch

    def preprocess_inference(self, batch, **kwargs):
        batch = super().preprocess_inference(batch)
        with torch.no_grad():
            self.process_esm(batch, self.encoder, self.alphabet, self.mapping,
                             inference=True)
        return batch


class MixedFineTune(MultimerFineTune):
    """Train the folding network; save source-specific losses and actual draws."""

    def setup(self, stage):
        super().setup(stage)
        self.source_counts = Counter()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.model_ema.update_parameters(self.model)
        # Lightning increments global_step after this hook. Its checkpoint
        # callbacks run before module batch-end hooks, so update EMA here.
        self._ema_step = self.global_step + 1

    def training_step(self, batch, batch_idx):
        source = SOURCES[int(batch['training_source'][0])]
        index = int(batch['training_plan_index'][0])
        assert index == self.global_step, (index, self.global_step)
        loss = super().training_step(batch, batch_idx)
        assert torch.isfinite(loss), (source, index)
        self.source_counts[source] += 1
        self.log(f'loss/{source}', loss.detach(), on_step=True, on_epoch=False,
                 batch_size=1)
        receipt = dict(step=int(self.global_step), source=source,
                       record_id=batch['record'][0]['id'],
                       residues=int(batch['cropped_num_tokens'][0]),
                       loss=float(loss.detach().cpu()))
        with (Path(self.sample_dir).parent / 'training-draws.jsonl').open('a') as f:
            f.write(json.dumps(receipt, allow_nan=False) + '\n')
        return loss
