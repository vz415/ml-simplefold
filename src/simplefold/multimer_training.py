"""Small-dataset adaptation using the upstream SimpleFold flow-matching loss."""
from pathlib import Path
import torch
from processor.protein_processor import ProteinDataProcessor
from model.simplefold import SimpleFold


class CachedESMProcessor(ProteinDataProcessor):
    """Read fixed, sequence-keyed ESM2 representations; no encoder in training."""
    def __init__(self, embedding_dir, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dir = Path(embedding_dir)
        self.cache = {}

    def add_embeddings(self, batch):
        import hashlib
        key = hashlib.sha256(batch['aa_seq'][0].encode()).hexdigest()
        if key not in self.cache:
            self.cache[key] = torch.load(self.embedding_dir / f'{key}.pt', map_location='cpu', weights_only=True)
        batch['esm_s'] = self.cache[key].to(self.device).expand(batch['coords'].shape[0], -1, -1, -1)
        return batch

    def preprocess_training(self, batch, **kwargs):
        return self.add_embeddings(super().preprocess_training(batch))

    def preprocess_inference(self, batch, **kwargs):
        return self.add_embeddings(super().preprocess_inference(batch))


class MultimerFineTune(SimpleFold):
    """Keep upstream loss; optimize folding weights only and update EMA post-step."""
    def setup(self, stage):
        self.processor = self.hparams.processor(device=self.device)
        self.training_gpus = 1
        self._ema_step = 0

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.model.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def on_before_zero_grad(self, optimizer):
        pass  # Native hook updates before the optimizer step; update below instead.

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step > self._ema_step:
            self.model_ema.update_parameters(self.model)
            self._ema_step = self.global_step
        super().on_train_batch_end(outputs, batch, batch_idx)
