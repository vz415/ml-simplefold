"""Load inference architecture weights when starting a new training run."""
from collections.abc import Mapping

import torch


def load_pretrained_folding_weights(model, checkpoint_path):
    """Warm-start current and EMA architectures without restoring training state."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("Pretrained folding weights must be a raw architecture state dict.")
    if "state_dict" in state:
        raise ValueError(
            "This is a Lightning training checkpoint. Use load_ckpt_path to resume "
            "training; pretrained_folding_ckpt_path requires raw folding architecture weights."
        )
    model.model.load_state_dict(state, strict=True)
    # Load the module directly: update_parameters would increment n_averaged.
    model.model_ema.module.load_state_dict(state, strict=True)
