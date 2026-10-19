"""Run ordinary EM sampling with shared inputs and small model forwards."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace


def run_baseline(cfg, paths, fasta, output):
    import torch
    import lightning.pytorch as pl
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/simplefold'))
    from inference import initialize_folding_model, initialize_esm_model, initialize_others
    from model.torch.fk_sampler import FKSampler
    from sample_complex_fk import expand_shared_batch
    from utils.datamodule_utils import process_one_inference_structure
    from utils.fasta_utils import check_fasta_inputs, download_fasta_utilities, process_fastas
    from utils.boltz_utils import process_structure, save_structure

    pl.seed_everything(cfg.seed, workers=True)
    args = SimpleNamespace(backend='torch', simplefold_model=cfg.model,
        ckpt_dir=str(paths['checkpoint_dir']), nsample_per_protein=1,
        num_steps=cfg.num_steps, tau=cfg.tau, plddt=False)
    model, device = initialize_folding_model(args)
    model.eval().requires_grad_(False).to('cpu')
    torch.cuda.empty_cache()
    esm, dictionary, mapping = initialize_esm_model(args, device)
    esm.eval().requires_grad_(False)
    tokenizer, featurizer, processor, flow, sampler = initialize_others(args, device)
    download_fasta_utilities(paths['ccd_dir'])
    data = check_fasta_inputs(fasta)
    if len(data) != 1:
        raise ValueError('Chunked baseline requires one multichain FASTA target')
    process_fastas(data=data, out_dir=output, ccd_path=paths['ccd_dir'] / 'ccd.pkl')
    struct_file, = sorted(output.glob('structures/*.npz'))
    with torch.no_grad():
        batch, structure, record = process_one_inference_structure(
            struct_file, output / 'records' / f'{struct_file.stem}.json',
            tokenizer, featurizer, processor, esm, dictionary, mapping)
    del esm
    torch.cuda.empty_cache()
    model.to(device)
    batch = expand_shared_batch(batch, cfg.samples)
    chunker = FKSampler(model_batch_size=cfg.sampling.model_batch_size)

    def model_forward(noised_pos, t, feats):
        return {'predict_velocity': chunker._velocity(model, noised_pos, t[0], feats)}

    # Only the existing velocity batching helper is reused from FK. The ordinary
    # EM sampler supplies all updates: no reward, selection, or resampling.
    print(f'Ordinary baseline: {cfg.samples} independent samples, model batch '
          f'{cfg.sampling.model_batch_size}; encoder released before sampling.', flush=True)
    with torch.no_grad():
        noise = torch.randn_like(batch['coords'])
        result = sampler.sample(model_forward, flow, noise, batch)
        result = processor.postprocess(result, batch)
    prediction_dir = output / f'predictions_{cfg.model}'
    prediction_dir.mkdir(exist_ok=True)
    for i in range(cfg.samples):
        converted = process_structure(deepcopy(structure), result['denoised_coords'][i],
                                      batch['atom_pad_mask'][i], record)
        save_structure(converted, prediction_dir, f'{record.id}_sampled_{i}', output_format='pdb')
