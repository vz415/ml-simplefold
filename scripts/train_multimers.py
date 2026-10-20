#!/usr/bin/env python3
"""Slurm-only all-complex 100M capacity pilot, followed by matched unguided samples."""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src/simplefold'), str(ROOT / 'scripts')]
import hydra
from omegaconf import OmegaConf


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


@hydra.main(version_base='1.3', config_path='../configs', config_name='multimer_train')
def main(cfg):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Training and inference run only inside Slurm')
    import numpy as np
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from prepare_multimer_training import prepare
    from processor.protein_processor import ProteinDataProcessor
    from utils.esm_utils import esm_registry, _af2_to_esm
    from utils.datamodule_utils import collate
    from utils.training_checkpoint import load_pretrained_folding_weights
    from utils.boltz_utils import process_structure, save_structure
    from boltz_data_pipeline.types import Record, Structure

    torch.set_float32_matmul_precision('medium')
    assert torch.cuda.is_available(), 'CUDA is required'
    output, dataset, trained = map(Path, [cfg.output_dir, cfg.dataset_dir, cfg.trained_dir])
    output.mkdir(parents=True, exist_ok=True); trained.mkdir(parents=True, exist_ok=True)
    if (output/'experiment.json').exists():
        raise RuntimeError('Refusing to overwrite an existing experiment')
    os.environ['TORCH_HOME'] = str(Path(cfg.artifact_root)/'torch')
    (output/'resolved-config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (output/'git-commit.txt').write_text(commit+'\n')
    write_json(output/'experiment.json',dict(job_id=cfg.job_id,source_commit=commit,
        mode='100M all-training synthetic capacity pilot',encoder='fixed cached ESM2-3B',
        validation='No held-out set; before/after unguided reconstruction diagnostics only',
        evaluation=OmegaConf.to_container(cfg.evaluation,resolve=True),device=torch.cuda.get_device_name(0)))
    pl.seed_everything(cfg.seed, workers=True)
    receipts = prepare(ROOT/cfg.selection, Path(cfg.source_dir), dataset, Path(cfg.artifact_root)/'ccd/ccd.pkl')
    write_json(output/'dataset-receipt.json', json.loads((dataset/'dataset-receipt.json').read_text()))
    data = hydra.utils.instantiate(cfg.data)
    # Inspect every record explicitly: never let the loader silently replace a failed complex.
    features = []
    for i,item in enumerate(receipts):
        f = data._train_set[i]
        assert f['record']['id'] == item['id']
        assert int(f['cropped_num_tokens']) == item['residues']
        features.append(f)
    print('Full-assembly dataset preflight passed:', [(r['id'],r['residues']) for r in receipts], flush=True)

    encoder, alphabet = esm_registry['esm2_3B']()
    encoder = encoder.eval().requires_grad_(False).cuda()
    mapping = _af2_to_esm(alphabet).cuda()
    processor = ProteinDataProcessor(device='cuda', multiplicity=1, inference_multiplicity=1)
    embedding_receipts = []
    with torch.no_grad():
        for f in features:
            batch = collate([f]); key = hashlib.sha256(batch['aa_seq'][0].encode()).hexdigest()
            path = dataset/'esm'/f'{key}.pt'
            # Regenerate in this job; cache is a dataset artifact, not a moving encoder.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                processor.process_esm(batch, encoder, alphabet, mapping)
            torch.save(batch['esm_s'].detach().cpu(), path)
            embedding_receipts.append(dict(sequence_sha256=key,file=path.name,sha256=digest(path)))
            del batch
    del encoder, mapping, processor
    gc.collect(); torch.cuda.empty_cache()
    write_json(output/'embedding-receipt.json',dict(encoder_sha256=digest(Path(cfg.artifact_root)/'torch/hub/checkpoints/esm2_t36_3B_UR50D.pt'),features=embedding_receipts))

    model = hydra.utils.instantiate(cfg.model)
    load_pretrained_folding_weights(model, cfg.pretrained_checkpoint)
    model.model_ema.requires_grad_(False)
    checkpoint = ModelCheckpoint(dirpath=trained,filename='step-{step:06d}',every_n_train_steps=cfg.checkpoint_every,
        save_top_k=-1,save_last=True,save_on_train_epoch_end=False)
    trainer = pl.Trainer(accelerator='gpu',devices=1,strategy='auto',precision='bf16-mixed',
        max_steps=cfg.max_steps,max_epochs=-1,limit_val_batches=0,num_sanity_val_steps=0,
        logger=CSVLogger(output/'logs',name='training'),callbacks=[checkpoint],log_every_n_steps=1,
        enable_model_summary=False)
    trainer.fit(model, datamodule=data)
    trainer.save_checkpoint(trained/'final.ckpt')
    for label,weights in [('raw',model.model),('ema',model.model_ema.module)]:
        torch.save({k:v.detach().cpu() for k,v in weights.state_dict().items()},trained/f'simplefold_100M_{label}.ckpt')
    write_json(output/'training-summary.json',dict(global_step=trainer.global_step,records=len(receipts),
        pretrained_sha256=digest(cfg.pretrained_checkpoint),ema_checkpoint=str(trained/'simplefold_100M_ema.ckpt'),
        raw_checkpoint=str(trained/'simplefold_100M_raw.ckpt'),ema_sha256=digest(trained/'simplefold_100M_ema.ckpt'),
        ema_decay=cfg.model.ema_decay,encoder_trainable=False,training_only=True))

    # Common initial noise and Brownian draws, without guidance, for both weight sets.
    model.to('cuda').eval(); model.processor = hydra.utils.instantiate(cfg.model.processor)(device='cuda')
    data._train_set.rotation_augment_coords = False
    sampler = hydra.utils.instantiate(cfg.model.sampler, num_timesteps=cfg.evaluation.steps,
        tau=cfg.evaluation.tau,log_timesteps=True)
    from lightning.fabric.utilities.apply_func import move_data_to_device
    predictions=[]
    for condition,weight_path in [('pretrained',Path(cfg.pretrained_checkpoint)),('fine_tuned',trained/'simplefold_100M_ema.ckpt')]:
        model.model.load_state_dict(torch.load(weight_path,map_location='cpu',weights_only=True),strict=True)
        folder=output/condition/'predictions_simplefold_100M';folder.mkdir(parents=True,exist_ok=True)
        for i,item in enumerate(receipts):
            for sample in range(cfg.evaluation.samples):
                seed=cfg.evaluation.seed+i*100+sample
                pl.seed_everything(seed,workers=True)
                batch=collate([data._train_set[i]])
                batch=move_data_to_device(batch,torch.device('cuda'))
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    batch=model.processor.preprocess_inference(batch)
                    out=sampler.sample(model.model.forward,model.path,torch.randn_like(batch['coords']),batch)
                    assert torch.isfinite(out['denoised_coords']).all()
                    out=model.processor.postprocess(out,batch)
                structure=Structure.load(dataset/'structures'/(item['id']+'.npz'))
                record=Record(**batch['record'][0])
                predicted=process_structure(structure,out['denoised_coords'][0].float(),batch['atom_pad_mask'][0],record)
                name=f'{item["id"]}_sampled_{sample}'
                save_structure(predicted,folder,name,output_format='pdb')
                path=folder/(name+'.pdb')
                predictions.append(dict(condition=condition,id=item['id'],sample=sample,seed=seed,
                    file=str(path.relative_to(output)),sha256=digest(path)))
                write_json(output/'prediction-receipt.json',predictions)
                print('Saved',condition,name,flush=True)
    write_json(output/'completion.json',dict(state='completed',samples=len(predictions),training_steps=trainer.global_step))


if __name__ == '__main__':
    main()
