#!/usr/bin/env python3
"""Fine-tune original100M on a frozen random mixture; all compute is Slurm-only."""
import copy
from collections import Counter
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src/simplefold'), str(ROOT / 'scripts')]
import hydra
from omegaconf import OmegaConf
from train_multimers import digest, write_json


def datasets(cfg):
    """Use native featurization, 512-residue general crops and whole complexes."""
    data = {}
    for source in ['pdb', 'swissprot', 'synthetic']:
        spec = copy.deepcopy(cfg.data)
        if source != 'synthetic':
            directory = Path(cfg.general_dataset_dir) / source
            item = spec.datasets[0]
            item.data_name = source
            item.tokenized_dir = str(directory)
            item.target_dir = str(directory)
            item.manifest_path = str(directory / 'manifest.json')
            item.record_list = str(directory / 'train.txt')
            item.cropper = dict(_target_='boltz_data_pipeline.crop.boltz.BoltzCropper',
                                min_neighborhood=0, max_neighborhood=40)
            spec.max_tokens = cfg.general.max_tokens
            spec.max_atoms = cfg.general.max_atoms
        data[source] = hydra.utils.instantiate(spec)
    return data


def evaluate_complexes(cfg, model, synthetic, output):
    """Matched seeds/settings against the same six synthetic training targets."""
    import numpy as np
    import torch
    import lightning.pytorch as pl
    from boltz_data_pipeline.types import Record, Structure
    from lightning.fabric.utilities.apply_func import move_data_to_device
    from utils.datamodule_utils import collate
    from utils.boltz_utils import process_structure, save_structure
    model.to('cuda').eval()
    model.processor.device = torch.device('cuda')
    model.processor.encoder.to('cuda')
    model.processor.mapping = model.processor.mapping.to('cuda')
    synthetic.rotation_augment_coords = False
    sampler = hydra.utils.instantiate(cfg.model.sampler,
        num_timesteps=cfg.evaluation.steps, tau=cfg.evaluation.tau, log_timesteps=True)
    predictions = []
    trained = Path(cfg.trained_dir) / 'simplefold_100M_ema.ckpt'
    selection = json.loads((ROOT / cfg.selection).read_text())['records']
    lookup = {record_id: index for index, (_, record_id) in enumerate(synthetic.samples)}
    for condition, weights in [('pretrained', Path(cfg.pretrained_checkpoint)),
                               ('fine_tuned', trained)]:
        model.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        folder = output / condition / 'predictions_simplefold_100M'
        folder.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(selection):
            for sample in range(cfg.evaluation.samples):
                seed = cfg.evaluation.seed + index * 100 + sample
                pl.seed_everything(seed, workers=True)
                batch = move_data_to_device(collate([synthetic[lookup[item['id']]]]), torch.device('cuda'))
                assert batch['record'][0]['id'] == item['id']
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    batch = model.processor.preprocess_inference(batch)
                    out = sampler.sample(model.model.forward, model.path,
                                         torch.randn_like(batch['coords']), batch)
                    assert torch.isfinite(out['denoised_coords']).all()
                    out = model.processor.postprocess(out, batch)
                structure = Structure.load(Path(cfg.dataset_dir) / 'structures' / (item['id'] + '.npz'))
                predicted = process_structure(structure, out['denoised_coords'][0].float(),
                                              batch['atom_pad_mask'][0], Record(**batch['record'][0]))
                name = f'{item["id"]}_sampled_{sample}'
                save_structure(predicted, folder, name, output_format='pdb')
                path = folder / (name + '.pdb')
                predictions.append(dict(condition=condition, id=item['id'], sample=sample,
                    seed=seed, file=str(path.relative_to(output)), sha256=digest(path)))
                write_json(output / 'prediction-receipt.json', predictions)
                print('Saved', condition, name, flush=True)
    return predictions


@hydra.main(version_base='1.3', config_path='../configs', config_name='mixed_train')
def main(cfg):
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm for preparation, training and inference'
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'experiment.json').exists(), 'Use a fresh run directory'
    (output / 'resolved-config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    (output / 'git-commit.txt').write_text(commit + '\n')
    write_json(output / 'experiment.json', dict(job_id=cfg.job_id, phase=cfg.phase,
        source_commit=commit, seed=cfg.seed, max_steps=cfg.max_steps,
        synthetic_probability=cfg.synthetic_probability,
        checkpoint=str(cfg.pretrained_checkpoint), encoder='frozen ESM2-3B',
        scope='Random general PDB/SwissProt replay mixed with six synthetic training complexes'))
    if cfg.phase == 'prepare':
        subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_general_training.py'),
            '--config', str(ROOT / cfg.general_data_config), '--dataset-dir', cfg.general_dataset_dir,
            '--synthetic-selection', str(ROOT / cfg.selection), '--steps', str(cfg.max_steps),
            '--synthetic-probability', str(cfg.synthetic_probability), '--seed', str(cfg.seed),
            '--ccd', str(Path(cfg.artifact_root) / 'ccd/ccd.pkl')], check=True, cwd=ROOT)
        for name in ['training-plan.json', 'dataset-receipt.json']:
            shutil.copy2(Path(cfg.general_dataset_dir) / name, output / name)
        write_json(output / 'completion.json', dict(state='prepared'))
        return

    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from mixed_training import MixtureDataModule
    from utils.training_checkpoint import load_pretrained_folding_weights
    assert digest(cfg.pretrained_checkpoint) == cfg.pretrained_sha256, 'Original released checkpoint required'
    os.environ['TORCH_HOME'] = str(Path(cfg.artifact_root) / 'torch')
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg.seed, workers=True)
    plan_path = Path(cfg.general_dataset_dir) / 'training-plan.json'
    plan = json.loads(plan_path.read_text())
    receipt = json.loads((Path(cfg.general_dataset_dir) / 'dataset-receipt.json').read_text())
    assert digest(plan_path) == receipt['training_plan_sha256']
    assert digest(ROOT / cfg.selection) == receipt['inputs']['synthetic_selection_sha256']
    assert digest(ROOT / cfg.general_data_config) == receipt['inputs']['config_sha256']
    locations = {'structure.npz': ('structures', '.npz'), 'record.json': ('records', '.json'),
                 'tokens.pkl': ('tokens', '.pkl'), 'sequence.fasta': ('fasta', '.fasta')}
    for record in receipt['records']:
        for name, (folder, extension) in locations.items():
            path = Path(cfg.general_dataset_dir) / record['source'] / folder / (record['id'] + extension)
            assert digest(path) == record['files'][name], path
    for item in json.loads((ROOT / cfg.selection).read_text())['records']:
        assert digest(Path(cfg.dataset_dir) / 'source' / item['file']) == item['sha256']
    assert plan['seed'] == cfg.seed and plan['synthetic_probability'] == cfg.synthetic_probability
    draws = plan['draws']
    assert len(draws) == cfg.max_steps
    for name in ['training-plan.json', 'dataset-receipt.json']:
        shutil.copy2(Path(cfg.general_dataset_dir) / name, output / name)
    modules = datasets(cfg)
    data = MixtureDataModule({key: module._train_set for key, module in modules.items()}, draws)
    # Check data paths and sampled record identity before reserving model memory.
    for index in range(len(data.dataset)):
        features = data.dataset[index]
        if draws[index]['source'] == 'synthetic':
            assert features['cropped_num_tokens'] == features['max_num_tokens']
        else:
            assert int(features['cropped_num_tokens']) <= cfg.general.max_tokens
    print('Mixture preflight passed:', dict(Counter(draw['source'] for draw in draws)), flush=True)
    if cfg.phase == 'preflight':
        write_json(output / 'completion.json', dict(state='preflight_passed', draws=len(draws)))
        return

    assert torch.cuda.is_available(), 'CUDA required'
    trained = Path(cfg.trained_dir)
    trained.mkdir(parents=True, exist_ok=True)
    model = hydra.utils.instantiate(cfg.model)
    load_pretrained_folding_weights(model, cfg.pretrained_checkpoint)
    model.model_ema.requires_grad_(False)
    checkpoints = ModelCheckpoint(dirpath=trained, filename='step-{step:06d}',
        every_n_train_steps=cfg.checkpoint_every, save_top_k=-1, save_last=True,
        save_on_train_epoch_end=False)
    trainer = pl.Trainer(accelerator='gpu', devices=1, strategy='auto', precision='bf16-mixed',
        max_steps=cfg.max_steps, max_epochs=1, limit_val_batches=0, num_sanity_val_steps=0,
        logger=CSVLogger(output / 'logs', name='training'), callbacks=[checkpoints],
        log_every_n_steps=1, enable_model_summary=False)
    trainer.fit(model, datamodule=data)
    assert trainer.global_step == cfg.max_steps
    assert dict(model.source_counts) == dict(Counter(draw['source'] for draw in draws))
    trainer.save_checkpoint(trained / 'final.ckpt')
    for label, weights in [('raw', model.model), ('ema', model.model_ema.module)]:
        torch.save({key: value.detach().cpu() for key, value in weights.state_dict().items()},
                   trained / f'simplefold_100M_{label}.ckpt')
    del weights
    write_json(output / 'training-summary.json', dict(global_step=trainer.global_step,
        source_counts=dict(model.source_counts), synthetic_probability=cfg.synthetic_probability,
        realized_synthetic_fraction=model.source_counts['synthetic'] / trainer.global_step,
        training_plan_sha256=digest(plan_path), pretrained_sha256=digest(cfg.pretrained_checkpoint),
        ema_checkpoint=str(trained / 'simplefold_100M_ema.ckpt'),
        ema_sha256=digest(trained / 'simplefold_100M_ema.ckpt'),
        ema_decay=cfg.model.ema_decay, encoder_trainable=False,
        validation='Complex reconstruction uses training targets; CAMEO retention is separate'))
    source_counts = dict(model.source_counts)
    predictions = evaluate_complexes(cfg, model, modules['synthetic']._train_set, output)
    if cfg.retention.enabled:
        # Release the training model and its separate frozen encoder before the
        # existing benchmark runner loads its own inference-only networks.
        del model.processor.encoder
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
        from run_cameo_benchmark import infer, score
        benchmark = hydra.compose(config_name='cameo_benchmark')
        benchmark.artifact_root = cfg.artifact_root
        benchmark.output_dir = str(output / 'retention')
        benchmark.checkpoints.fine_tuned.path = str(trained / 'simplefold_100M_ema.ckpt')
        benchmark.checkpoints.fine_tuned.sha256 = digest(trained / 'simplefold_100M_ema.ckpt')
        destination = Path(benchmark.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'resolved-config.yaml').write_text(OmegaConf.to_yaml(benchmark, resolve=True))
        write_json(destination / 'experiment.json', dict(parent_job=cfg.job_id,
            source_commit=commit, purpose='Fixed24-target CAMEO22 retention after mixed training',
            selection_sha256=digest(ROOT / benchmark.selection),
            sampling=OmegaConf.to_container(benchmark.sampling, resolve=True),
            checkpoints=OmegaConf.to_container(benchmark.checkpoints, resolve=True)))
        selection = json.loads((ROOT / benchmark.selection).read_text())
        infer(benchmark, selection, destination)
        score(benchmark, selection, destination)
        write_json(destination / 'completion.json', dict(state='completed', targets=len(selection['records'])))
    write_json(output / 'completion.json', dict(state='completed', training_steps=cfg.max_steps,
        predictions=len(predictions), source_counts=source_counts))


if __name__ == '__main__':
    main()
