#!/usr/bin/env python3
"""Fine-tune released folding weights on a frozen mixture; compute is Slurm-only."""
import copy
from collections import Counter
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

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


def retention_config(cfg, pretrained_hash):
    """Carry the actual folding architecture and both weights into the benchmark."""
    benchmark = hydra.compose(config_name='cameo_benchmark')
    benchmark.artifact_root = cfg.artifact_root
    benchmark.model_name = cfg.model_name
    benchmark.model.architecture = copy.deepcopy(cfg.model.architecture)
    benchmark.output_dir = str(Path(cfg.output_dir) / 'retention')
    benchmark.checkpoints.pretrained.path = str(cfg.pretrained_checkpoint)
    benchmark.checkpoints.pretrained.sha256 = pretrained_hash
    benchmark.checkpoints.fine_tuned.path = str(Path(cfg.trained_dir) / f'simplefold_{cfg.model_name}_ema.ckpt')
    return benchmark


def evaluate_complexes(cfg, synthetic, output):
    """Matched seeds/settings against the same six synthetic training targets."""
    import numpy as np
    import torch
    import lightning.pytorch as pl
    from boltz_data_pipeline.types import Record, Structure
    from lightning.fabric.utilities.apply_func import move_data_to_device
    from utils.datamodule_utils import collate
    from utils.boltz_utils import process_structure, save_structure
    model = hydra.utils.instantiate(cfg.model.architecture).eval().requires_grad_(False).cuda()
    processor = hydra.utils.instantiate(cfg.model.processor)(device=torch.device('cuda'))
    flow = hydra.utils.instantiate(cfg.model.path)
    synthetic.rotation_augment_coords = False
    sampler = hydra.utils.instantiate(cfg.model.sampler,
        num_timesteps=cfg.evaluation.steps, tau=cfg.evaluation.tau, log_timesteps=True)
    predictions = []
    trained = Path(cfg.trained_dir) / f'simplefold_{cfg.model_name}_ema.ckpt'
    selection = json.loads((ROOT / cfg.selection).read_text())['records']
    lookup = {record_id: index for index, (_, record_id) in enumerate(synthetic.samples)}
    for condition, weights in [('pretrained', Path(cfg.pretrained_checkpoint)),
                               ('fine_tuned', trained)]:
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        folder = output / condition / f'predictions_simplefold_{cfg.model_name}'
        folder.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(selection):
            for sample in range(cfg.evaluation.samples):
                seed = cfg.evaluation.seed + index * 100 + sample
                pl.seed_everything(seed, workers=True)
                batch = move_data_to_device(collate([synthetic[lookup[item['id']]]]), torch.device('cuda'))
                assert batch['record'][0]['id'] == item['id']
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    batch = processor.preprocess_inference(batch)
                    out = sampler.sample(model.forward, flow,
                                         torch.randn_like(batch['coords']), batch)
                    assert torch.isfinite(out['denoised_coords']).all()
                    out = processor.postprocess(out, batch)
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


def source_run(cfg):
    """Reuse the training run's resolved settings, never fresh evaluation defaults."""
    output = Path(cfg.source_run)
    saved = OmegaConf.load(output / 'resolved-config.yaml')
    experiment = json.loads((output / 'experiment.json').read_text())
    summary = json.loads((output / 'training-summary.json').read_text())
    training = json.loads((output / 'training-completion.json').read_text())
    assert training['config_sha256'] == digest(output / 'resolved-config.yaml')
    assert summary['model_name'] == saved.model_name
    assert digest(ROOT / saved.selection) == experiment['synthetic_selection_sha256']
    return output, saved, summary


def stage_receipt(output, phase, started, **values):
    write_json(output / f'{phase}-completion.json', dict(state=phase + '_completed',
        job_id=os.environ['SLURM_JOB_ID'], seconds=time.perf_counter()-started,
        config_sha256=digest(output / 'resolved-config.yaml'),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        **values))


def sample_run(cfg):
    import torch
    import lightning.pytorch as pl
    from run_cameo_benchmark import infer
    output, saved, summary = source_run(cfg)
    assert json.loads((output / 'training-completion.json').read_text())['state'] == 'training_completed'
    assert not (output / 'prediction-receipt.json').exists(), 'Sampling outputs already exist'
    assert torch.cuda.is_available(), 'Sampling requires CUDA'
    started = time.perf_counter()
    assert digest(saved.pretrained_checkpoint) == summary['pretrained_sha256']
    assert Path(summary['ema_checkpoint']) == Path(saved.trained_dir) / f'simplefold_{saved.model_name}_ema.ckpt'
    assert digest(summary['ema_checkpoint']) == summary['ema_sha256']
    os.environ['TORCH_HOME'] = str(Path(saved.artifact_root) / 'torch')
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(saved.seed, workers=True)
    data = hydra.utils.instantiate(saved.data)
    predictions = evaluate_complexes(saved, data._train_set, output)
    gc.collect()
    torch.cuda.empty_cache()
    if saved.retention.enabled:
        benchmark = retention_config(saved, summary['pretrained_sha256'])
        benchmark.checkpoints.fine_tuned.sha256 = summary['ema_sha256']
        destination = output / 'retention'
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'resolved-config.yaml').write_text(OmegaConf.to_yaml(benchmark, resolve=True))
        selection = json.loads((ROOT / benchmark.selection).read_text())
        write_json(destination / 'experiment.json', dict(parent_job=saved.job_id,
            sampling_job=os.environ['SLURM_JOB_ID'], purpose='Fixed24-target CAMEO22 retention after mixed training',
            selection_sha256=digest(ROOT / benchmark.selection),
            sampling=OmegaConf.to_container(benchmark.sampling, resolve=True),
            checkpoints=OmegaConf.to_container(benchmark.checkpoints, resolve=True)))
        infer(benchmark, selection, destination)
    stage_receipt(output, 'sampling', started, predictions=len(predictions),
        retention_config_sha256=digest(output / 'retention/resolved-config.yaml') if saved.retention.enabled else None)
    write_json(output / 'completion.json', dict(state='sampled', training_steps=saved.max_steps,
        predictions=len(predictions), source_counts=summary['source_counts']))


def score_run(cfg):
    """CPU-only OpenStructure evaluation; no model/encoder construction or CUDA calls."""
    from run_cameo_benchmark import score
    output, saved, summary = source_run(cfg)
    assert json.loads((output / 'sampling-completion.json').read_text())['state'] == 'sampling_completed'
    assert not (output / 'scoring-completion.json').exists(), 'Scoring already completed'
    started = time.perf_counter()
    if saved.retention.enabled:
        destination = output / 'retention'
        sampling = json.loads((output / 'sampling-completion.json').read_text())
        assert sampling['retention_config_sha256'] == digest(destination / 'resolved-config.yaml')
        benchmark = OmegaConf.load(destination / 'resolved-config.yaml')
        experiment = json.loads((destination / 'experiment.json').read_text())
        assert digest(ROOT / benchmark.selection) == experiment['selection_sha256']
        selection = json.loads((ROOT / benchmark.selection).read_text())
        score(benchmark, selection, destination)
        write_json(destination / 'completion.json', dict(state='completed', targets=len(selection['records'])))
    predictions = json.loads((output / 'prediction-receipt.json').read_text())
    stage_receipt(output, 'scoring', started)
    write_json(output / 'completion.json', dict(state='completed', training_steps=saved.max_steps,
        predictions=len(predictions), source_counts=summary['source_counts']))


def continue_sampling(cfg, output):
    """Replace the training process; reuse its Slurm allocation, free all training state."""
    command = [sys.executable, str(ROOT / 'scripts/train_mixed.py'),
        '--config-name=mixed_train', 'phase=sample', 'source_run=' + str(output),
        'hydra.run.dir=logs/mixed-sampling/' + os.environ['SLURM_JOB_ID']]
    if cfg.expected_commit:
        command.append('expected_commit=' + json.dumps(str(cfg.expected_commit)))
    print('Training complete; continuing GPU sampling in the same Slurm allocation.', flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, command)


@hydra.main(version_base='1.3', config_path='../configs', config_name='mixed_train')
def main(cfg):
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm for preparation, training and inference'
    phase_started = time.perf_counter()
    if cfg.expected_commit:
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == cfg.expected_commit
    if cfg.phase in ('sample', 'score'):
        return {'sample': sample_run, 'score': score_run}[cfg.phase](cfg)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'experiment.json').exists(), 'Use a fresh run directory'
    (output / 'resolved-config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    (output / 'git-commit.txt').write_text(commit + '\n')
    write_json(output / 'experiment.json', dict(job_id=cfg.job_id, phase=cfg.phase,
        source_commit=commit, seed=cfg.seed, max_steps=cfg.max_steps,
        synthetic_probability=cfg.synthetic_probability,
        synthetic_selection=str(cfg.selection),
        synthetic_selection_sha256=digest(ROOT / cfg.selection),
        synthetic_dataset=str(cfg.dataset_dir),
        checkpoint=str(cfg.pretrained_checkpoint), model_name=cfg.model_name, encoder='frozen ESM2-3B',
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
    pretrained_hash = digest(cfg.pretrained_checkpoint)
    expected_hash = cfg.pretrained_sha256
    if cfg.preflight_receipt:
        preflight = json.loads(Path(cfg.preflight_receipt).read_text())
        assert preflight['model_name'] == cfg.model_name
        expected_hash = preflight['pretrained_sha256']
    if expected_hash:
        assert pretrained_hash == expected_hash, 'Original released checkpoint required'
    os.environ['TORCH_HOME'] = str(Path(cfg.artifact_root) / 'torch')
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg.seed, workers=True)
    plan_path = Path(cfg.general_dataset_dir) / 'training-plan.json'
    plan = json.loads(plan_path.read_text())
    receipt = json.loads((Path(cfg.general_dataset_dir) / 'dataset-receipt.json').read_text())
    assert digest(plan_path) == receipt['training_plan_sha256']
    if cfg.preflight_receipt:
        assert preflight['training_plan_sha256'] == receipt['training_plan_sha256']
    prepared_selection = ROOT / cfg.get('general_replay_selection', cfg.selection)
    assert digest(prepared_selection) == receipt['inputs']['synthetic_selection_sha256']
    # Replay draws name complexes by ID. Cropped targets may reuse that frozen
    # schedule only when the same complexes and chain identities are retained.
    current_records = json.loads((ROOT / cfg.selection).read_text())['records']
    prepared_records = json.loads(prepared_selection.read_text())['records']
    assert [(r['id'], r['groups']) for r in current_records] == [
        (r['id'], r['groups']) for r in prepared_records]
    shutil.copy2(ROOT / cfg.selection, output / 'synthetic-selection.json')
    shutil.copy2(Path(cfg.dataset_dir) / 'dataset-receipt.json',
                 output / 'synthetic-dataset-receipt.json')
    write_json(output / 'replay-provenance.json', dict(
        prepared_selection=str(prepared_selection),
        prepared_selection_sha256=digest(prepared_selection),
        active_selection=str(cfg.selection), active_selection_sha256=digest(ROOT / cfg.selection),
        training_plan_sha256=digest(plan_path),
        general_dataset=str(cfg.general_dataset_dir),
        reuse='Unchanged general structures and draw schedule; same complex IDs and chain groups'))
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
    examples = []
    for index in range(len(data.dataset)):
        features = data.dataset[index]
        examples.append(dict(index=index, source=draws[index]['source'],
            record_id=draws[index]['record_id'], residues=int(features['cropped_num_tokens'])))
        if draws[index]['source'] == 'synthetic':
            assert features['cropped_num_tokens'] == features['max_num_tokens']
        else:
            assert int(features['cropped_num_tokens']) <= cfg.general.max_tokens
    print('Mixture preflight passed:', dict(Counter(draw['source'] for draw in draws)), flush=True)
    if cfg.phase == 'preflight':
        write_json(output / 'preflight-summary.json', dict(model_name=cfg.model_name,
            pretrained_checkpoint=str(cfg.pretrained_checkpoint), pretrained_sha256=pretrained_hash,
            training_plan_sha256=digest(plan_path), draws=len(draws), examples=examples))
        write_json(output / 'completion.json', dict(state='preflight_passed', draws=len(draws)))
        return

    probe = cfg.phase == 'memory_probe'
    if probe:
        # Include every full synthetic complex and the two largest general examples.
        chosen = {}
        for item in sorted(examples, key=lambda r: r['residues'], reverse=True):
            if item['source'] == 'synthetic':
                chosen.setdefault(item['record_id'], item)
        general = sorted((r for r in examples if r['source'] != 'synthetic'),
                         key=lambda r: r['residues'], reverse=True)[:2]
        selected = list(chosen.values()) + general
        write_json(output / 'probe-examples.json', selected)
        active_draws = [draws[r['index']] for r in selected]
        data = MixtureDataModule({key: module._train_set for key, module in modules.items()}, active_draws)
    else:
        active_draws = draws

    assert torch.cuda.is_available(), 'CUDA required'
    trained = Path(cfg.trained_dir)
    trained.mkdir(parents=True, exist_ok=True)
    model = hydra.utils.instantiate(cfg.model)
    load_pretrained_folding_weights(model, cfg.pretrained_checkpoint)
    model.model_ema.requires_grad_(False)
    if cfg.activation_checkpointing:
        from model.torch.blocks import HomogenTrunk
        for module in model.model.modules():
            if isinstance(module, HomogenTrunk):
                module.activation_checkpointing = True

    class Resources(pl.Callback):
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            torch.cuda.synchronize()
            self.started = time.perf_counter()

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            torch.cuda.synchronize()
            item = dict(step=batch_idx, seconds=time.perf_counter()-self.started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
            with (output / 'step-resources.jsonl').open('a') as stream:
                stream.write(json.dumps(item)+'\n')

    callbacks = [Resources()]
    checkpointing = not probe and cfg.checkpoint_every > 0
    if checkpointing:
        callbacks.append(ModelCheckpoint(dirpath=trained, filename='step-{step:06d}',
            every_n_train_steps=cfg.checkpoint_every, save_top_k=-1, save_last=cfg.checkpoint_save_last,
            save_on_train_epoch_end=False))
    trainer = pl.Trainer(accelerator='gpu', devices=1, strategy='auto', precision='bf16-mixed',
        max_steps=len(active_draws), max_epochs=1, limit_val_batches=0, num_sanity_val_steps=0,
        logger=CSVLogger(output / 'logs', name='training'), callbacks=callbacks, enable_checkpointing=checkpointing,
        log_every_n_steps=1, enable_model_summary=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(model, datamodule=data)
    torch.cuda.synchronize()
    assert trainer.global_step == len(active_draws)
    assert dict(model.source_counts) == dict(Counter(draw['source'] for draw in active_draws))
    device = torch.cuda.get_device_properties(0)
    write_json(output / 'resource-summary.json', dict(model_name=cfg.model_name,
        device=device.name, device_total_bytes=device.total_memory, updates=trainer.global_step,
        fit_seconds=time.perf_counter()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), activation_checkpointing=cfg.activation_checkpointing,
        pretrained_sha256=pretrained_hash, source_counts=dict(model.source_counts)))
    if probe:
        write_json(output / 'completion.json', dict(state='memory_probe_passed', steps=trainer.global_step))
        return
    if cfg.save_full_final:
        trainer.save_checkpoint(trained / 'final.ckpt')
    exports = [('ema', model.model_ema.module)]
    if cfg.export_raw:
        exports.insert(0, ('raw', model.model))
    for label, weights in exports:
        torch.save({key: value.detach().cpu() for key, value in weights.state_dict().items()},
                   trained / f'simplefold_{cfg.model_name}_{label}.ckpt')
    del weights, exports
    write_json(output / 'training-summary.json', dict(global_step=trainer.global_step,
        source_counts=dict(model.source_counts), synthetic_probability=cfg.synthetic_probability,
        realized_synthetic_fraction=model.source_counts['synthetic'] / trainer.global_step,
        training_plan_sha256=digest(plan_path), pretrained_sha256=pretrained_hash,
        model_name=cfg.model_name,
        ema_checkpoint=str(trained / f'simplefold_{cfg.model_name}_ema.ckpt'),
        ema_sha256=digest(trained / f'simplefold_{cfg.model_name}_ema.ckpt'),
        ema_decay=cfg.model.ema_decay, encoder_trainable=False,
        validation='Complex reconstruction uses training targets; CAMEO retention is separate'))
    stage_receipt(output, 'training', phase_started, training_steps=trainer.global_step)
    write_json(output / 'completion.json', dict(state='trained', training_steps=cfg.max_steps,
        source_counts=dict(model.source_counts)))
    if cfg.phase == 'train_sample':
        continue_sampling(cfg, output)


if __name__ == '__main__':
    main()
