#!/usr/bin/env python3
"""Fixed held-out CAMEO22 retention benchmark; sampling exclusively in Slurm."""
from pathlib import Path
import copy
import csv
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/simplefold'))
import hydra
from omegaconf import OmegaConf

METRICS = ('tm_score', 'oligo_gdtts', 'lddt', 'bb_lddt', 'rmsd')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def ost_score(cfg, model, reference, output):
    cmd=['apptainer','exec','--cleanenv','--bind',f'{cfg.artifact_root}:{cfg.artifact_root}',
         str(cfg.openstructure.path),'ost','compare-structures','-m',str(model),'-r',str(reference),
         '-o',str(output),'--fault-tolerant','--min-pep-length','4','--lddt','--bb-lddt','--rigid-scores','--tm-score']
    result=subprocess.run(cmd,text=True,capture_output=True)
    output.with_suffix('.log').write_text(result.stdout+'\n'+result.stderr)
    result.check_returncode()
    scores=json.loads(output.read_text())
    assert scores.get('status','SUCCESS')=='SUCCESS', scores.get('status')
    import math
    for key in METRICS:
        assert key in scores and math.isfinite(float(scores[key])), (key,output)
    return {key:float(scores[key]) for key in METRICS}


def prepare(cfg,selection,output):
    import gemmi
    dataset=Path(cfg.dataset_dir)
    for name in ['references','fasta','source']: (dataset/name).mkdir(parents=True,exist_ok=True)
    records=[]
    for item in selection['records']:
        ref=dataset/'references'/f'{item["id"]}.pdb'
        if not ref.exists():
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(item['reference_url'],timeout=60) as response:
                        content=response.read()
                    assert b'ATOM ' in content,ref
                    ref.write_bytes(content);break
                except Exception:
                    if attempt==2:raise
                    time.sleep(2)
        structure=gemmi.read_structure(str(ref));structure.remove_hydrogens();structure.remove_waters()
        protein=[c for c in structure[0] if any(gemmi.find_tabulated_residue(r.name).is_amino_acid() for r in c)]
        assert len(protein)==1,(item['id'],[c.name for c in protein])
        observed=sum(1 for r in protein[0] if r.find_atom('CA','*') is not None)
        assert observed==item['resolved_length'],(item['id'],observed,item['resolved_length'])
        (dataset/'fasta'/f'{item["id"]}.fasta').write_text(f'>A|{item["id"]}\n{item["sequence"]}\n')
        records.append(dict(**item,reference_sha256=digest(ref),reference_file=str(ref)))
    manifest=dict(**selection,prepared_records=records)
    write_json(dataset/'manifest.json',manifest)
    write_json(output/'dataset-receipt.json',manifest)
    image=Path(cfg.openstructure.path);image.parent.mkdir(parents=True,exist_ok=True)
    if not image.exists():
        cache=ROOT/'artifacts/environment/apptainer-cache';cache.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,APPTAINER_CACHEDIR=str(cache))
        with tempfile.TemporaryDirectory(prefix='simplefold-ost-') as tmp:
            env['APPTAINER_TMPDIR']=tmp
            subprocess.run(['apptainer','pull',str(image),cfg.openstructure.image],env=env,check=True)
    ref=Path(records[0]['reference_file'])
    self_score=ost_score(cfg,ref,ref,output/'openstructure-identity-control.json')
    assert self_score['tm_score']>.999 and self_score['lddt']>.999,self_score
    write_json(dataset/'openstructure-receipt.json',dict(image=cfg.openstructure.image,sif_sha256=digest(image),identity=self_score))
    print('Prepared',len(records),'fixed CAMEO22 targets and verified OpenStructure identity control',flush=True)


def infer(cfg,selection,output):
    import numpy as np
    import torch
    import lightning.pytorch as pl
    from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
    from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
    from processor.protein_processor import ProteinDataProcessor
    from utils.esm_utils import esm_registry,_af2_to_esm
    from utils.fasta_utils import process_fastas
    from utils.datamodule_utils import process_one_inference_structure
    from utils.boltz_utils import process_structure,save_structure
    from model.flow import LinearPath
    from model.torch.sampler import EMSampler

    assert torch.cuda.is_available(),'CUDA required for sampling'
    torch.set_float32_matmul_precision('medium')
    os.environ['TORCH_HOME']=str(Path(cfg.artifact_root)/'torch')
    dataset=Path(cfg.dataset_dir);data=json.loads((dataset/'manifest.json').read_text())
    assert data['records']==selection['records'],'Dataset selection changed'
    for item in data['prepared_records']:
        assert digest(item['reference_file'])==item['reference_sha256']
    write_json(output/'dataset-receipt.json',data)
    runtime=json.loads((dataset/'openstructure-receipt.json').read_text())
    assert digest(cfg.openstructure.path)==runtime['sif_sha256'], 'Scoring runtime changed'
    shutil.copy2(dataset/'openstructure-receipt.json',output/'openstructure-receipt.json')
    for checkpoint in cfg.checkpoints.values():assert digest(checkpoint.path)==checkpoint.sha256
    fasta=[dataset/'fasta'/f'{r["id"]}.fasta' for r in selection['records']]
    prepared=output/'prepared-inputs'
    process_fastas(fasta,prepared,Path(cfg.artifact_root)/'ccd/ccd.pkl')
    assert len(list((prepared/'structures').glob('*.npz')))==len(fasta)
    encoder,alphabet=esm_registry['esm2_3B']();encoder=encoder.eval().requires_grad_(False).cuda()
    mapping=_af2_to_esm(alphabet).cuda()
    processor=ProteinDataProcessor(device='cuda',multiplicity=1,inference_multiplicity=1)
    tokenizer,featurizer=BoltzTokenizer(),BoltzFeaturizer()
    model=hydra.utils.instantiate(cfg.model.architecture).eval().requires_grad_(False).cuda()
    sampler=EMSampler(num_timesteps=cfg.sampling.steps,tau=cfg.sampling.tau,log_timesteps=True)
    flow=LinearPath();predictions=[]
    states={condition:torch.load(checkpoint.path,map_location='cpu',weights_only=True) for condition,checkpoint in cfg.checkpoints.items()}
    # Prepare conditioning solely from sequence; experimental coordinates never enter model features.
    for index,item in enumerate(selection['records']):
        pl.seed_everything(cfg.sampling.seed+index*100,workers=True)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            batch,structure,record=process_one_inference_structure(prepared/'structures'/f'{item["id"]}.npz',
                prepared/'records'/f'{item["id"]}.json',tokenizer,featurizer,processor,encoder,alphabet,mapping)
        assert batch['aa_seq']==[item['sequence']]
        for condition,checkpoint in cfg.checkpoints.items():
            model.load_state_dict(states[condition],strict=True)
            folder=output/condition/f'predictions_simplefold_{cfg.model_name}';folder.mkdir(parents=True,exist_ok=True)
            for sample in range(cfg.sampling.samples):
                seed=cfg.sampling.seed+index*100+sample
                pl.seed_everything(seed,workers=True)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    out=sampler.sample(model.forward,flow,torch.randn_like(batch['coords']),batch)
                    assert torch.isfinite(out['denoised_coords']).all()
                    # Postprocess a copy so scale/centering bookkeeping cannot affect the other arm.
                    final=processor.postprocess(out,dict(batch))['denoised_coords'][0].float()
                predicted=process_structure(copy.deepcopy(structure),final,batch['atom_pad_mask'][0],record)
                name=f'{item["id"]}_sampled_{sample}'
                save_structure(predicted,folder,name,output_format='pdb')
                save_structure(predicted,folder,name,output_format='mmcif')
                path=folder/(name+'.pdb')
                predictions.append(dict(id=item['id'],condition=condition,sample=sample,seed=seed,
                    checkpoint_sha256=checkpoint.sha256,file=str(path.relative_to(output)),sha256=digest(path),
                    cif_file=str(path.with_suffix('.cif').relative_to(output)),cif_sha256=digest(path.with_suffix('.cif'))))
                write_json(output/'prediction-receipt.json',predictions)
                print('Saved',condition,name,flush=True)
    del encoder,model;gc.collect();torch.cuda.empty_cache()


def score(cfg,selection,output):
    import numpy as np
    import gemmi
    predictions=json.loads((output/'prediction-receipt.json').read_text())
    reference_dir=output/'references';reference_dir.mkdir(exist_ok=True)
    targets={r['id']:r for r in selection['records']}
    receipts=json.loads((output/'dataset-receipt.json').read_text())['prepared_records']
    refs={r['id']:r for r in receipts};rows=[]
    folder=output/'scores';folder.mkdir(exist_ok=True)
    validations=[]
    for p in predictions:
        path=output/p['file'];assert digest(path)==p['sha256']
        structure=gemmi.read_structure(str(path));assert len(structure[0])==1
        seq=''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in structure[0][0])
        assert seq==targets[p['id']]['sequence']
        coords=np.array([[a.pos.x,a.pos.y,a.pos.z] for c in structure[0] for r in c for a in r]);assert np.isfinite(coords).all()
        validations.append(dict(file=p['file'],sequence_matches=True,finite_coordinates=True,residues=len(seq)))
        reference=Path(refs[p['id']]['reference_file'])
        copied=reference_dir/reference.name
        if not copied.exists():shutil.copy2(reference,copied)
        values=ost_score(cfg,path,reference,folder/f'{p["condition"]}-{p["id"]}-{p["sample"]}.json')
        rows.append(dict(**p,**values))
        write_json(output/'benchmark-results.json',rows)
    write_json(output/'validation.json',validations)
    summary={}
    for condition in cfg.checkpoints:
        sub=[r for r in rows if r['condition']==condition]
        assert len(sub)==len(selection['records'])*cfg.sampling.samples
        summary[condition]={key:dict(mean=float(np.mean([r[key] for r in sub])),median=float(np.median([r[key] for r in sub]))) for key in METRICS}
    delta={}
    rng=np.random.default_rng(20261020)
    for key in METRICS:
        pairs=[]
        for target in targets:
            a=np.mean([r[key] for r in rows if r['id']==target and r['condition']=='pretrained'])
            b=np.mean([r[key] for r in rows if r['id']==target and r['condition']=='fine_tuned'])
            pairs.append(b-a)
        pairs=np.array(pairs);boot=np.mean(rng.choice(pairs,size=(10000,len(pairs)),replace=True),axis=1)
        delta[key]=dict(mean_fine_tuned_minus_pretrained=float(pairs.mean()),bootstrap_95_percent_CI=[float(x) for x in np.quantile(boot,[.025,.975])])
    write_json(output/'benchmark-summary.json',dict(targets=len(targets),samples=len(predictions),means_and_medians=summary,paired_differences=delta,
        scope='Fixed CAMEO22 subset held out from the six-complex fine-tuning set; not the full published benchmark. No best-of-sample selection.',metrics=dict(tm_score='TM-score ↑',oligo_gdtts='GDT-TS ↑',lddt='all-atom lDDT ↑',bb_lddt='Cα lDDT ↑',rmsd='RMSD Å ↓')))
    with (output/'benchmark-results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2),flush=True)


@hydra.main(version_base='1.3',config_path='../configs',config_name='cameo_benchmark')
def main(cfg):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm for benchmark preparation and inference')
    output=Path(cfg.output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'experiment.json').exists():raise RuntimeError('Existing run; use a new output directory')
    selection=json.loads((ROOT/cfg.selection).read_text())
    assert len({r['id'] for r in selection['records']})==len(selection['records'])
    (output/'resolved-config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (output/'git-commit.txt').write_text(commit+'\n')
    write_json(output/'experiment.json',dict(job_id=cfg.job_id,phase=cfg.phase,source_commit=commit,
        selection_sha256=digest(ROOT/cfg.selection),sampling=OmegaConf.to_container(cfg.sampling,resolve=True),
        checkpoints=OmegaConf.to_container(cfg.checkpoints,resolve=True),selection=selection['name']))
    if cfg.phase=='prepare':prepare(cfg,selection,output)
    elif cfg.phase=='all':infer(cfg,selection,output);score(cfg,selection,output)
    else:raise ValueError('phase must be prepare or all')
    write_json(output/'completion.json',dict(state='completed',phase=cfg.phase,targets=len(selection['records'])))


if __name__=='__main__':main()
