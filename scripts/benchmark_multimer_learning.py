"""Compare unguided before/after samples to synthetic training targets, not experiments.

Requires DockQ 2.1.3. Every receptor is assessed against the entire ligand dimer;
a single type-compatible chain assignment is used for all four receptor scores.
"""
from pathlib import Path
from itertools import permutations, product
import argparse
import csv
import json
import sys
import tempfile
import gemmi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.structure_clashes import analyze_clashes


def canonical(path):
    result = gemmi.read_structure(str(path))
    result.remove_hydrogens(); result.remove_waters(); result.remove_alternative_conformations()
    for chain in result[0]:
        for i,residue in enumerate(chain,1):
            residue.seqid = gemmi.SeqId(i,' ')
    return result


def pair(structure, ligands, receptor, path):
    output, model = gemmi.Structure(), gemmi.Model('1')
    for new_name, originals in [('X',ligands),('Y',[receptor])]:
        chain = gemmi.Chain(new_name)
        for copy,name in enumerate(originals):
            for residue in structure[0][name]:
                r=residue.clone(); r.seqid=gemmi.SeqId(r.seqid.num+copy*1000,' '); chain.add_residue(r)
        model.add_chain(chain)
    output.add_model(model); output.write_pdb(str(path))


def score(prediction, reference, groups):
    from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
    from DockQ import DockQ as engine
    pred, ref = canonical(prediction), canonical(reference)
    for name in 'ABCDEF':
        assert [r.name for r in pred[0][name]] == [r.name for r in ref[0][name]], name
    candidates=[]
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        for index,(ligands,type1,type2) in enumerate(product(permutations('AB'),permutations('CD'),permutations('EF'))):
            mapping=dict(zip('ABCDEF',ligands+type1+type2));rows=[]
            for rec in 'CDEF':
                native_path=tmp/f'native-{index}-{rec}.pdb';model_path=tmp/f'model-{index}-{rec}.pdb'
                pair(ref,'AB',rec,native_path); pair(pred,[mapping['A'],mapping['B']],mapping[rec],model_path)
                result,_=run_on_all_native_interfaces(load_PDB(str(model_path)),load_PDB(str(native_path)),chain_map={'X':'X','Y':'Y'},no_align=True)
                result=next(iter(result.values()))
                rows.append(dict(receptor=groups[rec],ligand=groups['A'],native_chain=rec,prediction_chain=mapping[rec],
                    DockQ=float(result['DockQ']),fnat=float(result['fnat']),iRMSD=float(result['iRMSD']),LRMSD=float(result['LRMSD'])))
            candidates.append(dict(mapping=mapping,mean_DockQ=float(np.mean([r['DockQ'] for r in rows])),
                weakest_DockQ=min(r['DockQ'] for r in rows),interfaces=rows))
            # DockQ uses structure-dependent cached distances; bound cache lifetime to one assignment.
            for name in ['get_residue_distances','get_aligned_residues']:
                fn=getattr(engine,name,None)
                if hasattr(fn,'cache_clear'): fn.cache_clear()
    best=max(candidates,key=lambda c:c['mean_DockQ'])
    clashes=analyze_clashes(prediction,groups)
    best['overlaps']=clashes['clash_count'];best['interchain_overlaps']=clashes['interchain_clash_count']
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    selection={r['id']:r for r in json.loads(a.selection.read_text())['records']}
    records=[]
    for item in json.loads((a.run/'prediction-receipt.json').read_text()):
        target=selection[item['id']]
        scored=score(a.run/item['file'],a.source/target['file'],target['groups'])
        records.append(dict(**item,**scored,reference_kind='synthetic training target',reference_sha256=target['sha256']))
        (a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
    summary=[]
    for target in selection:
        for condition in ['pretrained','fine_tuned']:
            rows=[r for r in records if r['id']==target and r['condition']==condition]
            summary.append(dict(complex=selection[target]['complex'],condition=condition,samples=len(rows),
                mean_DockQ=float(np.mean([r['mean_DockQ'] for r in rows])),
                std_DockQ=float(np.std([r['mean_DockQ'] for r in rows])),
                best_DockQ=max(r['mean_DockQ'] for r in rows),
                mean_overlaps=float(np.mean([r['overlaps'] for r in rows])),
                mean_interchain_overlaps=float(np.mean([r['interchain_overlaps'] for r in rows]))))
    with (a.output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
