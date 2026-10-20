"""Build a local benchmark summary and experimental-reference overlay viewer."""
from pathlib import Path
import argparse
import json
import shutil
import gemmi
import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices


def aligned_pdb(path, reference):
    model=gemmi.read_structure(str(path));native=gemmi.read_structure(str(reference))
    def ca(s):
        residues=[r for r in s[0][0] if r.find_atom('CA','*') is not None]
        seq=''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in residues)
        coords=np.array([[r.find_atom('CA','*').pos.x,r.find_atom('CA','*').pos.y,r.find_atom('CA','*').pos.z] for r in residues])
        return seq,coords
    ns,nc=ca(native);ms,mc=ca(model)
    aligner=PairwiseAligner();aligner.substitution_matrix=substitution_matrices.load('BLOSUM62');aligner.open_gap_score=-10;aligner.extend_gap_score=-.5
    first,second=aligner.align(ns,ms)[0].indices;mask=(first>=0)&(second>=0)
    fixed,moving=nc[first[mask]],mc[second[mask]]
    a,b=moving.mean(0),fixed.mean(0);u,_,vt=np.linalg.svd((moving-a).T@(fixed-b));d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt);rot=u@d@vt
    for c in model[0]:
        for r in c:
            for atom in r:
                xyz=(np.array([atom.pos.x,atom.pos.y,atom.pos.z])-a)@rot+b;atom.pos=gemmi.Position(*xyz)
    return model.make_pdb_string()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True);a=p.parse_args()
    summary=json.loads((a.run/'benchmark-summary.json').read_text());rows=json.loads((a.run/'benchmark-results.json').read_text());receipt=json.loads((a.run/'dataset-receipt.json').read_text())
    targets=[]
    for item in receipt['records']:
        ref=a.run/'references'/f'{item["id"]}.pdb';copies={}
        for arm in ['pretrained','fine_tuned']:
            scores=[r for r in rows if r['id']==item['id'] and r['condition']==arm];assert len(scores)==1
            copies[arm]=dict(scores=scores[0],pdb=aligned_pdb(a.run/scores[0]['file'],ref))
        targets.append(dict(id=item['id'],length=item['length'],reference=ref.read_text(),source=item['reference_url'],**copies))
    data=dict(summary=summary,targets=targets)
    a.output.mkdir(parents=True,exist_ok=True);shutil.copy2(a.library,a.output/'3Dmol-min.js')
    template='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><script src="3Dmol-min.js"></script><title>CAMEO22 checkpoint benchmark</title><style>
body{font:15px/1.5 system-ui,sans-serif;color:#233344;background:#f5f7fa;margin:0}main{max-width:1400px;margin:auto;padding:24px}h1{margin:0}h2{font-size:20px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px;background:white}th,td{padding:9px;border-bottom:1px solid #dce3eb;text-align:left}select,button{font:inherit;padding:6px}.note{padding:12px;background:#eaf0f6;border-radius:7px}.controls{display:flex;gap:18px;flex-wrap:wrap;margin:15px 0}.viewer{height:600px;position:relative;background:white;border:1px solid #dce3eb}.scroll{overflow:auto}tr[data-id]{cursor:pointer}tr[data-id]:hover{background:#eaf0f6}.legend{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.muted{font-size:12px;color:#596c7d}
</style></head><body><main><h1>CAMEO22: original vs fine-tuned 100M</h1><p class="note">A fixed, length-stratified subset held out from our six-complex fine-tuning dataset. Experimental reference structures; matched seeds, 500 sampling steps, τ=0.01. These subset results do not reproduce the paper’s full183-target averages.</p><h2>Dataset means</h2><div class="scroll" id="summary"></div><h2>Structure examples</h2><div class="controls"><label>Target <select id="target"></select></label><label><input type="checkbox" id="reference" checked><span class="legend" style="background:#96a9ba"></span>Experimental reference</label><label><input type="checkbox" id="pretrained" checked><span class="legend" style="background:#244c73"></span>Original100M</label><label><input type="checkbox" id="fine_tuned" checked><span class="legend" style="background:#d57728"></span>Fine-tuned100M</label><button id="reset">Reset view</button></div><div class="viewer" id="viewer"></div><p class="muted" id="detail"></p><p class="muted">Each prediction is fitted separately to sequence-matched experimental Cα atoms for display. This display fit does not change the OpenStructure scores.</p><h2>Per-target scores</h2><p class="muted">Each cell shows original → fine-tuned. Click a target row to inspect its structures. One sample per target per checkpoint.</p><div class="scroll" id="results"></div></main><script>
const data=__DATA__,metrics=[['tm_score','TM-score ↑'],['oligo_gdtts','GDT-TS ↑'],['lddt','All-atom lDDT ↑'],['bb_lddt','Backbone lDDT ↑'],['rmsd','RMSD ↓ (Å)']],fmt=x=>Number(x).toFixed(3),esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
document.getElementById('summary').innerHTML='<table><tr><th>Checkpoint</th>'+metrics.map(m=>'<th>'+m[1]+'</th>').join('')+'</tr>'+['pretrained','fine_tuned'].map(k=>'<tr><td>'+({pretrained:'Original100M',fine_tuned:'Fine-tuned100M'})[k]+'</td>'+metrics.map(m=>'<td>'+fmt(data.summary.means_and_medians[k][m[0]].mean)+'</td>').join('')+'</tr>').join('')+'</table>';
const target=document.getElementById('target');target.innerHTML=data.targets.map(t=>'<option value="'+esc(t.id)+'">'+esc(t.id)+' · '+t.length+' residues · TM '+fmt(t.pretrained.scores.tm_score)+' → '+fmt(t.fine_tuned.scores.tm_score)+'</option>').join('');
const viewer=$3Dmol.createViewer(document.getElementById('viewer'),{backgroundColor:'white',antialias:true});let startView=null;
function render(reset=false){const t=data.targets.find(t=>t.id===target.value),old=viewer.getView();viewer.removeAllModels();for(const [key,color] of [['reference','#96a9ba'],['pretrained','#244c73'],['fine_tuned','#d57728']])if(document.getElementById(key).checked){const m=viewer.addModel(key==='reference'?t.reference:t[key].pdb,'pdb');m.setStyle({},{cartoon:{color}});}if(reset){viewer.zoomTo({});startView=viewer.getView();}else viewer.setView(old);viewer.render();document.getElementById('detail').innerHTML=esc(t.id)+' · '+t.length+' residues · <a href="'+esc(t.source)+'">Experimental reference and provenance</a>';}
target.onchange=()=>render(true);for(const id of ['reference','pretrained','fine_tuned'])document.getElementById(id).onchange=()=>render();document.getElementById('reset').onclick=()=>{viewer.setView(startView);viewer.render();};
document.getElementById('results').innerHTML='<table><tr><th>Target</th><th>Residues</th>'+metrics.map(m=>'<th>'+m[1]+'</th>').join('')+'</tr>'+data.targets.map(t=>'<tr data-id="'+esc(t.id)+'"><td>'+esc(t.id)+'</td><td>'+t.length+'</td>'+metrics.map(m=>'<td>'+fmt(t.pretrained.scores[m[0]])+' → '+fmt(t.fine_tuned.scores[m[0]])+'</td>').join('')+'</tr>').join('')+'</table>';for(const row of document.querySelectorAll('tr[data-id]'))row.onclick=()=>{target.value=row.dataset.id;render(true);document.getElementById('viewer').scrollIntoView({behavior:'smooth',block:'center'});};render(true);window.viewerReady=true;
</script></body></html>'''
    (a.output/'index.html').write_text(template.replace('__DATA__',json.dumps(data).replace('</','<\\/')))
    (a.output/'report-data.json').write_text(json.dumps(data))
    print(a.output/'index.html')


if __name__=='__main__':main()
