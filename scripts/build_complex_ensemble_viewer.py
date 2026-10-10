#!/usr/bin/env python3
"""Build a standalone, opaque 3Dmol ensemble viewer from complex metrics.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>2GOO · SimpleFold ensembles</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
:root{color-scheme:light}*{box-sizing:border-box}body{font:15px/1.5 system-ui,sans-serif;background:#f5f6f8;color:#202b37;margin:0}main{max-width:1560px;margin:auto;padding:28px}h1{font-size:28px;margin:0 0 8px}h2{margin:0;font-size:23px}h3{margin:18px 0 8px;font-size:16px}p{margin:8px 0 16px}.intro{max-width:1060px;color:#475466}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,610px),1fr));gap:24px;align-items:start}.card{background:white;border:1px solid #dce1e6;border-radius:12px;padding:22px;min-width:0}.subtle{font-size:13px;color:#526172}.legend,.stats{width:100%;border-collapse:collapse;font-size:13px}.legend th,.legend td,.stats th,.stats td{text-align:left;padding:7px 9px;border-bottom:1px solid #e6e9ec}.legend th,.stats th{font-weight:600}.legend i{display:inline-block;width:15px;height:15px;border-radius:3px;vertical-align:middle;margin-right:7px;border:1px solid #0001}.view{height:530px;position:relative;border:1px solid #dce1e6;border-radius:8px;overflow:hidden}.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:12px 0}.toolbar label{display:flex;gap:5px;align-items:center}.toolbar select,.toolbar button{font:inherit;border:1px solid #c7d0d9;padding:5px 9px;border-radius:5px;background:white}.toolbar select{max-width:260px}.scroll{overflow-x:auto}.stats td{font-variant-numeric:tabular-nums;white-space:nowrap}.stats th:first-child{min-width:130px}.links{display:flex;gap:15px;flex-wrap:wrap;font-size:13px;margin-top:12px}a{color:#175b95}.empty{padding:30px;color:#526172}.loaderror{padding:18px;color:#8b2700;background:#fff5e9;border:1px solid #efdbbc;border-radius:8px;margin-bottom:18px}details{margin-top:12px}summary{cursor:pointer;font-size:13px;color:#526172}.selected{font-size:13px;color:#475466;margin:5px 0 14px}.foot{font-size:13px;color:#526172;max-width:1050px;margin-top:24px}@media(max-width:680px){main{padding:16px}.card{padding:15px}.view{height:430px}.legend{font-size:12px}.stats{font-size:12px}}
</style></head><body><main>
<h1>2GOO · SimpleFold ensembles</h1>
<p class="intro">The reference is a crystal-symmetry reconstruction from 2GOO chains A–C and their symmetry mates, restoring the native BMP2 homodimer. The six deposited asymmetric-unit chains alone are two separate half-complexes. The reconstructed assembly contains two copies each of BMP2, BMPR1A / ALK3, and ActRIIA. Each model’s samples are aligned to the resolved experimental Cα positions, allowing identical copies to swap. Compare the assembly arrangement below; the separate chain fits measure individual protein folds.</p>
<div id="load-error"></div><div class="grid" id="grid"></div>
<p class="foot" id="coverage"></p>
<p class="foot">The experimental coordinates retain unresolved termini and local validation issues. Cα metrics use observed residues; the predictions include the full deposited sequences. No structural repairs were applied. Ensemble diversity describes variation among samples, without a preferred direction.</p>
<div class="links"><a href="metrics.json">Full metrics (JSON)</a><a href="reference.pdb">Experimental reference (PDB)</a></div>
</main><script>
'use strict';
const payload = __PAYLOAD__;
const groups = [
 {label:'BMP2', ref:['A','D'], pred:['A','B'], blue:'#08519c', orange:'#a63603', key:'BMP2_chain_CA_RMSD_angstrom'},
 {label:'BMPR1A / ALK3', ref:['B','E'], pred:['C','D'], blue:'#3182bd', orange:'#f16913', key:'BMPR1A_chain_CA_RMSD_angstrom'},
 {label:'ActRIIA / ACVR2A', ref:['C','F'], pred:['E','F'], blue:'#6baed6', orange:'#fdae6b', key:'ActRIIA_chain_CA_RMSD_angstrom'}
];
const assemblyMetrics = [
 ['global_CA_RMSD_angstrom','Assembly Cα RMSD ↓','Å'],
 ['assembly_CA_lddt','Assembly Cα lDDT ↑',''],
 ['CA_interchain_contact_recall','Interchain contact recall ↑',''],
 ['CA_interchain_contact_precision','Interchain contact precision ↑','']
];
const clashMetrics = [
 ['clash_count','Total clashes ↓','count'],
 ['intrachain_clash_count','Within-chain clashes ↓','count'],
 ['interchain_clash_count','Between-chain clashes ↓','count'],
 ['clashes_per_1000_heavy_atoms','Clashes / 1,000 heavy atoms ↓',''],
 ['max_overlap_angstrom','Worst overlap ↓','Å'],
 ['CA_nonadjacent_clash_count','Nonadjacent Cα clashes ↓','count']
];
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v, unit='') => typeof v==='number' && Number.isFinite(v) ? v.toFixed(unit==='count'?1:unit==='Å'?2:3) : '—';
const statCells = (s={}, unit='') => `<td>${fmt(s.mean,unit)} ± ${fmt(s.sd,unit)}</td><td>${fmt(s.median,unit)}</td><td>${fmt(s.min,unit)}–${fmt(s.max,unit)}</td>`;
function statsTable(rows, summary){return `<div class="scroll"><table class="stats"><thead><tr><th>Metric</th><th>Mean ± sample SD</th><th>Median</th><th>Range</th></tr></thead><tbody>${rows.map(([key,label,unit])=>`<tr><th scope="row">${esc(label)}${unit&&unit!=='count'?' ('+unit+')':''}</th>${statCells(summary[key],unit)}</tr>`).join('')}</tbody></table></div>`;}
function renderCard(model,index){
 const samples=model.samples||[], card=document.createElement('section');card.className='card';
 const legend=`<table class="legend" aria-label="Protein color key"><thead><tr><th>Protein · 2 copies</th><th>Experimental 2GOO</th><th>SimpleFold ${esc(model.name)}</th></tr></thead><tbody>${groups.map(g=>`<tr><th scope="row">${esc(g.label)}</th><td><i style="background:${g.blue}"></i>Chains ${g.ref.join(', ')}</td><td><i style="background:${g.orange}"></i>Chains ${g.pred.join(', ')}</td></tr>`).join('')}</tbody></table>`;
 card.innerHTML=`<h2>SimpleFold ${esc(model.name)}</h2><p class="subtle">${samples.length} available samples${samples.length?' · solid cartoons':''}${model.status?' · '+esc(model.status):''}</p>${legend}<div class="toolbar"><label>Mode <select class="mode"><option value="overlay">Overlay</option><option value="reference">Reference only</option><option value="predictions">Predictions only</option></select></label><label>Samples <select class="sample"><option value="all">All ${samples.length} samples</option>${samples.map((s,j)=>`<option value="${j}">Sample ${esc(s.id)}</option>`).join('')}</select></label><button class="reset" type="button">Reset view</button></div><div class="toolbar components" aria-label="Visible proteins">${groups.map((g,j)=>`<label><input type="checkbox" data-group="${j}" checked>${esc(g.label)}</label>`).join('')}<label><input type="checkbox" class="ends" checked>Receptor C-ends · red dots</label></div><p class="subtle">Red dots mark the last available Cα of each receptor chain. These are extracellular fragments; their membrane-spanning helices are absent. Unresolved reference tails are not shown.</p><div class="view" id="viewer-${index}"></div><p class="selected"></p><h3>Assembly agreement</h3>${statsTable(assemblyMetrics,model.summary||{})}<h3>Steric clashes ↓</h3>${statsTable(clashMetrics,model.summary||{})}<p class="subtle">Nonbonded heavy-atom overlap &gt;0.4 Å, excluding covalent neighbors through three bonds, hydrogens and zero-occupancy atoms. Geometric counts; not MolProbity clashscore. Reference: ${payload.reference_clashes?.clash_count??'—'} total (${payload.reference_clashes?.intrachain_clash_count??'—'} within-chain, ${payload.reference_clashes?.interchain_clash_count??'—'} between-chain); ${fmt(payload.reference_clashes?.clashes_per_1000_heavy_atoms)} per 1,000 atoms. Reference has unresolved atoms, so raw totals have different coverage.</p><h3>BMP2 dimer geometry</h3>${statsTable([['BMP2_dimer_SG_distance_angstrom','Cys78–Cys78 sulfur distance','Å']],model.summary||{})}<p class="subtle">Reference sulfur distance: ${fmt(payload.reference_BMP2_dimer_SG_distance_angstrom,'Å')} Å. Shorter is not automatically better; this checks the native disulfide site without enforcing it.</p><h3>Individual folds · separate chain fits</h3>${statsTable(groups.map(g=>[g.key,g.label+' Cα RMSD ↓','Å']),model.summary||{})}<h3>Ensemble diversity ↔</h3>${statsTable([['pairwise','Pairwise Cα RMSD ↔','Å']],{pairwise:(model.diversity||{}).summary})}<p class="subtle">${(model.diversity||{}).pair_count||0} pairwise comparisons. Variation is measured after aligning each pair over the matched Cα residues.</p><details><summary>Sample metrics and structure files</summary><div class="scroll"><table class="stats"><thead><tr><th>Sample</th><th>Assembly RMSD ↓ (Å)</th><th>Cα lDDT ↑</th><th>Contact recall ↑</th><th>Clashes ↓</th><th>Between chains ↓</th><th>PDB files</th></tr></thead><tbody>${samples.map(s=>`<tr><th scope="row">${esc(s.id)}</th><td>${fmt((s.metrics||{}).global_CA_RMSD_angstrom,'Å')}</td><td>${fmt((s.metrics||{}).assembly_CA_lddt)}</td><td>${fmt((s.metrics||{}).CA_interchain_contact_recall)}</td><td>${(s.metrics||{}).clash_count??'—'}</td><td>${(s.metrics||{}).interchain_clash_count??'—'}</td><td><a href="${esc(s.aligned_pdb)}">Aligned</a>${s.raw_link?` · <a href="${esc(s.raw_link)}">Raw</a>`:''}</td></tr>`).join('')}</tbody></table></div></details>`;
 document.getElementById('grid').appendChild(card);
 const selected=card.querySelector('.selected');
 const selector=card.querySelector('.sample');
 const mode=card.querySelector('.mode');
 function showStats(){const value=selector.value;if(mode.value==='reference'){selected.textContent='Experimental 2GOO only. Tables summarize the complete prediction ensemble.';}else if(value==='all'){selected.textContent=`All ${samples.length} samples${mode.value==='overlay'?' over the experimental reference':''}. Tables summarize the complete ensemble.`;}else{const s=samples[Number(value)],m=s.metrics||{};selected.textContent=`Sample ${s.id}: assembly Cα RMSD ${fmt(m.global_CA_RMSD_angstrom,'Å')} Å · Cα lDDT ${fmt(m.assembly_CA_lddt)} · contact recall ${fmt(m.CA_interchain_contact_recall)} · ${m.clash_count??'—'} clashes (${m.interchain_clash_count??'—'} between chains). Tables summarize the complete ensemble.`;}}
 showStats();
 if(!samples.length){card.querySelector('.view').innerHTML='<p class="empty">Sampling has not completed for this model.</p>';return;}
 if(typeof $3Dmol==='undefined'){card.querySelector('.view').innerHTML='<p class="empty">3D viewer unavailable. Connect to the internet to load 3Dmol, then refresh. Metrics and structure links remain available.</p>';selector.onchange=showStats;mode.onchange=showStats;return;}
 const viewer=$3Dmol.createViewer(card.querySelector('.view'),{backgroundColor:'white'});
 const reference=viewer.addModel(payload.reference,'pdb');
 const predictions=samples.map(s=>viewer.addModel(s.pdb,'pdb'));
 const ends=card.querySelector('.ends');
 let endLabels=[];
 function markReceptorEnds(structure,chains,label,withLabels){
  chains.forEach(chain=>{
   const atoms=structure.selectedAtoms({chain,atom:'CA'});
   if(!atoms.length)return;
   const atom=atoms.reduce((last,a)=>Number(a.resi)>=Number(last.resi)?a:last);
   structure.setStyle({chain,resi:atom.resi,atom:'CA'},{sphere:{color:'#d7191c',radius:1.8}},true);
   if(withLabels)endLabels.push(viewer.addLabel(label,{
    position:{x:atom.x,y:atom.y,z:atom.z},fontSize:11,fontColor:'#b51219',
    backgroundColor:'white',backgroundOpacity:0.85,showBackground:true,
    borderThickness:0,inFront:true}));
  });
 }
 function applyStyles(){
  endLabels.forEach(label=>viewer.removeLabel(label));endLabels=[];
  reference.setStyle({},{});predictions.forEach(m=>m.setStyle({},{}));
  card.querySelectorAll('[data-group]').forEach(input=>{if(!input.checked)return;const g=groups[Number(input.dataset.group)];reference.setStyle({chain:g.ref},{cartoon:{color:g.blue}});predictions.forEach(m=>m.setStyle({chain:g.pred},{cartoon:{color:g.orange}}));});
  if(mode.value!=='predictions')reference.show();else reference.hide();
  selector.disabled=mode.value==='reference';
  predictions.forEach((m,j)=>{if(mode.value!=='reference'&&(selector.value==='all'||Number(selector.value)===j))m.show();else m.hide();});
  if(ends.checked){
   card.querySelectorAll('[data-group]').forEach(input=>{
    const groupIndex=Number(input.dataset.group);
    if(!input.checked||groupIndex===0)return; // BMP2 is a secreted ligand.
    const g=groups[groupIndex],name=groupIndex===1?'BMPR1A':'ActRIIA';
    if(mode.value!=='predictions')markReceptorEnds(reference,g.ref,name,true);
    predictions.forEach((m,j)=>{
     if(mode.value==='reference'||(selector.value!=='all'&&Number(selector.value)!==j))return;
     const withLabels=mode.value==='predictions'&&(selector.value!=='all'||samples.length===1);
     markReceptorEnds(m,g.pred,name,withLabels);
    });
   });
  }
  viewer.render();showStats();
 }
 selector.onchange=applyStyles;mode.onchange=applyStyles;ends.onchange=applyStyles;card.querySelectorAll('[data-group]').forEach(i=>{i.onchange=applyStyles;});card.querySelector('.reset').onclick=()=>{viewer.zoomTo();viewer.render();};
 applyStyles();viewer.zoomTo();viewer.render();
}
(payload.models||[]).forEach(renderCard);
document.getElementById('coverage').textContent=`Reference coverage: ${payload.matched_reference_CA||'—'} resolved Cα positions; predictions: ${payload.prediction_CA||'—'} sequence residues. Statistics report sample SD (n − 1 denominator); a dash means unavailable.`;
if(typeof $3Dmol==='undefined')document.getElementById('load-error').innerHTML='<p class="loaderror">The 3Dmol library could not load. Metrics are available below; refresh with an internet connection to view the structures.</p>';
</script></body></html>'''


def build(analysis_dir: Path) -> Path:
    """Embed PDBs without changing coordinates; write index.html beside metrics."""
    analysis_dir = analysis_dir.resolve()
    metrics = json.loads((analysis_dir / 'metrics.json').read_text())
    reference_path = analysis_dir / metrics.get('reference_pdb', 'reference.pdb')
    metrics['reference'] = reference_path.read_text()
    for model in metrics.get('models', []):
        for sample in model.get('samples', []):
            sample['pdb'] = (analysis_dir / sample['aligned_pdb']).read_text()
            source = sample.get('raw_pdb') or sample.get('source')
            if source:
                # The analyzer may give an absolute path; use a relative browser link.
                import os
                source_path = Path(source)
                if not source_path.is_absolute():
                    source_path = analysis_dir / source_path
                sample['raw_link'] = os.path.relpath(source_path, analysis_dir)
    # Prevent PDB remarks or metadata from terminating the script element.
    encoded = json.dumps(metrics, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    output = analysis_dir / 'index.html'
    output.write_text(HTML.replace('__PAYLOAD__', encoded))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', required=True, type=Path)
    args = parser.parse_args()
    print(build(args.analysis_dir))


if __name__ == '__main__':
    main()
