#!/usr/bin/env python3
"""Shared three-checkpoint geometry tables and per-target inspection."""
import argparse
import html
import json
from pathlib import Path
import shutil

LABELS = {
    'molprobity_clashscore': 'MolProbity clashscore ↓',
    'rama_outlier_percent': 'Ramachandran outliers ↓ (%)',
    'rotamer_outlier_percent': 'Rotamer outliers ↓ (%)',
    'bonds_outlier_percent': 'Bond-length outliers ↓ (%)',
    'angles_outlier_percent': 'Bond-angle outliers ↓ (%)',
    'chiralities_outlier_percent': 'Chirality outliers ↓ (%)',
    'planarities_outlier_percent': 'Planarity outliers ↓ (%)',
    'heavy_atom_overlaps': 'Heavy-atom overlaps ↓',
    'overlaps_per_1000_heavy_atoms': 'Overlaps / 1,000 heavy atoms ↓',
    'backbone_backbone_overlaps': 'Backbone–backbone overlaps ↓',
    'backbone_sidechain_overlaps': 'Backbone–side-chain overlaps ↓',
    'sidechain_sidechain_overlaps': 'Side-chain–side-chain overlaps ↓',
    'interchain_overlaps': 'Interchain overlaps ↓',
    'clashing_residue_percent': 'Residues involved in overlaps ↓ (%)',
    'max_overlap_angstrom': 'Maximum overlap ↓ (Å)',
    'p95_overlap_angstrom': '95th-percentile overlap ↓ (Å)',
    'severe_overlaps': 'Overlaps ≥1 Å ↓',
    'incomplete_residue_percent': 'Incomplete observed residues (%)',
    'peptide_extreme_length_percent': 'Extreme peptide C–N lengths ↓ (%)',
    'peptide_links_over_2_angstrom': 'Peptide C–N links >2 Å ↓',
}
ARMS = {'original': 'Original 100M', 'six_only': 'Six-only fine-tuning',
        'mixed': '90% PDB / 10% complexes', 'refined_target': 'Refined training targets',
        'experimental_reference': 'Experimental references'}


def summary_table(summary, cohort, references=True):
    groups = [g for g in summary['groups'] if g['cohort'] == cohort and
              (references or g['arm'] in ['original', 'six_only', 'mixed'])]
    keys = [k for k in LABELS if any(k in g['metrics'] for g in groups)]
    heads = ''.join(f'<th>{html.escape(ARMS[g["arm"]])}<br><small>n={g["structures"]}</small></th>' for g in groups)
    rows = []
    for key in keys:
        cells = []
        for g in groups:
            v = g['metrics'].get(key)
            cell = '—' if v is None or v['mean'] is None else f'{v["mean"]:.2f} ± {v["std"]:.2f}'
            if v and v['n'] != g['structures']:
                cell += f' <small>(n={v["n"]})</small>'
            cells.append(f'<td>{cell}</td>')
        rows.append(f'<tr><th>{html.escape(LABELS[key])}</th>'+''.join(cells)+'</tr>')
    return '<div class="scroll"><table><thead><tr><th>Metric · mean ± SD</th>'+heads+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.run/'geometry-summary.json').read_text())
    records = json.loads((args.run/'geometry-results.json').read_text())
    data = dict(labels=LABELS, arms=ARMS, records=[{k: r[k] for k in
        ['cohort', 'arm', 'id', 'sample', 'sha256', 'metrics']} for r in records])
    status = ('MolProbity and coordinate checks complete.' if 'molprobity_clashscore' in records[0]['metrics']
              else 'Coordinate checks complete. Hydrogen-inclusive MolProbity checks are pending.')
    source = json.dumps(data).replace('</', '<\\/')
    tables = '<h2>Six synthetic training complexes</h2>'+summary_table(summary, 'complex')
    tables += '<h2>24 experimental CAMEO targets</h2>'+summary_table(summary, 'cameo')
    page = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Training comparison · structure geometry</title>
<style>body{font:15px/1.5 system-ui;color:#233344;background:#f5f7fa;margin:0}main{max-width:1500px;margin:auto;padding:24px}table{border-collapse:collapse;width:100%;background:white;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:8px 12px;border-bottom:1px solid #dce3eb;text-align:left;white-space:nowrap}small{font-weight:normal}.scroll{overflow:auto}.note{padding:12px;background:#eaf0f6;border-radius:7px}select{font:inherit;padding:6px;margin-right:15px}h1{margin-bottom:8px}h2{margin-top:28px}a{color:#244c73}</style></head>
<body><main><h1>Structure geometry · fine-tuning comparison</h1><p class="note">Existing unguided predictions: original released 100M, six-only fine-tuning and 90% PDB / 10% complex fine-tuning. No prediction is relaxed or changed by validation. Both tuned models used 600 updates, but synthetic exposure was 600 versus 60. CAMEO geometry covers full predicted sequences; experimental references may have unresolved residues.</p>
<p><strong>__STATUS__</strong> MolProbity adds and optimizes hydrogens on temporary copies, without side-chain flips. Covalent outliers use a 4σ threshold on inferred restraints. Independent peptide C–N checks flag lengths outside1.1–1.6 Å and count links &gt;2 Å, respecting residue-number gaps. Our separate heavy-atom overlap calculation uses overlaps &gt;0.4 Å and excludes covalent 1–2/1–3/1–4 neighbors; its normalized count is not MolProbity clashscore. Reference columns provide context: training targets were Rosetta-refined, whereas generated samples were not.</p>
__TABLES__
<h2>Inspect one target</h2><label>Dataset <select id="cohort"><option value="complex">Six training complexes</option><option value="cameo">CAMEO</option></select></label><label>Target <select id="target"></select></label><div id="target-table" class="scroll"></div>
<p><a href="geometry-results.csv">Per-structure metrics CSV</a> · <a href="geometry-summary.json">Summary and paired confidence intervals</a> · <a href="geometry-results.json">Full diagnostics and residue locations</a></p>
<p>DockQ assesses agreement with our synthetic training targets. These checks assess physical geometry independently of target agreement. Lower is better for outlier and overlap measures.</p>
</main><script>const data=__DATA__,cohort=document.getElementById('cohort'),target=document.getElementById('target'),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function fill(){target.innerHTML=[...new Set(data.records.filter(r=>r.cohort===cohort.value).map(r=>r.id))].sort().map(id=>'<option>'+esc(id)+'</option>').join('');render();}
function render(){const rows=data.records.filter(r=>r.cohort===cohort.value&&r.id===target.value),keys=Object.keys(data.labels).filter(k=>rows.some(r=>r.metrics[k]!==undefined));document.getElementById('target-table').innerHTML='<table><tr><th>Model · saved sample</th>'+keys.map(k=>'<th>'+esc(data.labels[k])+'</th>').join('')+'</tr>'+rows.map(r=>'<tr><th>'+esc(data.arms[r.arm])+' · '+(r.arm.includes('reference')||r.arm==='refined_target'?'reference':r.sample+1)+'</th>'+keys.map(k=>'<td>'+(r.metrics[k]==null?'—':Number(r.metrics[k]).toFixed(2))+'</td>').join('')+'</tr>').join('')+'</table>';}
cohort.onchange=fill;target.onchange=render;fill();</script></body></html>'''
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ['geometry-results.json', 'geometry-summary.json', 'geometry-results.csv']:
        if (args.output/name).resolve() != (args.run/name).resolve():
            shutil.copy2(args.run/name, args.output/name)
    (args.output/'index.html').write_text(page.replace('__STATUS__', status).replace('__TABLES__', tables).replace('__DATA__', source))
    print(args.output/'index.html')


if __name__ == '__main__':
    main()
