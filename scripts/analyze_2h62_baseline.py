#!/usr/bin/env python3
"""Compare six-chain 2H62 baseline samples against its four observed chains.

The absent receptor copies have no experimental coordinates in this reference
and never contribute to native-contact or placement metrics. This is analysis
of unconditional baseline samples, not coordinate-conditioned inpainting.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

import gemmi
import numpy as np

try:
    from .compare_structures import (ChainCA, amino_acid, altloc_priority,
                                    kabsch, reference_ca, transformed_pdb)
    from .complex_interfaces import _contacts, _read_atoms
    from .structure_clashes import analyze_clashes
except ImportError:
    from compare_structures import (ChainCA, amino_acid, altloc_priority,
                                   kabsch, reference_ca, transformed_pdb)
    from complex_interfaces import _contacts, _read_atoms
    from structure_clashes import analyze_clashes

COMPONENTS = dict(A="BMP2", B="BMP2", C="BMPR1A", D="BMPR1A", E="ACVR2B", F="ACVR2B")
RECEPTORS = (("BMPR1A", "C", ("C", "D")), ("ACVR2B", "D", ("E", "F")))


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", Path(path).name)]


def read_prediction(path, references):
    result = {}
    for chain in gemmi.read_structure(str(path))[0]:
        letters, xyz, authors = [], [], []
        for residue in chain:
            letter = amino_acid(residue.name)
            if letter is None:
                continue
            atoms = [a for a in residue if a.name == "CA" and a.occ > 0]
            if not atoms:
                raise ValueError(f"Missing occupied CA: {chain.name}:{residue.seqid}")
            atom = min(atoms, key=lambda a: altloc_priority(a.occ, a.altloc.strip("\x00")))
            letters.append(letter)
            xyz.append([atom.pos.x, atom.pos.y, atom.pos.z])
            authors.append(str(residue.seqid))
        if letters:
            sequence = "".join(letters)
            result[chain.name] = ChainCA(sequence, np.array(xyz), list(range(len(xyz))),
                                       authors, chain.name, sequence)
    if set(result) != set(COMPONENTS):
        raise ValueError(f"Expected six chains A–F in {path}")
    for pc, rc in (("A", "A"), ("B", "B"), ("C", "C"), ("D", "C"), ("E", "D"), ("F", "D")):
        if result[pc].sequence != references[rc].full_sequence:
            raise ValueError(f"Prediction {pc} differs from reference {rc} full sequence")
    return result


def rmsd(first, second):
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def score_prediction(references, predicted, reference_atoms, predicted_atoms):
    """Fit ligands once; score only one observed receptor site of each type."""
    fixed = np.concatenate([references[c].coordinates for c in ("A", "B")])
    candidates = []
    for ligand_copies in (("A", "B"), ("B", "A")):
        mapping = dict(zip(("A", "B"), ligand_copies))
        moving = np.concatenate([predicted[mapping[c]].coordinates[references[c].full_positions]
                                 for c in ("A", "B")])
        rotation, translation, distances = kabsch(moving, fixed)
        candidates.append((float(np.sqrt(np.mean(distances ** 2))), mapping, rotation, translation))
    ligand_rmsd, mapping, rotation, translation = min(candidates, key=lambda item: item[0])
    receptors, unscored = [], []
    for component, rc, copies in RECEPTORS:
        native = references[rc]
        pc = min(copies, key=lambda c: rmsd(
            predicted[c].coordinates[native.full_positions] @ rotation + translation,
            native.coordinates))
        mapping[rc] = pc
        unscored.append(next(c for c in copies if c != pc))
        moving = predicted[pc].coordinates[native.full_positions]
        _, _, fold_distances = kabsch(moving, native.coordinates)
        # Restrict prediction contacts to atom names and full-sequence positions
        # actually observed in this experimental subcomplex. No fabricated atoms
        # or the unobserved second receptor copy enter either contact set.
        paired_atoms = {}
        for (chain, position, name), xyz in reference_atoms.items():
            if chain not in ("A", "B", rc):
                continue
            key = (mapping[chain], position, name)
            if key in predicted_atoms:
                paired_atoms[(chain, position, name)] = predicted_atoms[key]
        native_contacts = _contacts(reference_atoms, rc, ("A", "B"))
        predicted_contacts = _contacts(paired_atoms, rc, ("A", "B"))
        recovered = len(native_contacts & predicted_contacts)
        receptors.append(dict(
            receptor=component, reference_chain=rc, prediction_chain=pc,
            placement_CA_RMSD_angstrom=rmsd(moving @ rotation + translation, native.coordinates),
            separately_fitted_CA_RMSD_angstrom=float(np.sqrt(np.mean(fold_distances ** 2))),
            observed_CA_count=len(native.coordinates), full_sequence_length=len(native.full_sequence),
            native_contacts=len(native_contacts), predicted_contacts=len(predicted_contacts),
            recovered_contacts=recovered,
            native_contact_recovery=recovered / len(native_contacts) if native_contacts else None,
            native_contact_precision=recovered / len(predicted_contacts) if predicted_contacts else None,
            matched_observed_heavy_atoms=len(paired_atoms),
        ))
    return dict(BMP2_dimer_CA_RMSD_angstrom=ligand_rmsd, mapping=mapping,
                receptors=receptors, unscored_receptor_chains=unscored), rotation, translation


def write_viewer(output_dir, reference_pdb, rows):
    data = dict(reference=reference_pdb.read_text(), samples=[
        dict(**row, pdb=(output_dir / row["aligned_file"]).read_text()) for row in rows])
    payload = json.dumps(data).replace("</", "<\\/")
    source_links = "".join(
        f'<tr><td>Sample {r["sample"]}</td><td>{html.escape(r["source_name"])}</td>'
        f'<td><a href="{r["raw_file"]}">raw PDB</a> · <a href="{r["aligned_file"]}">aligned PDB</a></td></tr>'
        for r in rows)
    page = """<!doctype html><html><head><meta charset="utf-8"><title>2h62-inpaint · baseline</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script><style>
body{font:15px system-ui;color:#233344;margin:24px auto;max-width:1200px;padding:0 18px}h1{font-size:25px}
#viewer{height:580px;position:relative;border:1px solid #d7dfe6;border-radius:8px;margin:16px 0}
select,label{margin-right:14px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #dfe5eb}
.note{background:#eef4f8;padding:14px;border-radius:8px;line-height:1.5}.legend{display:flex;gap:16px;margin-top:12px;flex-wrap:wrap}.swatch{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:5px}a{color:#245f91}</style></head><body>
<h1>2h62-inpaint · six-chain baseline</h1><p class="note">Experimental 2H62 contains two BMP2 chains, one BMPR1A, and one ACVR2B (2:1:1). These baseline predictions contain two copies of each (2:2:2). The two additional receptors have no experimental coordinates here and are <b>unscored</b> for native interfaces. “Inpaint” names the completion project; these samples use ordinary sequence-only inference.</p>
<label>Samples <select id="sample"></select></label><label>View <select id="mode"><option value="overlay">Overlay</option><option value="pred">Predictions only</option><option value="ref">Experimental reference only</option></select></label>
<label><input id="ends" type="checkbox" checked>Receptor ends</label><button id="reset">Reset view</button>
<div class="legend" id="legend"></div><div id="viewer"></div><div id="stats"></div>
<p>All structures share a BMP2-dimer fit; receptor placement scores retain that frame. Receptor fold RMSD uses a separate fit. Contact recovery and precision use observed occupied heavy atoms at ≤4 Å, matched by sequence position and atom name. One nearest receptor copy of each type is selected for the observed interface; extra copies contribute to whole-prediction clashes but not native-interface scores. Clash counts are geometric heavy-atom overlaps, not MolProbity clashscore. No hotspot numbering is inferred from ACVR2A for ACVR2B.</p>
<p><a href="metrics.json">Metrics JSON</a> · <a href="reference.pdb">Four-chain reference PDB</a> · <a href="source.cif">Original mmCIF</a></p>
<details><summary>Sample metrics and structure files</summary><table><tr><th>Sample</th><th>Source filename</th><th>Structures</th></tr>__LINKS__</table></details>
<script>const data=__DATA__;
const viewer=$3Dmol.createViewer('viewer',{backgroundColor:'white'});
const palette={ref:{BMP2:'#155b87',BMPR1A:'#427fa6',ACVR2B:'#85b9d3'},pred:{BMP2:'#b65316',BMPR1A:'#eb8424',ACVR2B:'#e8b338'}};
const refGroups={A:'BMP2',B:'BMP2',C:'BMPR1A',D:'ACVR2B'},predGroups={A:'BMP2',B:'BMP2',C:'BMPR1A',D:'BMPR1A',E:'ACVR2B',F:'ACVR2B'};
const reference=viewer.addModel(data.reference,'pdb'),models=data.samples.map(s=>viewer.addModel(s.pdb,'pdb'));
function style(model,groups,colors){for(const [chain,group] of Object.entries(groups))model.setStyle({chain},{cartoon:{color:colors[group],opacity:1}});}
style(reference,refGroups,palette.ref);models.forEach(m=>style(m,predGroups,palette.pred));
document.getElementById('legend').innerHTML=['ref','pred'].flatMap(k=>Object.entries(palette[k]).map(([name,color])=>`<span><i class="swatch" style="background:${color}"></i>${k==='ref'?'Reference':'Prediction'} ${name}</span>`)).join('');
const sample=document.getElementById('sample');sample.innerHTML=`<option value="all">All ${data.samples.length} samples</option>`+data.samples.map((s,i)=>`<option value="${i}">Sample ${s.sample}</option>`).join('');
sample.value='0';
function ends(model,groups){for(const [chain,name] of Object.entries(groups)){if(name==='BMP2')continue;const atoms=model.selectedAtoms({chain,atom:'CA'});if(!atoms.length)continue;const a=atoms.reduce((x,y)=>x.resi>y.resi?x:y);viewer.addSphere({center:{x:a.x,y:a.y,z:a.z},radius:1.1,color:'#c93232'});viewer.addLabel(name,{position:{x:a.x,y:a.y,z:a.z},fontColor:'#b72222',fontSize:12,backgroundOpacity:0,inFront:true});}}
const number=x=>x===null?'—':x.toFixed(2),percent=x=>x===null?'—':(100*x).toFixed(1)+'%';
function render(){const mode=document.getElementById('mode').value,selected=sample.value==='all'?data.samples.map((_,i)=>i):[+sample.value];reference[mode==='pred'?'hide':'show']();models.forEach((m,i)=>m[mode==='ref'||!selected.includes(i)?'hide':'show']());viewer.removeAllShapes();viewer.removeAllLabels();if(document.getElementById('ends').checked){if(mode!=='pred')ends(reference,refGroups);if(mode!=='ref')selected.forEach(i=>ends(models[i],predGroups));}
if(mode==='ref'){document.getElementById('stats').innerHTML='<p>Experimental reference only: four observed chains (two BMP2, one BMPR1A, one ACVR2B). The two missing receptor copies have no experimental coordinates and are not evaluated here.</p>';viewer.render();return;}
let stats='<h3>Observed receptor interfaces · '+(sample.value==='all'?'all samples':'sample '+data.samples[+sample.value].sample)+'</h3><table><tr><th>Sample / receptor mapping</th><th>Native recovery ↑</th><th>Contact precision ↑</th><th>Placement CA RMSD ↓ (Å)</th><th>Fold CA RMSD ↓ (Å)</th></tr>';
for(const i of selected){const s=data.samples[i];for(const r of s.receptors)stats+=`<tr><td>${s.sample} · ${r.receptor} · reference ${r.reference_chain} → prediction ${r.prediction_chain}</td><td>${percent(r.native_contact_recovery)} (${r.recovered_contacts}/${r.native_contacts})</td><td>${percent(r.native_contact_precision)} (${r.recovered_contacts}/${r.predicted_contacts})</td><td>${number(r.placement_CA_RMSD_angstrom)}</td><td>${number(r.separately_fitted_CA_RMSD_angstrom)}</td></tr>`;}
stats+='</table>';if(selected.length===1){const s=data.samples[selected[0]];stats+=`<p>BMP2 dimer CA RMSD ↓: ${number(s.BMP2_dimer_CA_RMSD_angstrom)} Å. Unscored extra receptor chains: ${s.unscored_receptor_chains.join(', ')}. Whole-prediction clashes ↓: ${s.clashes.clash_count} (${s.clashes.interchain_clash_count} interchain); ${number(s.clashes.clashes_per_1000_heavy_atoms)} per 1000 heavy atoms.</p>`;}document.getElementById('stats').innerHTML=stats;viewer.render();}
sample.onchange=render;document.getElementById('mode').onchange=render;document.getElementById('ends').onchange=render;document.getElementById('reset').onclick=()=>{viewer.zoomTo({model:reference});viewer.render();};render();viewer.zoomTo({model:reference});viewer.render();
</script></body></html>"""
    (output_dir / "index.html").write_text(page.replace("__DATA__", payload).replace("__LINKS__", source_links))


def analyze(args):
    references = {c: reference_ca(args.source_cif, c) for c in "ABCD"}
    reference_atoms = _read_atoms(args.source_cif, references)
    paths = sorted(args.prediction_dir.glob("*.pdb"), key=natural_key)
    if not paths:
        raise ValueError(f"No prediction PDBs in {args.prediction_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("raw_predictions", "aligned_predictions"):
        (args.output_dir / folder).mkdir(exist_ok=True)
    shutil.copy2(args.source_cif, args.output_dir / "source.cif")
    structure = gemmi.read_structure(str(args.source_cif))
    for chain in list(structure[0]):
        if chain.name not in references:
            del structure[0][chain.name]
    # Do not render crystal waters or ligands as protein ground truth.
    for chain in structure[0]:
        for index in range(len(chain) - 1, -1, -1):
            if amino_acid(chain[index].name) is None:
                del chain[index]
    reference_pdb = args.output_dir / "reference.pdb"
    structure.write_pdb(str(reference_pdb))
    rows = []
    for index, path in enumerate(paths, 1):
        predicted = read_prediction(path, references)
        metrics, rotation, translation = score_prediction(
            references, predicted, reference_atoms, _read_atoms(path, predicted))
        raw = Path("raw_predictions") / path.name
        aligned = Path("aligned_predictions") / path.name
        shutil.copy2(path, args.output_dir / raw)
        transformed_pdb(path, args.output_dir / aligned, rotation, translation)
        metrics.update(sample=index, source_name=path.name, raw_file=str(raw), aligned_file=str(aligned),
                       raw_prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       clashes=analyze_clashes(path, chain_components=COMPONENTS))
        rows.append(metrics)
    record = dict(project="2h62-inpaint", stage="sequence-only six-chain baseline",
                  source_cif_sha256=hashlib.sha256(args.source_cif.read_bytes()).hexdigest(),
                  reference_stoichiometry="2:1:1", prediction_stoichiometry="2:2:2",
                  native_reference_chains={"BMP2": ["A", "B"], "BMPR1A": ["C"], "ACVR2B": ["D"]},
                  observed_CA_counts={c: len(chain.coordinates) for c, chain in references.items()},
                  full_sequence_lengths={c: len(chain.full_sequence) for c, chain in references.items()},
                  methods={
                      "fit": "Proper Kabsch fit of observed BMP2 dimer CAs, trying both equivalent ligand-copy assignments and retaining the minimum ligand RMSD.",
                      "copy_assignment": "Nearest CA receptor copy in the fixed BMP2 frame, independently for the one observed BMPR1A and one observed ACVR2B site.",
                      "contacts": "Residue-pair contact at <=4 angstrom occupied heavy-atom distance. Prediction atoms are restricted to sequence-position/atom-name intersections with the observed reference. Native denominators retain all observed native contacts; undefined ratios are null.",
                      "unknown_copies": "The extra BMPR1A and ACVR2B copies are included in geometric clash counts but have no native placement or interface score.",
                  },
                  missing_receptor_copies_are_native_unscored=True, samples=rows)
    (args.output_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    write_viewer(args.output_dir, reference_pdb, rows)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cif", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args)
    print(f"Analyzed {len(result['samples'])} samples: {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
