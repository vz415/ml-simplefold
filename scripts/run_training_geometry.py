#!/usr/bin/env python3
"""Validate existing predictions; no sampling, refinement or coordinate updates."""
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.structure_clashes import analyze_clashes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.partial')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def inputs(cfg):
    """One original arm, two tuned arms, plus both kinds of reference."""
    selection = json.loads((ROOT / cfg.selection).read_text())['records']
    groups = {r['id']: r['groups'] for r in selection}
    rows = []
    for cohort, runs in [('complex', cfg.complex_runs), ('cameo', cfg.cameo_runs)]:
        for arm, job in runs.items():
            run = Path(cfg.run_root) / job
            desired = 'pretrained' if arm == 'original' else 'fine_tuned'
            for item in json.loads((run / 'prediction-receipt.json').read_text()):
                if item['condition'] != desired:
                    continue
                rows.append(dict(cohort=cohort, arm=arm, id=item['id'], sample=item['sample'],
                    path=str(run / item['file']), sha256=item['sha256'],
                    groups=groups[item['id']] if cohort == 'complex' else {}))
    for item in selection:
        rows.append(dict(cohort='complex', arm='refined_target', id=item['id'], sample=0,
            path=str(Path(cfg.complex_dataset) / item['file']), sha256=item['sha256'], groups=item['groups']))
    run = Path(cfg.run_root) / cfg.cameo_runs.original
    receipt = json.loads((run / 'dataset-receipt.json').read_text())
    for item in receipt['prepared_records']:
        rows.append(dict(cohort='cameo', arm='experimental_reference', id=item['id'], sample=0,
            path=str(run / 'references' / f'{item["id"]}.pdb'),
            sha256=item['reference_sha256'], groups={}))
    return rows


def analyze_one(task):
    row, backend, cache_dir, method_hash = task
    path = Path(row['path'])
    assert digest(path) == row['sha256'], path
    group_hash = hashlib.sha256(json.dumps(row['groups'], sort_keys=True).encode()).hexdigest()[:12]
    cached = Path(cache_dir) / method_hash / f'{row["sha256"]}-{group_hash}.json'
    if cached.exists():
        return json.loads(cached.read_text())
    coordinate = analyze_clashes(path, row['groups'])
    result = dict(input_sha256=row['sha256'], coordinate=coordinate)
    if backend == 'molprobity':
        from molprobity_geometry import analyze
        result['molprobity'] = analyze(path)
    save(cached, result)
    return result


def metrics(result):
    c = result['coordinate']
    values = dict(heavy_atom_overlaps=c['clash_count'], interchain_overlaps=c['interchain_clash_count'],
        overlaps_per_1000_heavy_atoms=c['clashes_per_1000_heavy_atoms'],
        max_overlap_angstrom=c['max_overlap_angstrom'], p95_overlap_angstrom=c['overlap_p95_angstrom'],
        severe_overlaps=c['overlaps_at_least_1_angstrom'],
        clashing_residue_percent=c['clashing_residue_percent'],
        backbone_backbone_overlaps=c['clashes_by_atom_class']['backbone_backbone'],
        backbone_sidechain_overlaps=c['clashes_by_atom_class']['backbone_sidechain'],
        sidechain_sidechain_overlaps=c['clashes_by_atom_class']['sidechain_sidechain'])
    values.update(peptide_extreme_length_percent=c['peptide_links']['extreme_length_percent'],
                  peptide_links_over_2_angstrom=c['peptide_links']['links_over_2_angstrom'])
    if 'molprobity' in result:
        m = result['molprobity']
        values.update(molprobity_clashscore=m['clashscore'],
            rama_outlier_percent=m['ramachandran']['outlier_percent'],
            rotamer_outlier_percent=m['rotamers']['outlier_percent'],
            incomplete_residue_percent=m['missing_atoms']['incomplete_percent'])
        for key in ['bonds', 'angles', 'chiralities', 'planarities']:
            values[key+'_outlier_percent'] = m['geometry'][key]['outlier_percent']
            values[key+'_outliers'] = m['geometry'][key]['outliers']
    return values


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['cohort'], row['arm']].append(row)
    summary = []
    for (cohort, arm), group in grouped.items():
        stats = {}
        for key in group[0]['metrics']:
            values = [r['metrics'][key] for r in group if r['metrics'][key] is not None]
            stats[key] = dict(mean=float(np.mean(values)) if values else None,
                std=float(np.std(values)) if values else None, n=len(values))
        summary.append(dict(cohort=cohort, arm=arm, structures=len(group), metrics=stats))
    # Pair target means, so three samples per complex don't masquerade as
    # independent proteins. Intervals describe this small fixed target set.
    paired = []
    rng = np.random.default_rng(82501)
    for cohort in ['complex', 'cameo']:
        baseline = grouped[cohort, 'original']
        if not baseline:
            continue
        for arm in ['six_only', 'mixed']:
            changed = grouped[cohort, arm]
            if not changed:
                continue
            targets = sorted({r['id'] for r in baseline})
            for key in baseline[0]['metrics']:
                differences = []
                for target in targets:
                    b = [r['metrics'][key] for r in baseline if r['id'] == target and r['metrics'][key] is not None]
                    a = [r['metrics'][key] for r in changed if r['id'] == target and r['metrics'][key] is not None]
                    if b and a:
                        differences.append(float(np.mean(a)-np.mean(b)))
                if differences:
                    boot = np.mean(rng.choice(differences, size=(10000, len(differences))), axis=1)
                    paired.append(dict(cohort=cohort, arm=arm, metric=key, targets=len(differences),
                        mean_change=float(np.mean(differences)),
                        bootstrap_95_CI=[float(v) for v in np.quantile(boot, [.025, .975])]))
    return dict(groups=summary, paired_differences=paired,
        scope='Coordinate validation of existing samples; complex agreement uses synthetic training targets; CAMEO geometry covers full predicted sequences.',
        aggregation='Arithmetic mean across structures, not pooled atom counts. Paired confidence intervals resample target means.',
        reference_caveat='Experimental references can have unresolved residues. Refined targets were relaxed; generated samples were not.')


@hydra.main(version_base='1.3', config_path='../configs', config_name='training_geometry')
def main(cfg):
    if cfg.backend == 'molprobity':
        assert os.environ.get('SLURM_JOB_ID'), 'Use CPU Slurm for remote MolProbity validation'
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved-config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (output / 'git-commit.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True))
    rows = inputs(cfg)
    if cfg.max_structures is not None:
        rows = rows[:cfg.max_structures]
    save(output / 'input-manifest.json', rows)
    method_hash = hashlib.sha256((digest(__file__) + digest(ROOT/'scripts/structure_clashes.py') +
        (digest(ROOT/'scripts/molprobity_geometry.py') if cfg.backend == 'molprobity' else '') +
        cfg.backend + cfg.cctbx_version).encode()).hexdigest()
    tasks = [(row, cfg.backend, cfg.cache_dir, method_hash) for row in rows]
    results = []
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        for row, result in zip(rows, pool.map(analyze_one, tasks)):
            record = dict(**row, metrics=metrics(result), diagnostics=result)
            results.append(record)
            save(output / 'geometry-results.json', results)
            print(f'Validated {len(results)}/{len(rows)}: {row["cohort"]}/{row["arm"]}/{row["id"]}', flush=True)
    save(output / 'geometry-summary.json', summarize(results))
    flattened = [dict(cohort=r['cohort'], arm=r['arm'], id=r['id'], sample=r['sample'], **r['metrics']) for r in results]
    with (output / 'geometry-results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=flattened[0])
        writer.writeheader(); writer.writerows(flattened)
    save(output / 'completion.json', dict(state='completed', structures=len(results), backend=cfg.backend, method_hash=method_hash))


if __name__ == '__main__':
    main()
