"""Uniform metrics reports for the automated training check."""
import csv
import html
import statistics

from scripts.build_training_geometry_report import LABELS, write_report


INTERFACES = {'mean_DockQ': 'Mean receptor DockQ ↑', 'weakest_DockQ': 'Mean weakest-receptor DockQ ↑',
              'fnat': 'Native-contact recovery ↑', 'iRMSD': 'Interface RMSD (Å) ↓',
              'LRMSD': 'Docking displacement / LRMSD (Å) ↓'}
FOLDS = {'tm_score': 'TM-score ↑', 'oligo_gdtts': 'GDT-TS ↑', 'lddt': 'All-atom lDDT ↑',
         'bb_lddt': 'Cα lDDT ↑', 'rmsd': 'RMSD (Å) ↓'}


def interface_means(records):
    arms = {}
    for condition in ['pretrained', 'fine_tuned']:
        rows = [r for r in records if r['condition'] == condition]
        values = {}
        for key in INTERFACES:
            xs = [r[key] if key in r else statistics.mean(i[key] for i in r['interfaces']) for r in rows]
            values[key] = dict(mean=statistics.mean(xs), std=statistics.pstdev(xs), n=len(xs))
        arms[condition] = values
    return arms


def build_report(check, interfaces, geometry):
    from scripts.check_training_run import load, save, token
    out = check.output
    summary = load(out / 'geometry-summary.json')
    folds = load(check.run / 'retention/benchmark-summary.json')
    groups = {(g['cohort'], g['arm']): g for g in summary['groups']}
    label = str(check.cfg.label)
    scores = interface_means(interfaces)
    # Historical general-data comparisons require identical references, seeds AND baseline predictions.
    refs = {r['id']: r['sha256'] for r in geometry if r['arm'] == 'experimental_reference'}
    baseline = load(check.run / 'retention/prediction-receipt.json')
    retention_identity = token(sorted((r['id'], r['sample'], r['seed'], r['sha256'], refs[r['id']])
                                      for r in baseline if r['condition'] == 'pretrained'))
    result = dict(run_id=str(check.cfg.run_id), label=label, interfaces=scores,
                  general_fold=folds, geometry=summary, retention_identity=retention_identity,
                  reference_scope='Synthetic training-target reconstruction; CAMEO retention uses experimental references')
    save(out / 'summary.json', result)
    compact = out / 'geometry-input'
    save(compact / 'geometry-results.json', geometry)
    save(compact / 'geometry-summary.json', summary)
    with (compact / 'geometry-results.csv').open('w') as stream:
        rows = [dict(cohort=r['cohort'], arm=r['arm'], id=r['id'], sample=r['sample'], **r['metrics']) for r in geometry]
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    description = (f'Training run {check.cfg.run_id}: original released checkpoint versus {label}. '
        'Complex comparisons use the same input sequences and synthetic training targets in both arms. '
        'General-protein retention uses matched experimental targets and seeds. '
        'No predictions were relaxed or changed by this check. See the parent report for training exposure and fold metrics.')
    write_report(compact, out / 'geometry', description, label)
    flat, sections = [], []

    def section(title, labels, columns):
        rows = []
        for key, name in labels.items():
            values = []
            for column, metrics in columns.items():
                stat = metrics.get(key, {})
                value = stat.get('mean')
                values.append('—' if value is None else f'{value:.4f}')
                flat.append(dict(section=title, arm=column, metric=key, mean=value,
                                 std=stat.get('std'), n=stat.get('n')))
            rows.append([name, *values])
        headers = ['Metric', *columns]
        sections.append((title, headers, rows))

    section('Complex interfaces', INTERFACES, {'Original': scores['pretrained'], label: scores['fine_tuned']})
    for cohort, title, reference in [('complex', 'Complex geometry', 'refined_target'),
                                      ('cameo', 'General-protein geometry', 'experimental_reference')]:
        section(title, LABELS, {'Original': groups[cohort, 'original']['metrics'],
                label: groups[cohort, 'mixed']['metrics'],
                'Synthetic targets' if cohort == 'complex' else 'Experimental references': groups[cohort, reference]['metrics']})
    fold_arms = {'Original': folds['means_and_medians']['pretrained'], label: folds['means_and_medians']['fine_tuned']}
    history = []
    for path in sorted((check.artifacts / 'analysis/training-checks').glob('*/summary.json')):
        other = load(path)
        if path.parent == out or other['retention_identity'] != retention_identity:
            continue
        name = f"{other['label']} · {other['run_id']}"
        fold_arms[name] = other['general_fold']['means_and_medians']['fine_tuned']
        history.append(other['run_id'])
    section('General-protein fold retention', FOLDS, fold_arms)
    training = load(check.run / 'training-summary.json')
    intro = (f'Run {check.cfg.run_id}: {label}. {training.get("global_step", "Unknown")} updates; '
             f'source counts: {training.get("source_counts", {})}. '
             f'Complex predictions: {sum(r["cohort"] == "complex" and r["arm"] in ("original", "mixed") for r in geometry)}; '
             f'CAMEO predictions: {sum(r["cohort"] == "cameo" and r["arm"] in ("original", "mixed") for r in geometry)}. '
             'Means are across structures, not pooled atoms. Complex agreement measures reconstruction of synthetic training targets, '
             'not independent experimental accuracy. Cropped and full-length complex scores are not directly interchangeable.')
    explanations = (
        'DockQ uses one coherent type-compatible assignment for all four receptors, each against the whole ligand dimer. '
        'Native-contact recovery is DockQ fnat (5 Å), distinct from the viewer’s 4 Å statistic. '
        'Weakest-receptor DockQ averages the minimum receptor score within each prediction. '
        'Heavy-atom overlaps use >0.4 Å vdW overlap and covalent exclusions; they are not MolProbity clashscore. '
        'MolProbity temporarily rebuilds hydrogens without moving heavy atoms or flipping side chains. '
        'Bond/angle outliers use the existing 4σ native restraints; reference structures may have unresolved residues. '
        'Paired intervals resample target means rather than individual samples. '
        'Fewer clashes alone do not establish better covalent geometry. '
        'Historical CAMEO columns are included only when reference hashes, seeds and original predictions match exactly.')
    markdown = ['# Automated training evaluation', '', intro, '', explanations, '']
    tables = []
    for title, headers, rows in sections:
        markdown += ['## ' + title, '', '| ' + ' | '.join(headers) + ' |',
                     '|' + '|'.join(['---'] * len(headers)) + '|']
        markdown += ['| ' + ' | '.join(row) + ' |' for row in rows]
        markdown.append('')
        heads = ''.join('<th>' + html.escape(h) + '</th>' for h in headers)
        body = ''.join('<tr>' + ''.join('<td>' + html.escape(v) + '</td>' for v in row) + '</tr>' for row in rows)
        tables.append(f'<h2>{html.escape(title)}</h2><div class="scroll"><table><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div>')
    markdown += ['[Per-target geometry](geometry/index.html) · [CSV](metrics.csv) · [JSON](summary.json)',
                 'Paired geometry confidence intervals: `geometry-summary.json`. '
                 'Paired fold confidence intervals: `summary.json` → `general_fold`.', '']
    (out / 'report.md').write_text('\n'.join(markdown))
    with (out / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['section', 'arm', 'metric', 'mean', 'std', 'n'])
        writer.writeheader()
        writer.writerows(flat)
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training evaluation</title><style>body{font:15px/1.5 system-ui;color:#233344;background:#f5f7fa;margin:0}main{max-width:1450px;margin:auto;padding:24px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:7px 12px;border-bottom:1px solid #dce3eb;white-space:nowrap;font-variant-numeric:tabular-nums}a{color:#244c73}.note{padding:12px;background:#eaf0f6;border-radius:7px}</style>
<main><h1>Training evaluation</h1><p class="note">__INTRO__</p><p><a href="geometry/index.html">Inspect individual targets and samples</a> · <a href="report.md">Markdown</a> · <a href="metrics.csv">CSV</a> · <a href="summary.json">JSON</a></p>__TABLES__<h2>Definitions</h2><p>__DEFINITIONS__</p></main>'''
    (out / 'index.html').write_text(page.replace('__INTRO__', html.escape(intro)).replace('__TABLES__', ''.join(tables))
                                  .replace('__DEFINITIONS__', html.escape(explanations)))
    check.record(compared_cameo_runs=history)
    return {condition: {k: v['mean'] for k, v in values.items()} for condition, values in scores.items()}
