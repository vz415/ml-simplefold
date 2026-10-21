#!/usr/bin/env python3
"""Collect and evaluate an existing training run; never sample or train locally."""
from collections import Counter
import hashlib
import json
import math
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_training_geometry import metrics, summarize
from scripts.structure_clashes import analyze_clashes

GEOMETRY_FILES = ('completion.json', 'geometry-results.json', 'geometry-summary.json',
                  'geometry-results.csv', 'input-manifest.json', 'resolved-config.yaml',
                  'git-commit.txt')
ALLOWED = {'.pdb', '.cif', '.json', '.jsonl', '.yaml', '.yml', '.csv', '.txt', '.log'}
EXCLUDED = {'conditioning', 'trajectories', 'checkpoints'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def token(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(rows):
    """Paths and row order may change during staging; scientific identity may not."""
    return token(sorted([{
        k: row[k] for k in ('cohort', 'arm', 'id', 'sample', 'sha256', 'groups')
    } for row in rows], key=lambda r: (r['cohort'], r['arm'], r['id'], r['sample'])))


def native_method(version):
    return hashlib.sha256((''.join(digest(ROOT / 'scripts' / name) for name in
        ('run_training_geometry.py', 'structure_clashes.py', 'molprobity_geometry.py'))
        + 'molprobity' + version).encode()).hexdigest()


def checked_file(root, relative, expected):
    path = Path(relative)
    assert not path.is_absolute() and '..' not in path.parts, relative
    path = root / path
    assert digest(path) == expected, f'Checksum mismatch: {path}'
    return path


def balanced(rows):
    assert {r['condition'] for r in rows} == {'pretrained', 'fine_tuned'}
    original = Counter((r['id'], r['sample'], r['seed']) for r in rows if r['condition'] == 'pretrained')
    tuned = Counter((r['id'], r['sample'], r['seed']) for r in rows if r['condition'] == 'fine_tuned')
    assert original and original == tuned, 'Missing or unpaired prediction/seed'
    assert all(n == 1 for n in original.values()), 'Duplicate prediction identity'


def verify_fold_scores(base):
    """A copied score table must still describe these exact paired predictions."""
    import numpy as np
    predictions = load(base / 'prediction-receipt.json')
    scored = load(base / 'benchmark-results.json')
    by_key = lambda rows: {(r['condition'], r['id'], r['sample']): r for r in rows}
    expected, actual = by_key(predictions), by_key(scored)
    assert len(actual) == len(scored) and actual.keys() == expected.keys()
    for key, row in actual.items():
        for field in ('seed', 'sha256', 'checkpoint_sha256'):
            assert row[field] == expected[key][field], (key, field)
    summary = load(base / 'benchmark-summary.json')
    for arm, stats in summary['means_and_medians'].items():
        rows = [r for r in scored if r['condition'] == arm]
        for metric, stat in stats.items():
            values = [r[metric] for r in rows]
            assert all(math.isfinite(v) for v in values)
            assert abs(stat['mean'] - float(np.mean(values))) < 1e-12
            assert abs(stat['median'] - float(np.median(values))) < 1e-12


class Check:
    def __init__(self, cfg):
        self.cfg = cfg
        self.artifacts = Path(cfg.artifact_dir).expanduser().resolve()
        self.run = self.artifacts / 'remote-runs' / str(cfg.run_id)
        self.output = Path(cfg.output_dir).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.receipt = self.output / 'receipt.json'
        self.state = load(self.receipt) if self.receipt.exists() else {}
        self.connected = False

    def record(self, **values):
        self.state.update(values)
        save(self.receipt, self.state)

    def remote(self, command):
        assert not self.cfg.offline, 'Offline mode cannot contact HPC'
        if not self.connected:
            subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                            self.cfg.hpc.probe_host, 'hostname; whoami; pwd'], check=True)
            self.connected = True
        return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                        self.cfg.hpc.host, command], text=True, timeout=90)

    def transfer(self, remote, destination, files=None):
        if not self.connected:
            self.remote('hostname; whoami; pwd')
        destination.mkdir(parents=True, exist_ok=True)
        cmd = ['rsync', '-rtz', '-e', shlex.join(['ssh', '-o', 'BatchMode=yes', '-o',
               'ConnectTimeout=10', '-o', 'HostName=' + self.cfg.hpc.transfer_host])]
        if files is not None:
            listing = self.output / 'transfer-files.txt'
            listing.write_text(''.join(f'{name}\n' for name in files))
            cmd += ['--files-from=' + str(listing)]
        subprocess.run(cmd + [f'{self.cfg.hpc.host}:{remote}/', str(destination) + '/'], check=True)

    def collect(self):
        if self.cfg.offline:
            assert (self.run / 'completion.json').exists(), 'Download the completed run before offline evaluation'
            return
        if not self.cfg.refresh_downloads and self.local_ready():
            self.record(downloaded_files=0, local_predictions_verified=True)
            print('Verified existing predictions; skipping remote collection.', flush=True)
            return
        remote = f'{self.cfg.hpc.artifact_dir}/runs/{self.cfg.run_id}'
        # File enumeration and checksums only; scientific computation uses Slurm.
        script = '''import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
assert json.loads((root/'completion.json').read_text())['state']=='completed'
for parent,dirs,files in os.walk(root):
 dirs[:]=[d for d in dirs if d not in {'conditioning','trajectories','checkpoints'}]
 for name in files:
  p=Path(parent)/name
  if p.suffix.lower() not in {'.pdb','.cif','.json','.jsonl','.yaml','.yml','.csv','.txt','.log'}: continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(8388608),b''): h.update(chunk)
  rows.append({'relative':str(p.relative_to(root)),'sha256':h.hexdigest()})
print(json.dumps(rows))'''
        manifest = json.loads(self.remote(shlex.join(['python3', '-c', script, remote])))
        save(self.output / 'remote-files.json', manifest)
        missing = []
        for row in manifest:
            p = Path(row['relative'])
            assert not p.is_absolute() and '..' not in p.parts
            assert p.suffix.lower() in ALLOWED and not EXCLUDED.intersection(p.parts)
            local = self.run / p
            if not local.exists() or digest(local) != row['sha256']:
                missing.append(row['relative'])
        if missing:
            self.transfer(remote, self.run, missing)
        for row in manifest:
            checked_file(self.run, row['relative'], row['sha256'])
        self.record(downloaded_files=len(missing), verified_files=len(manifest))

    def local_ready(self):
        """Use completed local outputs; a missing/changed structure triggers collection."""
        try:
            for base in [self.run, self.run / 'retention']:
                assert load(base / 'completion.json')['state'] == 'completed'
                for item in load(base / 'prediction-receipt.json'):
                    checked_file(base, item['file'], item['sha256'])
                    if 'cif_file' in item:
                        checked_file(base, item['cif_file'], item['cif_sha256'])
            for item in load(self.run / 'retention/dataset-receipt.json')['prepared_records']:
                checked_file(self.run / 'retention', 'references/' + item['id'] + '.pdb', item['reference_sha256'])
            required = ['experiment.json', 'training-summary.json', 'resolved-config.yaml',
                        'retention/benchmark-results.json', 'retention/benchmark-summary.json']
            assert all((self.run / path).exists() for path in required)
            assert self.cfg.selection or (self.run / 'synthetic-selection.json').exists()
        except (FileNotFoundError, AssertionError, KeyError, json.JSONDecodeError):
            return False
        return True

    def inputs(self):
        selection_path = Path(self.cfg.selection) if self.cfg.selection else self.run / 'synthetic-selection.json'
        selection = load(selection_path)
        targets = {r['id']: r for r in selection['records']}
        source = self.output / 'targets'
        source.mkdir(exist_ok=True)
        for row in targets.values():
            path = source / row['file']
            assert Path(row['file']).name == row['file']
            if not path.exists():
                local = Path(row.get('source_local', ''))
                if local.is_file():
                    path.write_bytes(local.read_bytes())
                else:
                    assert not self.cfg.offline, f'Missing synthetic target {row["id"]}'
                    remote = load(self.run / 'experiment.json')['synthetic_dataset'] + '/source'
                    self.transfer(remote, source, [row['file']])
            assert digest(path) == row['sha256'], path
        save(source / 'selection.json', selection)
        rows = []
        validation_path = self.output / 'input-checks.json'
        validated = set(load(validation_path)) if validation_path.exists() else set()
        import gemmi
        for cohort, base in [('complex', self.run), ('cameo', self.run / 'retention')]:
            predictions = load(base / 'prediction-receipt.json')
            balanced(predictions)
            for item in predictions:
                path = checked_file(base, item['file'], item['sha256'])
                if 'cif_file' in item:
                    checked_file(base, item['cif_file'], item['cif_sha256'])
                if cohort == 'complex':
                    target = targets[item['id']]
                    key = token([item['sha256'], target['sha256'], target['groups']])
                    if key not in validated:
                        model = gemmi.read_structure(str(path))[0]
                        reference = gemmi.read_structure(str(source / target['file']))[0]
                        assert [c.name for c in model] == list('ABCDEF')
                        for chain in model:
                            assert [r.name for r in chain] == [r.name for r in reference[chain.name]]
                            assert all(math.isfinite(v) for r in chain for a in r for v in (a.pos.x, a.pos.y, a.pos.z))
                        validated.add(key)
                rows.append(dict(cohort=cohort, arm='original' if item['condition'] == 'pretrained' else 'mixed',
                    id=item['id'], sample=item['sample'], path=str(path), sha256=item['sha256'],
                    groups=targets[item['id']]['groups'] if cohort == 'complex' else {}))
        assert set(r['id'] for r in rows if r['cohort'] == 'complex') == set(targets)
        for row in targets.values():
            rows.append(dict(cohort='complex', arm='refined_target', id=row['id'], sample=0,
                path=str(source / row['file']), sha256=row['sha256'], groups=row['groups']))
        for item in load(self.run / 'retention/dataset-receipt.json')['prepared_records']:
            path = checked_file(self.run / 'retention', 'references/' + item['id'] + '.pdb', item['reference_sha256'])
            rows.append(dict(cohort='cameo', arm='experimental_reference', id=item['id'], sample=0,
                path=str(path), sha256=item['reference_sha256'], groups={}))
        verify_fold_scores(self.run / 'retention')
        save(validation_path, sorted(validated))
        assert len({(r['cohort'], r['arm'], r['id'], r['sample']) for r in rows}) == len(rows)
        assert {r['id'] for r in rows if r['cohort'] == 'cameo' and r['arm'] == 'original'} == {
            r['id'] for r in rows if r['cohort'] == 'cameo' and r['arm'] == 'experimental_reference'}
        save(self.output / 'input-manifest.json', rows)
        request = token([identity(rows), native_method(self.cfg.geometry.cctbx_version)])
        if self.state.get('geometry_request') not in (None, request):
            self.state.pop('geometry_job', None)
        self.record(input_identity=identity(rows), input_structures=len(rows), geometry_request=request)
        return selection, rows

    def docking(self, selection):
        sys.path.insert(0, str(Path(self.cfg.dockq_path)))
        from scripts.benchmark_multimer_learning import score
        from importlib.metadata import version, PackageNotFoundError
        try:
            dockq_version = version('DockQ')
        except PackageNotFoundError:
            import DockQ
            dockq_version = digest(Path(DockQ.__path__[0]) / 'DockQ.py')
        method = token([digest(ROOT / 'scripts/benchmark_multimer_learning.py'),
                        digest(ROOT / 'scripts/structure_clashes.py'), dockq_version])
        targets = {r['id']: r for r in selection['records']}
        cache = self.artifacts / 'analysis-cache/training-check-dockq'
        records = []
        computed = 0
        for item in load(self.run / 'prediction-receipt.json'):
            target = targets[item['id']]
            key = token([method, item['sha256'], target['sha256'], target['groups']])
            path = cache / (key + '.json')
            if path.exists():
                result = load(path)
            else:
                result = score(self.run / item['file'], self.output / 'targets' / target['file'], target['groups'])
                save(path, result)
                computed += 1
            records.append(dict(**item, **result, reference_sha256=target['sha256']))
            print(f'DockQ {len(records)}: {item["condition"]}/{item["id"]}', flush=True)
        save(self.output / 'interface-results.json', records)
        self.record(dockq_method=method, dockq_computed=computed, dockq_cached=len(records)-computed)
        return records

    def find_geometry(self, rows):
        expected = identity(rows)
        method = native_method(self.cfg.geometry.cctbx_version)
        candidates = [str(self.cfg.geometry.job_id)] if self.cfg.geometry.job_id else [
            p.parent.name for p in (self.artifacts / 'remote-runs').glob('*/completion.json')]
        for job in candidates:
            base = self.artifacts / 'remote-runs' / job
            if not (base / 'completion.json').exists() or not (base / 'input-manifest.json').exists():
                continue
            completion = load(base / 'completion.json')
            if (completion.get('state') == 'completed' and completion.get('backend') == 'molprobity'
                and completion.get('method_hash') == method and identity(load(base / 'input-manifest.json')) == expected):
                self.record(geometry_job=job, geometry_reused=True)
                return base
        return None

    def submit_geometry(self, rows):
        import yaml
        policy = yaml.safe_load((ROOT / '.agents/hpc-experiments.yaml').read_text())
        assert Path(self.cfg.hpc.script).name in policy['allowed_scripts']
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(), 'Commit and push source before submitting'
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
        assert branch == self.cfg.hpc.branch
        checkout = shlex.quote(self.cfg.hpc.checkout)
        location = self.remote(f'hostname; whoami; cd {checkout} && pwd && git branch --show-current').splitlines()
        assert location[1] == self.cfg.hpc.user and location[2] == self.cfg.hpc.checkout
        assert location[3] == branch
        sync = f'cd {checkout} && test -z "$(git status --porcelain)" && '
        sync += f'test "$(git branch --show-current)" = {shlex.quote(branch)} && '
        sync += shlex.join(['git', 'fetch', 'experiments', branch]) + ' && '
        sync += shlex.join(['git', 'merge', '--ff-only', 'FETCH_HEAD'])
        self.remote(sync)
        actual = self.remote(f'cd {checkout} && hostname && whoami && pwd && git rev-parse HEAD')
        assert actual.splitlines()[-1] == commit
        input_dir = f'{self.cfg.geometry.staging_dir}/inputs-{identity(rows)}'
        remote_rows = [dict(r, path=input_dir + '/' + r['sha256'] + '.pdb') for r in rows]
        manifest = self.output / 'remote-input-manifest.json'
        save(manifest, remote_rows)
        archive = self.output / 'geometry-inputs.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            seen = set()
            for row in rows:
                if row['sha256'] not in seen:
                    tar.add(row['path'], arcname=row['sha256'] + '.pdb')
                    seen.add(row['sha256'])
            tar.add(manifest, arcname='input-manifest.json')
        self.remote(shlex.join(['mkdir', '-p', input_dir]))
        subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'HostName=' + self.cfg.hpc.transfer_host,
                        str(archive), f'{self.cfg.hpc.host}:{input_dir}/inputs.tar.gz'], check=True)
        remote_hash = self.remote(shlex.join(['sha256sum', input_dir + '/inputs.tar.gz'])).split()[0]
        assert remote_hash == digest(archive)
        self.remote(shlex.join(['tar', '-xzf', input_dir + '/inputs.tar.gz', '-C', input_dir]))
        overrides = [f'workers={self.cfg.geometry.cpus}', f'input_manifest={input_dir}/input-manifest.json',
                     f'staging_root={self.cfg.geometry.staging_dir}', f'cache_dir={self.cfg.geometry.cache_dir}',
                     f'cctbx_version={self.cfg.geometry.cctbx_version}']
        command = ['sbatch', '--parsable', '--job-name=training-check-' + str(self.cfg.run_id),
                   '--partition=' + self.cfg.hpc.partition, '--account=' + self.cfg.hpc.account,
                   '--qos=' + self.cfg.hpc.qos, '--gres=none', f'--cpus-per-task={self.cfg.geometry.cpus}',
                   '--mem=' + self.cfg.geometry.memory, '--time=' + self.cfg.geometry.time,
                   self.cfg.hpc.script, '--validate-geometry', *overrides]
        job = self.remote(f'cd {checkout} && ' + shlex.join(command)).strip().split(';')[0]
        assert job.isdigit(), job
        self.record(geometry_job=job, geometry_reused=False, geometry_command=command,
                    project=policy['project_name'], remote_checkout=self.cfg.hpc.checkout,
                    remote_source_commit=commit, geometry_overrides=overrides,
                    geometry_python_command=[
                        '/data/homezvol2/ynkim4/.conda/envs/simplefold-geometry/bin/python',
                        'scripts/run_training_geometry.py', *overrides])
        self.note(f'geometry-{job}', 'Submitted CPU geometry validation', self.state)
        return job

    def collect_geometry(self, job):
        remote = f'{self.cfg.geometry.staging_dir}/{job}'
        # An explicit existing job may have used the durable destination instead.
        if self.cfg.geometry.job_id:
            remote = f'{self.cfg.hpc.artifact_dir}/runs/{job}'
        started = time.monotonic()
        while True:
            response = self.remote(f'if test -f {shlex.quote(remote + "/completion.json")}; then '
                f'cat {shlex.quote(remote + "/completion.json")}; else '
                + shlex.join(['sacct', '-X', '-n', '-j', job, '--format=State', '--parsable2']) + '; fi').strip()
            if response.startswith('{'):
                assert json.loads(response)['state'] == 'completed'
                break
            if any(s in response for s in ['FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL']):
                raise RuntimeError(f'Geometry job {job}: {response}; not resubmitted automatically')
            if time.monotonic() - started >= self.cfg.geometry.wait_seconds:
                self.record(status='waiting_for_geometry')
                print(f'Geometry job {job} still {response}; rerun the same command to resume.', flush=True)
                return None
            print(f'Geometry job {job}: {response or "starting"}', flush=True)
            time.sleep(min(30, self.cfg.geometry.poll_seconds))
        checksums = self.remote('cd ' + shlex.quote(remote) + ' && ' + shlex.join(['sha256sum', *GEOMETRY_FILES]))
        base = self.artifacts / 'remote-runs' / job
        wanted = []
        manifest = []
        for line in checksums.splitlines():
            sha, relative = line.split(None, 1)
            relative = relative.strip()
            manifest.append(dict(relative=relative, sha256=sha))
            if not (base / relative).exists() or digest(base / relative) != sha:
                wanted.append(relative)
        if wanted:
            self.transfer(remote, base, wanted)
        for row in manifest:
            checked_file(base, row['relative'], row['sha256'])
        save(base / 'training-check-transfer.json', dict(source=remote, files=manifest))
        return base

    def geometry(self, rows):
        base = self.find_geometry(rows)
        if base is None:
            assert not self.cfg.offline, 'No matching native geometry results; run online to submit CPU validation'
            job = str(self.cfg.geometry.job_id or self.state.get('geometry_job') or self.submit_geometry(rows))
            base = self.collect_geometry(job)
            if base is None:
                return None
        completion = load(base / 'completion.json')
        assert completion['backend'] == 'molprobity' and completion['state'] == 'completed'
        assert completion['method_hash'] == native_method(self.cfg.geometry.cctbx_version)
        assert identity(load(base / 'input-manifest.json')) == identity(rows)
        expected = token([identity(rows), completion['method_hash'], digest(base / 'geometry-results.json')])
        audit_path = self.output / 'geometry-audit.json'
        compact_path = self.output / 'geometry-metrics.json'
        summary_path = self.output / 'geometry-summary.json'
        if audit_path.exists() and compact_path.exists() and summary_path.exists():
            audit = load(audit_path)
            if (audit.get('fingerprint') == expected and audit.get('metrics_sha256') == digest(compact_path)
                and audit.get('summary_sha256') == digest(summary_path)):
                return load(compact_path)
        measured = load(base / 'geometry-results.json')
        assert identity(measured) == identity(rows) and len(measured) == len(rows)
        compact = []
        local = {(r['cohort'], r['arm'], r['id'], r['sample']): r for r in rows}
        max_move = 0
        for row in measured:
            input_row = local[row['cohort'], row['arm'], row['id'], row['sample']]
            coordinate = metrics(dict(coordinate=analyze_clashes(input_row['path'], input_row['groups'])))
            for key, value in coordinate.items():
                assert row['metrics'][key] == value or (value is not None and abs(row['metrics'][key] - value) < 1e-10)
            m = row['diagnostics']['molprobity']
            assert m['input_sha256'] == row['sha256']
            assert m['settings']['covalent_topology']['incidental_automatic_crosslinks'] is False
            assert not m['settings']['do_sidechain_flips']
            assert 'involved in Misc. bond' not in m['tool_log']
            max_move = max(max_move, m['settings']['heavy_atom_max_displacement_during_hydrogenation'])
            norm = m['clashscore_normalization']
            assert abs(m['clashscore'] - 1000 * norm['clashes_scored'] / norm['atoms_scored']) < 1e-9
            assert all(v is None or math.isfinite(v) for v in row['metrics'].values())
            compact.append({**input_row, 'metrics': row['metrics']})
        assert max_move < 1e-6
        save(compact_path, compact)
        # Summaries and paired intervals are recomputed from the verified rows.
        save(summary_path, summarize(compact))
        save(audit_path, dict(fingerprint=expected, structures=len(rows), coordinate_metrics_match=True,
             heavy_atom_max_displacement=max_move, native_normalizations_match=True,
             incidental_crosslinks=False, method_hash=completion['method_hash'], metrics_sha256=digest(compact_path),
             summary_sha256=digest(summary_path)))
        return compact

    def note(self, suffix, heading, content):
        if not self.cfg.notes.enabled:
            return
        from datetime import datetime
        date = datetime.now().astimezone().date().isoformat()
        folder = Path(self.cfg.notes.directory).expanduser() / 'experiments/hpc_sample' / date
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'{self.cfg.run_id} training-check {suffix}.md'
        marker = '<!-- training-check:' + token(content) + ' -->'
        if path.exists() and marker in path.read_text():
            return
        with path.open('a') as stream:
            stream.write(f'\n## {heading}\n\n{marker}\n\n```json\n{json.dumps(content, indent=2)}\n```\n')

    def execute(self):
        self.state.pop('error', None)
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
        self.record(status='collecting', run_id=str(self.cfg.run_id), local_checkout=str(ROOT),
                    workflow_commit=git('rev-parse', 'HEAD'), source_branch=git('branch', '--show-current'),
                    local_git_state=git('status', '--porcelain') or 'clean')
        self.collect()
        selection, rows = self.inputs()
        self.record(status='scoring_interfaces')
        interfaces = self.docking(selection)
        self.record(status='checking_geometry')
        geometry = self.geometry(rows)
        if geometry is None:
            return
        from scripts.training_check_report import build_report
        summary = build_report(self, interfaces, geometry)
        self.record(status='completed', report=str(self.output / 'index.html'))
        self.note('complete', 'Training evaluation complete', dict(run_id=str(self.cfg.run_id),
                  geometry_job=self.state['geometry_job'], report=self.state['report'],
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  results=summary))
        print(f'Complete: {self.output / "index.html"}', flush=True)


@hydra.main(version_base='1.3', config_path='../configs', config_name='training_check')
def main(cfg):
    check = Check(cfg)
    with (check.output / '.lock').open('a') as lock:
        # Two terminals/agents must not submit the same validation simultaneously.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (check.output / 'resolved-config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
        try:
            check.execute()
        except Exception as exc:
            check.record(status='failed', error=str(exc))
            raise


if __name__ == '__main__':
    main()
