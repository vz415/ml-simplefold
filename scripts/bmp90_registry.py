#!/usr/bin/env python3
"""Local-only BMP90 inventory, provenance and readiness checks (stdlib)."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "artifacts/bmp90"
DEFAULT_PLAN = DEFAULT_ROOT / "BMP_90_target_plan.json"
STATUSES = ("prepared_blocked", "ready_for_submission", "superseded", "submitted", "running",
            "completed", "failed", "cancelled")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def immutable_write(path, data):
    """Exclusive creation: never truncate an earlier artifact, even on retry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_plan(path):
    data = Path(path).read_bytes()
    plan = json.loads(data)
    jobs = plan["jobs"]
    ids = [job["target_id"] for job in jobs]
    if len(ids) != 90 or len(ids) != len(set(ids)):
        raise ValueError("Expected 90 distinct plan targets")
    return plan, {"path": str(Path(path).resolve()), "sha256": sha(data)}


def csv_bytes(rows, fields):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


def inventory(plan_path, root):
    plan, identity = load_plan(plan_path)
    root = Path(root)
    paths = [root / "target_inventory.json", root / "target_inventory.csv"]
    if any(p.exists() for p in paths):
        raise FileExistsError("Inventory already exists; preserve the plan snapshot")
    data = {"created_at": now(), "plan": identity, "stage": plan["stage"],
            "panel_status": plan["panel_status"], "targets": plan["jobs"]}
    fields = ["target_id", "ligand", "type_I", "type_II", "wave", "status",
              "input_status", "paper_review_status", "rule_activation_status"]
    immutable_write(paths[0], encode(data))
    immutable_write(paths[1], csv_bytes(plan["jobs"], fields))
    return {"target_count": len(plan["jobs"]), "files": [str(p) for p in paths]}


def positive(value):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def safe_name(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)


def fasta(text):
    result, key = {}, None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:].split()
            if not header or not header[0].split("|")[0]:
                raise ValueError("FASTA chain ID is missing")
            key = header[0].split("|")[0]
            if key in result:
                raise ValueError("Duplicate FASTA chain: " + key)
            result[key] = ""
        elif key is None:
            raise ValueError("FASTA sequence precedes its header")
        else:
            result[key] += line.upper()
    if not result or any(not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", v)
                         for v in result.values()):
        raise ValueError("FASTA must contain nonempty, unambiguous protein sequences")
    return result


def preflight(spec_path, plan_path=DEFAULT_PLAN):
    """Check artifacts on this host; remote paths must be checked on that host."""
    spec_path = Path(spec_path).resolve()
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    plan, plan_id = load_plan(plan_path)
    issues, artifacts, snapshots = [], {}, {}
    targets = {j["target_id"]: j for j in plan["jobs"]}
    target = targets.get(spec.get("target_id"))
    if target is None:
        issues.append("target_id is absent from the 90-target plan")
    if spec.get("condition") not in plan["protocol"]["paired_conditions"]:
        issues.append("condition must name one of the plan's paired conditions")
    if not safe_name(spec.get("scoring_version")):
        issues.append("scoring_version is missing or unsafe")
    for key in ("seed", "replicate"):
        if not integer(spec.get(key)):
            issues.append(key + " must be an explicit nonnegative integer")

    def artifact(key, keep=False, value=None):
        value = spec.get(key) if value is None else value
        if not isinstance(value, str) or not value:
            issues.append(key + " path is missing")
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = spec_path.parent / path
        path = path.resolve()
        try:
            digest, size, chunks = hashlib.sha256(), 0, []
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    if keep:
                        chunks.append(chunk)
            if size == 0:
                raise ValueError("empty artifact")
            artifacts[key] = {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}
            if keep:
                snapshots[key] = b"".join(chunks).decode("utf-8")
                return snapshots[key]
        except (OSError, ValueError, UnicodeError) as error:
            issues.append(key + ": " + str(error))
        return None

    sequences_text = artifact("sequences_fasta", True)
    mappings_text = artifact("residue_mapping_json", True)
    scoring_text = artifact("scoring_json", True)
    artifact("runner")
    artifact("checkpoint")
    expected_checkpoint = spec.get("checkpoint_expected_sha256")
    if expected_checkpoint is not None:
        if not isinstance(expected_checkpoint, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_checkpoint):
            issues.append("checkpoint_expected_sha256 must be an observed lowercase SHA256")
        elif "checkpoint" in artifacts and artifacts["checkpoint"]["sha256"] != expected_checkpoint:
            issues.append("checkpoint SHA256 differs from checkpoint_expected_sha256")
    if spec.get("hydra_experiment") or spec.get("resolved_config"):
        artifact("resolved_config", True)
    auxiliary = spec.get("auxiliary_inputs", {})
    if not isinstance(auxiliary, dict):
        issues.append("auxiliary_inputs must map labels to artifact paths")
    else:
        for label, value in auxiliary.items():
            artifact("auxiliary_inputs." + label, value=value)
    seqs, mappings, scoring = {}, {}, {}
    try:
        if sequences_text is not None:
            seqs = fasta(sequences_text)
    except ValueError as error:
        issues.append(str(error))
    for label, text in (("mapping", mappings_text), ("scoring", scoring_text)):
        if text is not None:
            try:
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError("must be a JSON object")
                if label == "mapping":
                    mappings = value.get("chains", {})
                else:
                    scoring = value
            except ValueError as error:
                issues.append(label + ": " + str(error))
    if not isinstance(mappings, dict):
        issues.append("mapping.chains must be an object")
        mappings = {}
    chain_genes = spec.get("chain_genes", {})
    if (not isinstance(chain_genes, dict)
            or any(not isinstance(g, str) for g in chain_genes.values())):
        chain_genes = {}
    if target:
        expected = sorted([target["ligand"], target["type_I"], target["type_II"]] * 2)
        if sorted(chain_genes.values()) != expected or set(chain_genes) != set(seqs):
            issues.append("chain_genes/FASTA must specify the target's six chains (2:2:2)")
        for gene in set(expected):
            copies = [seqs[c] for c, g in chain_genes.items() if g == gene and c in seqs]
            if len(copies) == 2 and copies[0] != copies[1]:
                issues.append("duplicate chains for " + gene + " must have identical sequences")
    mapped = set()
    for chain, sequence in seqs.items():
        entry = mappings.get(chain, {})
        rows = entry.get("residues", []) if isinstance(entry, dict) else []
        if not isinstance(rows, list):
            rows = []
        if not isinstance(entry, dict) or entry.get("gene") != chain_genes.get(chain):
            issues.append("mapping gene mismatch for chain " + chain)
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                issues.append("malformed residue mapping in chain " + chain)
                continue
            pos = row.get("input_residue")
            if (not integer(pos, 1) or pos > len(sequence) or pos in seen
                    or row.get("aa") != sequence[pos - 1]
                    or not row.get("source") or row.get("source_residue") in (None, "")):
                issues.append("invalid/duplicate residue mapping in chain " + chain)
                continue
            seen.add(pos)
            mapped.add((chain, pos))
        if seen != set(range(1, len(sequence) + 1)):
            issues.append("incomplete residue mapping for chain " + chain)
    if not seqs or not mappings:
        issues.append("sequences and complete residue mappings are required")
    if scoring.get("version") != spec.get("scoring_version"):
        issues.append("scoring file version does not match scoring_version")
    rules = scoring.get("rules")
    if not isinstance(rules, list):
        issues.append("scoring.rules must be an explicit list (empty for unguided)")
        rules = []
    active, rule_ids, active_interfaces = [], set(), set()
    for rule in rules:
        if not isinstance(rule, dict) or type(rule.get("active")) is not bool:
            issues.append("every scoring rule needs an explicit boolean active field")
            continue
        if not rule["active"]:
            continue
        active.append(rule)
        rule_id = rule.get("rule_id")
        if not safe_name(rule_id) or not isinstance(rule.get("evidence"), str) or not rule["evidence"].strip():
            issues.append("active rules require rule_id and evidence")
        elif rule_id in rule_ids:
            issues.append("active rule IDs must be unique")
        else:
            rule_ids.add(rule_id)
        if not positive(rule.get("weight")):
            issues.append("active rules require a finite positive weight")
        if "mapping_status" in rule and rule["mapping_status"] != "verified":
            issues.append("active rule mapping_status must be verified when supplied")
        if "activation_status" in rule and rule["activation_status"] != "active":
            issues.append("active rule activation_status must be active when supplied")
        kind, interface = rule.get("kind"), rule.get("interface")
        if kind not in ("residue_contact", "residue_geometry", "global_geometry"):
            issues.append("active rule kind must be residue_contact, residue_geometry or global_geometry")
        if interface not in ("type_I", "type_II", "geometry"):
            issues.append("active rule interface must be type_I, type_II or geometry")
        else:
            active_interfaces.add(interface)
        if interface in ("type_I", "type_II") and kind == "global_geometry":
            issues.append("interface guidance requires a mapped residue rule")
        if kind in ("residue_contact", "residue_geometry") and rule.get("residue_based") is not True:
            issues.append("residue rule kinds must declare residue_based=true")
        if kind == "global_geometry" and rule.get("residue_based") is not False:
            issues.append("global_geometry must declare residue_based=false")
        refs = rule.get("residues", [])
        if type(rule.get("residue_based")) is not bool:
            issues.append("active rules must explicitly declare residue_based")
        if rule.get("residue_based") and not refs:
            issues.append("active residue rule has no mapped residue references")
        if not isinstance(refs, list):
            issues.append("rule residues must be a list")
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                issues.append("malformed active residue reference")
                continue
            chain, pos = ref.get("chain"), ref.get("input_residue")
            if (not isinstance(chain, str) or not integer(pos, 1)
                    or (chain, pos) not in mapped or ref.get("aa") != seqs[chain][pos - 1]):
                issues.append("active residue rule references an unmapped/mismatched residue")
        if interface in ("type_I", "type_II") and target:
            genes = {chain_genes.get(ref.get("chain")) for ref in refs
                     if isinstance(ref, dict) and isinstance(ref.get("chain"), str)}
            if target["ligand"] not in genes or target[interface] not in genes:
                issues.append("interface rule must map both the target ligand and its receptor")
    if spec.get("condition") == "unguided" and active:
        issues.append("unguided condition cannot contain active guidance rules")
    elif spec.get("condition") != "unguided" and not active:
        issues.append("guided condition requires active, mapped rules")
    required_interfaces = {
        "type_I_interface_guidance_only": {"type_I"},
        "type_I_and_type_II_interface_guidance": {"type_I", "type_II"},
        "full_interface_guidance_plus_target_specific_geometry": {"type_I", "type_II", "geometry"},
    }.get(spec.get("condition"), set())
    if required_interfaces - active_interfaces:
        issues.append("guided condition lacks active rules for: " + ", ".join(sorted(required_interfaces - active_interfaces)))
    if spec.get("condition") == "type_I_interface_guidance_only" and active_interfaces - {"type_I"}:
        issues.append("type-I-only condition cannot activate type-II or geometry guidance")
    command = spec.get("command_argv")
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
        issues.append("command_argv must record the exact nonempty argv; no command is inferred")
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", str(spec.get("code_revision", ""))):
        issues.append("code_revision must be a full git commit ID")
    for key in ("repository", "construct_handling", "glycan_handling", "disulfide_handling"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            issues.append(key + " must be explicitly documented")
    resources = spec.get("resources") or {}
    budget = spec.get("budget") or {}
    sampling = spec.get("sampling") or {}
    if not all(isinstance(x, dict) for x in (resources, budget, sampling)):
        issues.append("resources, budget and sampling must be objects")
        resources, budget, sampling = {}, {}, {}
    for key in ("account", "partition", "gpu_type"):
        if not isinstance(resources.get(key), str) or not resources[key]:
            issues.append("resources." + key + " is required")
    for key in ("gpu_count", "cpus"):
        if not integer(resources.get(key), 1):
            issues.append("resources." + key + " must be a positive integer")
    for key in ("walltime_minutes", "memory_gb"):
        if not positive(resources.get(key)):
            issues.append("resources." + key + " must be positive")
    for key in ("samples", "particles"):
        if not integer(sampling.get(key), 1):
            issues.append("sampling." + key + " must be a positive integer")
        maximum = budget.get("max_" + key)
        if not integer(maximum, 1):
            issues.append("budget.max_" + key + " is required")
        elif integer(sampling.get(key), 1) and sampling[key] > maximum:
            issues.append("sampling." + key + " exceeds the recorded budget")
    if not positive(budget.get("max_gpu_hours")) or not budget.get("scope"):
        issues.append("budget requires positive max_gpu_hours and a scope description")
    elif all(positive(resources.get(k)) for k in ("gpu_count", "walltime_minutes")):
        if resources["gpu_count"] * resources["walltime_minutes"] / 60 > budget["max_gpu_hours"]:
            issues.append("requested GPU walltime exceeds the recorded budget")
    return {"ready": not issues, "issues": issues, "artifacts": artifacts,
            "specification": spec, "snapshots": snapshots, "plan": plan_id,
            "specification_sha256": sha(spec_bytes)}


def run_dir(root, run_id):
    if not safe_name(run_id):
        raise ValueError("Unsafe run ID")
    return Path(root) / "runs" / run_id


def create_run(spec_path, root, plan_path=DEFAULT_PLAN, retry_of=None):
    report = preflight(spec_path, plan_path)
    spec = report["specification"]
    if not safe_name(spec.get("target_id")) or not safe_name(spec.get("scoring_version")):
        raise ValueError("A registered run requires safe target_id and scoring_version identifiers")
    if retry_of:
        prior = read_json(run_dir(root, retry_of) / "manifest.json")
        if prior["specification"]["target_id"] != spec["target_id"]:
            raise ValueError("A retry must retain the earlier target")
    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_"
              + spec["target_id"] + "_" + spec["scoring_version"] + "_" + uuid.uuid4().hex[:8])
    directory = run_dir(root, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "run_id": run_id, "created_at": now(),
                "retry_of": retry_of, "status": "ready_for_submission" if report["ready"] else "prepared_blocked",
                "job_id": None, **report}
    immutable_write(directory / "manifest.json", encode(manifest))
    append_event(root, run_id, manifest["status"], None, "Manifest registered; no job submitted")
    return manifest


def append_event(root, run_id, status, job_id=None, note="", output_dir=None, local_mirror_dir=None):
    manifest_path = run_dir(root, run_id) / "manifest.json"
    manifest_data = manifest_path.read_bytes()
    manifest = json.loads(manifest_data)
    if status not in STATUSES:
        raise ValueError("Unknown status")
    if status in ("ready_for_submission", "submitted", "running", "completed") and not manifest["ready"]:
        raise ValueError("Blocked manifest cannot acquire a launch/ready/completed status; create a new run after resolving preflight")
    if status in ("submitted", "running", "completed", "failed", "cancelled") and not job_id:
        raise ValueError("Slurm state events require an actual job ID")
    if job_id is not None and not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
        raise ValueError("Expected an actual numeric Slurm job ID (optionally array task)")
    events = sorted((manifest_path.parent / "events").glob("*.json"))
    prior_jobs = {read_json(p).get("job_id") for p in events} - {None}
    if status == "superseded" and (job_id or prior_jobs or manifest.get("job_id")):
        raise ValueError("Only unsubmitted preparations can be superseded; preserve observed Slurm states")
    if status == "superseded" and not note.strip():
        raise ValueError("A superseded preparation requires a reason in --note")
    if any(read_json(p)["status"] == "superseded" for p in events):
        raise ValueError("Superseded preparations are terminal; register a new run")
    if job_id and prior_jobs and job_id not in prior_jobs:
        raise ValueError("A different job ID requires a new run linked with --retry-of")
    timestamp = now()
    event = {"timestamp": timestamp, "run_id": run_id, "status": status,
             "job_id": job_id, "note": note, "manifest_sha256": sha(manifest_data)}
    if output_dir:
        event["output_dir"] = output_dir
    if local_mirror_dir:
        event["local_mirror_dir"] = local_mirror_dir
    name = timestamp.replace(":", "").replace("+", "_") + "_" + uuid.uuid4().hex + ".json"
    immutable_write(manifest_path.parent / "events" / name, encode(event))
    return event


def index_csv(root):
    rows = []
    for path in sorted((Path(root) / "runs").glob("*/manifest.json")):
        data = path.read_bytes()
        manifest, spec = json.loads(data), json.loads(data)["specification"]
        row = {key: spec.get(key) for key in ("target_id", "condition", "scoring_version", "seed", "replicate")}
        row.update({key: manifest.get(key) for key in ("run_id", "created_at", "retry_of", "status", "job_id")})
        for event_path in sorted((path.parent / "events").glob("*.json")):
            event = read_json(event_path)
            if event["manifest_sha256"] != sha(data):
                raise ValueError("Manifest changed after event: " + str(path))
            row["status"] = event["status"]
            if event.get("job_id"):
                row["job_id"] = event["job_id"]
            for key in ("output_dir", "local_mirror_dir"):
                if event.get(key):
                    row[key] = event[key]
        row["manifest_path"] = str(path.resolve())
        rows.append(row)
    fields = ["run_id", "target_id", "condition", "scoring_version", "seed", "replicate",
              "created_at", "retry_of", "status", "job_id", "output_dir", "local_mirror_dir", "manifest_path"]
    return csv_bytes(rows, fields)


HELP = """No commands are executed and no jobs are submitted by this utility.
Run preflight on the host holding the runner, checkpoint and input files.
Hashing reads entire checkpoints, potentially many GB. Run remote preflight/create
in a Slurm CPU compute job, not on a login node. No GPU is needed for hashing.
File paths in a specification are relative to that specification's directory.

From the repository root:
  python scripts/bmp90_registry.py --help
  python scripts/bmp90_registry.py inventory
  python scripts/bmp90_registry.py preflight artifacts/bmp90/run-spec.json
  python scripts/bmp90_registry.py create artifacts/bmp90/run-spec.json
  python scripts/bmp90_registry.py index > artifacts/bmp90/run_index.csv
Default plan: artifacts/bmp90/BMP_90_target_plan.json; default registry root:
artifacts/bmp90/. Override using --plan/--root before the subcommand.

Specification keys:
  target_id, condition (one of the plan paired_conditions), scoring_version,
  seed, replicate; sequences_fasta, residue_mapping_json, scoring_json, runner,
  checkpoint; command_argv (exact string array), repository, code_revision
  (full git hash), construct_handling, glycan_handling, disulfide_handling;
  chain_genes (FASTA chain ID -> gene, exactly two chains for each target gene);
  resources: {account, partition, gpu_type, gpu_count, cpus, memory_gb,
              walltime_minutes}; sampling: {samples, particles};
  budget: {max_gpu_hours, max_samples, max_particles, scope}.
Optional: hydra_experiment (named profile), resolved_config (resolved YAML
  path, required when hydra_experiment is set), auxiliary_inputs (label ->
  file path; reference structures, config sources or other dependencies),
  checkpoint_expected_sha256 (previously observed hash; rechecked against bytes).

Mapping JSON: {"chains": {"A": {"gene": "BMP4", "residues": [
  {"input_residue": 1, "aa": "S", "source": "UniProt:P12644",
   "source_residue": 293}]}}}. Supply a mapping for EVERY input residue;
  input_residue is 1-based within each FASTA chain, never a global offset.
  SimpleFold headers such as >A|protein are accepted as input chain A.
Scoring JSON: {"version": "v001", "rules": [{"rule_id": "example",
  "active": true, "residue_based": true, "evidence": "source and mapping basis",
  "weight": 1.0, "kind": "residue_contact", "interface": "type_I",
  "residues": [{"chain": "A", "input_residue": 1, "aa": "S"},
               {"chain": "C", "input_residue": 1, "aa": "Q"}]}]}.
  Active interface rules must map both ligand and receptor; kinds are
  residue_contact/residue_geometry/global_geometry and interface is
  type_I/type_II/geometry. Weight must be finite and positive. If supplied,
  mapping_status must be verified and activation_status must be active.
  Empty rules are allowed for unguided. These are provenance records, not
  a new sampling/scoring API. Freeze the real Hydra configuration using
  resolved_config and record its named experiment profile/actual argv.

create records blocked preparations as prepared_blocked. It freezes input
text and hashes every artifact, including the checkpoint; it never edits a
manifest. Run preflight again before a real submission to detect changed files.
Use --retry-of for every new attempt; each attempt receives a new run ID.
event adds one immutable JSON file; use verified Slurm IDs and observed states.
  Mark an outdated, never-submitted preparation with event RUN_ID superseded
  --note 'Reason for replacement'. This never alters the original manifest or
  pretends a scheduler cancellation occurred. Register a new run for new inputs.
  Link existing SimpleFold receipts with --output-dir and --local-mirror-dir:
  HPC /pub/ynkim4/ml-simplefold/artifacts/runs/JOB_ID/
  local <ml-simplefold>/artifacts/remote-runs/JOB_ID/.
  This registry's runs/ directory holds provenance only; sampling results stay
  in those existing job directories. It does not replace experiment.json,
  fk-experiment.json, resolved-config.yaml, validation.json or Hydra receipts.
  Evaluate PDB chains using validation.json saved_chain_to_input_chain:
  the existing writer regroups identical sequences when assigning PDB chains.
index emits a derived CSV to stdout (redirect as needed); source records remain
unchanged. A ready result checks consistency and required presence, not biological
validity, scheduler capacity, an executable runner API, or overall campaign spend.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=HELP,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Registry output directory (default: artifacts/bmp90)")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("inventory", help="Exclusively create the 90-target plan snapshot and CSV")
    for name in ("preflight", "create"):
        command = sub.add_parser(name, help="Validate spec" if name == "preflight" else "Register an immutable run, without submitting")
        command.add_argument("spec", type=Path)
        if name == "create":
            command.add_argument("--retry-of")
    event = sub.add_parser("event", help="Append a verified state observation")
    event.add_argument("run_id")
    event.add_argument("status", choices=STATUSES)
    event.add_argument("--job-id")
    event.add_argument("--note", default="")
    event.add_argument("--output-dir", help="Existing canonical HPC job output path")
    event.add_argument("--local-mirror-dir", help="Existing local remote-runs/JOB_ID mirror")
    sub.add_parser("index", help="Print derived run CSV to stdout")
    args = parser.parse_args(argv)
    try:
        if args.action == "inventory":
            result = inventory(args.plan, args.root)
        elif args.action == "preflight":
            result = preflight(args.spec, args.plan)
            result.pop("snapshots")
        elif args.action == "create":
            manifest = create_run(args.spec, args.root, args.plan, args.retry_of)
            result = {key: manifest[key] for key in ("run_id", "status", "ready", "issues")}
        elif args.action == "event":
            result = append_event(args.root, args.run_id, args.status, args.job_id, args.note,
                                  args.output_dir, args.local_mirror_dir)
        else:
            sys.stdout.write(index_csv(args.root).decode())
            return 0
        print(json.dumps(result, indent=2))
        return 2 if args.action == "preflight" and not result["ready"] else 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print("Error: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
