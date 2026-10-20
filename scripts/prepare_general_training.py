"""Prefetch a finite mixed-training plan from full PDB and authors' SwissProt catalogs.

Downloads and preparation run in Slurm. The immutable plan is replayed by training;
raw/processed examples are reusable across plans. This does not mirror the full
corpus or reconstruct the byte-identical historical SimpleFold training dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/simplefold"))
PREPARATION_VERSION = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run general-data downloads and preparation through Slurm")


def download(url, path, cfg):
    """Download once, then verify cached content and URL on every reuse."""
    require_slurm()
    path = Path(path)
    receipt_path = path.with_name(path.name + ".download.json")
    if path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["url"] != url or digest(path) != receipt["sha256"]:
            raise RuntimeError(f"Download cache integrity mismatch: {path}")
        return receipt
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(int(cfg["download_attempts"])):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SimpleFold-finetuning/1.0"})
            with urllib.request.urlopen(request, timeout=cfg["timeout_seconds"]) as response:
                temporary = path.with_name(path.name + ".partial")
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            if not temporary.stat().st_size:
                raise ValueError("Empty response")
            temporary.replace(path)
            receipt = dict(url=url, sha256=digest(path), bytes=path.stat().st_size,
                           downloaded_at=datetime.now(timezone.utc).isoformat())
            write_json(receipt_path, receipt)
            return receipt
        except urllib.error.HTTPError as error:
            if error.code in (400, 401, 403, 404, 410):
                raise
            if attempt + 1 == int(cfg["download_attempts"]):
                raise
        except (OSError, ValueError):
            if attempt + 1 == int(cfg["download_attempts"]):
                raise
        time.sleep(2 ** attempt)


def rcsb_query(cfg):
    def terminal(attribute, operator, value):
        return dict(type="terminal", service="text", parameters=dict(
            attribute=attribute, operator=operator, value=value))
    return dict(query=dict(type="group", logical_operator="and", nodes=[
        terminal("rcsb_accession_info.initial_release_date", "less_or_equal",
                 cfg["pdb"]["released_before"]),
        terminal("rcsb_entry_info.resolution_combined", "less_or_equal",
                 cfg["pdb"]["max_resolution"]),
        terminal("rcsb_entry_info.polymer_entity_count_protein", "greater", 0),
    ]), return_type="entry", request_options=dict(return_all_hits=True,
                                                  results_content_type=["experimental"]))


def build_catalog(cfg, cache):
    """Cache the complete metadata candidate pool before any finite-plan selection."""
    require_slurm()
    cache = Path(cache)
    metadata = cache / "catalog"
    query = rcsb_query(cfg)
    url = cfg["pdb"]["search_url"] + "?" + urllib.parse.urlencode({"json": json.dumps(query)})
    pdb_path = metadata / "rcsb-search.json"
    pdb_receipt = download(url, pdb_path, cfg)
    result = json.loads(pdb_path.read_text())
    pdb_ids = sorted({row["identifier"].lower() for row in result["result_set"]})
    if len(pdb_ids) != result["total_count"] or not pdb_ids:
        raise ValueError("Incomplete or empty RCSB catalog response")
    swiss_path = metadata / "swissprot-list.csv"
    swiss_receipt = download(cfg["swissprot"]["catalog_url"], swiss_path, cfg)
    swiss_ids = sorted({row[0].strip() for row in csv.reader(io.StringIO(swiss_path.read_text())) if row})
    if not swiss_ids or any(not re.fullmatch(r"AF-[A-Z0-9]+-F\d+-model_v4", x) for x in swiss_ids):
        raise ValueError("Unexpected authors' SwissProt list format/version")
    catalog = dict(schema_version=1, pdb=pdb_ids, swissprot=swiss_ids,
                   counts=dict(pdb=len(pdb_ids), swissprot=len(swiss_ids)),
                   sources=dict(pdb=pdb_receipt, swissprot=swiss_receipt), rcsb_query=query,
                   scope="Complete metadata candidate catalogs; sequence/structure eligibility is checked on demand. "
                         "Current PDB revisions and exact authors' AFDB v4 IDs; not a historical corpus mirror.")
    write_json(metadata / "catalog.json", catalog)
    return catalog


def load_cameo(cfg, cache):
    provenance = ROOT / cfg["cameo"]["provenance"]
    info = json.loads(provenance.read_text())
    path = Path(cache) / "catalog/cameo2022.csv"
    receipt = download(info["source_url"], path, cfg)
    if receipt["sha256"] != info["source_sha256"]:
        raise ValueError("Pinned full CAMEO22 CSV hash mismatch")
    targets = [dict(id=row["name"], sequence=row["seqres"].upper())
               for row in csv.DictReader(io.StringIO(path.read_text()))]
    if len(targets) != int(cfg["cameo"]["targets"]) or any(not x["sequence"] for x in targets):
        raise ValueError("Incomplete full CAMEO22 sequence set")
    return targets, dict(**receipt, provenance=str(provenance), provenance_sha256=digest(provenance),
                         targets=len(targets), settings=cfg["cameo"])


def general_candidate(rng, catalog, pdb_probability=None):
    pdb, swiss = catalog["pdb"], catalog["swissprot"]
    if pdb_probability is None:
        index = rng.randrange(len(pdb) + len(swiss))
        return ("pdb", pdb[index]) if index < len(pdb) else ("swissprot", swiss[index - len(pdb)])
    source = "pdb" if rng.random() < pdb_probability else "swissprot"
    return source, rng.choice(catalog[source])


def make_plan(catalog, synthetic_ids, steps, synthetic_probability, seed, pdb_probability=None):
    """Freeze source/record draws before any I/O or candidate failure can affect RNG."""
    if steps < 1 or not 0 <= synthetic_probability <= 1:
        raise ValueError("Invalid steps or synthetic probability")
    if pdb_probability is not None and not 0 <= pdb_probability <= 1:
        raise ValueError("Invalid conditional PDB probability")
    rng = random.Random(seed)
    draws = []
    for step in range(steps):
        if rng.random() < synthetic_probability:
            source, record_id = "synthetic", rng.choice(synthetic_ids)
        else:
            source, record_id = general_candidate(rng, catalog, pdb_probability)
        draws.append(dict(step=step, source=source, record_id=record_id.lower(), catalog_id=record_id))
    return dict(schema_version=1, seed=seed, steps=steps, synthetic_probability=synthetic_probability,
                pdb_probability=pdb_probability, draws=draws)


class HomologyScreen:
    def __init__(self, targets, cfg):
        from Bio.Align import PairwiseAligner, substitution_matrices
        self.aligner = PairwiseAligner(mode="local")
        self.aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        self.aligner.open_gap_score = cfg["open_gap_score"]
        self.aligner.extend_gap_score = cfg["extend_gap_score"]
        self.targets, self.cfg, self.checked = targets, cfg, {}

    def match(self, sequence):
        # Convert unsupported modified/ambiguous codes consistently with protein tokens.
        sequence = re.sub("[^ARNDCQEGHILKMFPSTWYVBZX*]", "X", sequence.upper())
        if sequence in self.checked:
            return self.checked[sequence]
        result = None
        for target in self.targets:
            other = re.sub("[^ARNDCQEGHILKMFPSTWYVBZX*]", "X", target["sequence"])
            if sequence == other:
                result = dict(target=target["id"], identity=1.0, shorter_coverage=1.0)
                break
            alignments = self.aligner.align(sequence, other)
            try:
                alignment = next(iter(alignments))
            except StopIteration:
                continue
            first, second = alignment.indices
            paired = (first >= 0) & (second >= 0)
            columns = len(first)
            matches = sum(sequence[int(a)] == other[int(b)] for a, b in zip(first[paired], second[paired]))
            identity = matches / columns if columns else 0.0
            coverage = int(paired.sum()) / min(len(sequence), len(other))
            if identity >= self.cfg["identity"] and coverage >= self.cfg["shorter_coverage"]:
                result = dict(target=target["id"], identity=identity, shorter_coverage=coverage)
                break
        self.checked[sequence] = result
        return result


class ExcludedCandidate(ValueError):
    pass


def raw_path(source, catalog_id, cache):
    return Path(cache) / "raw" / source / f"{catalog_id}.cif"


def download_candidate(source, catalog_id, cache, cfg):
    return download(cfg[source]["structure_url"].format(id=catalog_id),
                    raw_path(source, catalog_id, cache), cfg)


def protein_cif(raw, destination, cfg):
    """Keep protein polymers and their full SEQRES while expanding the first assembly."""
    import gemmi
    structure = gemmi.read_structure(str(raw))
    structure.merge_chain_parts()
    structure.remove_hydrogens()
    structure.remove_waters()
    structure.remove_alternative_conformations()
    proteins = {entity.name: list(entity.full_sequence) for entity in structure.entities
                if entity.entity_type == gemmi.EntityType.Polymer
                and entity.polymer_type == gemmi.PolymerType.PeptideL}
    if not proteins or any(not seq for seq in proteins.values()):
        raise ExcludedCandidate("No annotated L-protein entity with full sequence")
    assembly = None
    if cfg["pdb"]["biological_assembly"] and structure.assemblies:
        assembly = structure.assemblies[0].name
        structure.transform_to_assembly(assembly, gemmi.HowToNameCopiedChain.AddNumber)
    # Retain only model 1 and protein polymer residues; ligands/NA cannot enter the
    # upstream parser, which otherwise unconditionally treats every subchain as protein.
    while len(structure) > 1:
        del structure[-1]
    for chain in structure[0]:
        for index in range(len(chain) - 1, -1, -1):
            if chain[index].entity_id not in proteins:
                del chain[index]
    structure.remove_empty_chains()
    metadata, addresses = {}, set()
    for chain in structure[0]:
        for span in list(chain.subchains()):
            name = f"G{len(metadata)}"
            entity_id = span[0].entity_id
            metadata[name] = dict(entity=entity_id, full_sequence=proteins[entity_id],
                                  author_chain=chain.name, source_subchain=span.subchain_id())
            for residue in span:
                addresses.add((chain.name, str(residue.seqid)))
                residue.subchain = name
    if not metadata or len(metadata) > int(cfg["pdb"]["max_chains"]):
        raise ExcludedCandidate(f"Protein chain count outside 1..{cfg['pdb']['max_chains']}: {len(metadata)}")
    for index in range(len(structure.connections) - 1, -1, -1):
        connection = structure.connections[index]
        if any((partner.chain_name, str(partner.res_id.seqid)) not in addresses
               for partner in (connection.partner1, connection.partner2)):
            del structure.connections[index]
    structure.make_mmcif_document().write_file(str(destination))
    return metadata, assembly


def prepare_candidate(source, catalog_id, cache, cfg, components, screen, screen_hash):
    import gemmi
    import numpy as np
    from boltz_data_pipeline import const
    from boltz_data_pipeline.types import ChainInfo, Input, InterfaceInfo, Record
    from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
    from utils import mmcif_utils

    record_id = catalog_id.lower()
    raw = raw_path(source, catalog_id, cache)
    raw_receipt = download_candidate(source, catalog_id, cache, cfg)
    # Address both preparation settings and raw content: updated PDB revisions
    # cannot overwrite files hardlinked into an earlier immutable training plan.
    output = Path(cache) / "processed" / screen_hash / source / record_id / raw_receipt["sha256"]
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt["preparation_version"] == PREPARATION_VERSION
                and receipt["raw"]["sha256"] == raw_receipt["sha256"]
                and receipt["screen_sha256"] == screen_hash):
            for name, expected in receipt["files"].items():
                if digest(output / name) != expected:
                    raise RuntimeError(f"Processed cache integrity mismatch: {output / name}")
            return receipt, output
    output.mkdir(parents=True, exist_ok=True)
    cif = output / "protein.cif"
    metadata, assembly = protein_cif(raw, cif, cfg)
    original_parse_polymer = mmcif_utils.parse_polymer

    def parse_full_sequence(**kwargs):
        kwargs["sequence"] = metadata[kwargs["chain_id"]]["full_sequence"]
        return original_parse_polymer(**kwargs)

    # The upstream parser's observed-only sequence argument loses unresolved
    # residues. Override that one argument within this single-threaded parse.
    with patch.object(mmcif_utils, "parse_polymer", parse_full_sequence):
        parsed = mmcif_utils.parse_mmcif(str(cif), components, use_assembly=False)
    target = parsed.data
    original = gemmi.cif.read(str(raw))[0]
    deposited, released, revised = mmcif_utils.get_dates(original)
    info = replace(parsed.info, deposited=deposited, released=released, revised=revised,
                   resolution=mmcif_utils.get_resolution(original), method=mmcif_utils.get_method(original))
    if source == "pdb" and released and released > cfg["pdb"]["released_before"]:
        raise ExcludedCandidate(f"Release date {released} exceeds cutoff")
    if source == "pdb" and info.resolution > cfg["pdb"]["max_resolution"]:
        raise ExcludedCandidate(f"Resolution {info.resolution} exceeds cutoff")
    entities, copies, chains = {}, {}, []
    for chain in target.chains:
        meta = metadata[str(chain["name"])]
        entity = entities.setdefault(meta["entity"], len(entities))
        chain["entity_id"], chain["sym_id"] = entity, copies.get(entity, 0)
        copies[entity] = int(chain["sym_id"]) + 1
        start, count = int(chain["res_idx"]), int(chain["res_num"])
        residues = target.residues[start:start + count]
        if count != len(meta["full_sequence"]):
            raise ExcludedCandidate("Parser lost residues from the full entity sequence")
        # Token types define the exact ESM input, including UNK for modified residues.
        sequence = "".join(gemmi.find_tabulated_residue(const.tokens[int(r["res_type"])]).one_letter_code
                           if const.tokens[int(r["res_type"])] != "UNK" else "X" for r in residues).upper()
        match = screen.match(sequence)
        if match:
            raise ExcludedCandidate("CAMEO22 homology: " + json.dumps(dict(chain=str(chain["name"]), **match)))
        chains.append(dict(chain=str(chain["name"]), sequence=sequence,
                           entity_id=entity, sym_id=int(chain["sym_id"]),
                           unresolved_residues=int((~residues["is_present"]).sum()),
                           source_entity=meta["entity"], author_chain=meta["author_chain"],
                           source_subchain=meta["source_subchain"]))
    if not np.isfinite(target.atoms["coords"]).all() or not np.isfinite(target.atoms["conformer"]).all():
        raise ExcludedCandidate("Nonfinite atom coordinates or reference conformers")
    if not target.atoms["is_present"].any():
        raise ExcludedCandidate("No resolved protein atoms")
    tokens = BoltzTokenizer().tokenize(Input(target, {}))
    if len(tokens.tokens) != sum(len(chain["sequence"]) for chain in chains):
        raise ExcludedCandidate("Protein token/sequence length mismatch")
    info_chains = [ChainInfo(chain_id=i, chain_name=str(c["name"]), mol_type=int(c["mol_type"]),
                             cluster_id=-1, msa_id="", num_residues=int(c["res_num"]),
                             entity_id=int(c["entity_id"])) for i, c in enumerate(target.chains)]
    interfaces = [InterfaceInfo(int(x["chain_1"]), int(x["chain_2"])) for x in target.interfaces]
    record = Record(record_id, info, info_chains, interfaces)
    target.dump(output / "structure.npz")
    record.dump(output / "record.json")
    with (output / "tokens.pkl").open("wb") as stream:
        pickle.dump(tokens, stream)
    (output / "sequence.fasta").write_text("".join(f'>{x["chain"]}|{record_id}\n{x["sequence"]}\n' for x in chains))
    files = {name: digest(output / name) for name in
             ("structure.npz", "record.json", "tokens.pkl", "sequence.fasta", "protein.cif")}
    receipt = dict(preparation_version=PREPARATION_VERSION, source=source, id=record_id,
                   catalog_id=catalog_id, raw=raw_receipt, screen_sha256=screen_hash,
                   assembly=assembly, chains=chains, residues=len(target.residues),
                   resolved_atoms=int(target.atoms["is_present"].sum()), files=files)
    write_json(receipt_path, receipt)
    return receipt, output


def materialize(source, selected, dataset):
    directory = Path(dataset) / source
    mapping = {"structure.npz": ("structures", ".npz"), "record.json": ("records", ".json"),
               "tokens.pkl": ("tokens", ".pkl"), "sequence.fasta": ("fasta", ".fasta")}
    records = []
    for record_id, (_, cached) in sorted(selected.items()):
        for name, (subdir, suffix) in mapping.items():
            destination = directory / subdir / (record_id + suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if digest(destination) != digest(cached / name):
                    raise RuntimeError(f"Plan dataset file changed: {destination}")
                continue
            try:
                os.link(cached / name, destination)
            except OSError:
                shutil.copy2(cached / name, destination)
        records.append(json.loads((cached / "record.json").read_text()))
    write_json(directory / "manifest.json", records)
    (directory / "train.txt").write_text("".join(row["id"] + "\n" for row in records))
    (directory / "esm").mkdir(exist_ok=True)


def export_receipts(dataset, receipt_dir):
    if receipt_dir:
        for name in ("training-plan.json", "requested-training-plan.json", "plan-inputs.json",
                     "dataset-receipt.json", "candidate-exclusions.json", "candidate-replacements.json"):
            destination = Path(receipt_dir) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.resolve() != (dataset / name).resolve():
                shutil.copy2(dataset / name, destination)


def completed_plan(dataset, expected_inputs, receipt_dir):
    """Reuse a frozen complete plan only after checking its materialized files."""
    path = dataset / "dataset-receipt.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if any(receipt["inputs"].get(key) != value for key, value in expected_inputs.items()):
        raise RuntimeError("Completed plan inputs changed; choose a new dataset-dir")
    if json.loads((dataset / "plan-inputs.json").read_text()) != receipt["inputs"]:
        raise RuntimeError("Frozen plan inputs changed")
    if digest(dataset / "training-plan.json") != receipt["training_plan_sha256"]:
        raise RuntimeError("Frozen training plan changed")
    mapping = {"structure.npz": ("structures", ".npz"), "record.json": ("records", ".json"),
               "tokens.pkl": ("tokens", ".pkl"), "sequence.fasta": ("fasta", ".fasta")}
    expected_ids = {"pdb": set(), "swissprot": set()}
    for record in receipt["records"]:
        expected_ids[record["source"]].add(record["id"])
        for name, (subdir, suffix) in mapping.items():
            path = dataset / record["source"] / subdir / (record["id"] + suffix)
            if digest(path) != record["files"][name]:
                raise RuntimeError(f"Plan file integrity mismatch: {path}")
    for source, ids in expected_ids.items():
        rows = json.loads((dataset / source / "manifest.json").read_text())
        if {row["id"] for row in rows} != ids:
            raise RuntimeError(f"Plan manifest selection changed: {source}")
        if set((dataset / source / "train.txt").read_text().splitlines()) != ids:
            raise RuntimeError(f"Plan record list changed: {source}")
    export_receipts(dataset, receipt_dir)
    return receipt


def prepare(config_path, dataset_dir, ccd_path, synthetic_selection, steps=600,
            synthetic_probability=0.1, seed=82301, cache_dir=None, receipt_dir=None, phase="all"):
    import yaml
    require_slurm()
    config_path, dataset = Path(config_path), Path(dataset_dir)
    cfg = yaml.safe_load(config_path.read_text())
    cache = Path(cache_dir) if cache_dir else dataset.parent / "cache"
    dataset.mkdir(parents=True, exist_ok=True)
    expected = dict(config_sha256=digest(config_path), synthetic_selection_sha256=digest(synthetic_selection),
                    preparation_version=PREPARATION_VERSION, seed=seed, steps=steps,
                    synthetic_probability=synthetic_probability, pdb_probability=cfg.get("pdb_probability"))
    if phase != "catalog":
        expected["ccd_sha256"] = digest(ccd_path)
        finished = completed_plan(dataset, expected, receipt_dir)
        if finished is not None:
            return finished
    catalog = build_catalog(cfg, cache)
    if phase == "catalog":
        return dict(catalog=str(cache / "catalog/catalog.json"), counts=catalog["counts"])
    targets, cameo = load_cameo(cfg, cache)
    selection = json.loads(Path(synthetic_selection).read_text())
    synthetic_ids = [row["id"] for row in selection["records"]]
    inputs = dict(**expected, catalog_sha256=digest(cache / "catalog/catalog.json"),
                  cameo_sha256=cameo["sha256"])
    frozen = dataset / "plan-inputs.json"
    if frozen.exists() and json.loads(frozen.read_text()) != inputs:
        raise RuntimeError("Plan inputs changed; select a new dataset-dir for a different training plan")
    write_json(frozen, inputs)
    plan = make_plan(catalog, synthetic_ids, steps, synthetic_probability, seed, cfg.get("pdb_probability"))
    plan["inputs"] = inputs
    write_json(dataset / "requested-training-plan.json", plan)
    screen_hash = hashlib.sha256(json.dumps(dict(cameo=cameo["sha256"], settings=cfg,
        ccd_sha256=inputs["ccd_sha256"], preparation_version=PREPARATION_VERSION), sort_keys=True).encode()).hexdigest()
    screen = HomologyScreen(targets, cfg["cameo"])
    with Path(ccd_path).open("rb") as stream:
        components = pickle.load(stream)
    selected = {"pdb": {}, "swissprot": {}}
    exclusions_path = dataset / "candidate-exclusions.json"
    exclusions = json.loads(exclusions_path.read_text()) if exclusions_path.exists() else {}
    replacements = []
    candidates = sorted({(row["source"], row["catalog_id"]) for row in plan["draws"] if row["source"] != "synthetic"})

    def prefetch(candidate):
        try:
            download_candidate(*candidate, cache, cfg)
        except Exception as error:
            return candidate, f"{type(error).__name__}: {error}"
        return candidate, None

    # Network concurrency only; scoped parser overrides remain strictly sequential.
    with ThreadPoolExecutor(max_workers=max(1, min(4, int(cfg["download_workers"])))) as executor:
        for candidate, error in executor.map(prefetch, candidates):
            if error:
                exclusions["/".join(candidate)] = dict(source=candidate[0], catalog_id=candidate[1], reason=error)
    write_json(exclusions_path, exclusions)
    for row in plan["draws"]:
        if row["source"] == "synthetic":
            continue
        replacement_rng = random.Random(f"{seed}:{row['step']}:replacement")
        original_draw = dict(row)
        for attempt in range(int(cfg["max_candidate_attempts"])):
            source, catalog_id = row["source"], row["catalog_id"]
            key = f"{source}/{catalog_id}"
            if key not in exclusions:
                if row["record_id"] in selected[source]:
                    break
                try:
                    selected[source][row["record_id"]] = prepare_candidate(
                        source, catalog_id, cache, cfg, components, screen, screen_hash)
                    break
                except (ExcludedCandidate, ValueError, KeyError, OSError, RuntimeError) as error:
                    exclusions[key] = dict(source=source, catalog_id=catalog_id,
                                           reason=f"{type(error).__name__}: {error}")
                    write_json(exclusions_path, exclusions)
            replacements.append(dict(step=row["step"], rejected=dict(row), reason=exclusions[key]["reason"]))
            source, catalog_id = general_candidate(replacement_rng, catalog, cfg.get("pdb_probability"))
            row.update(source=source, record_id=catalog_id.lower(), catalog_id=catalog_id)
        else:
            raise RuntimeError(f"Could not prepare general draw at step {row['step']} after {attempt+1} candidates")
        if row != original_draw:
            row["requested"] = original_draw
        print(f"Prepared step {row['step']+1}/{steps}: {row['source']}/{row['record_id']}", flush=True)
        # Resume uses deterministic requested draws and the explicit exclusion log.
        write_json(dataset / "candidate-replacements.json", replacements)
    for source in selected:
        materialize(source, selected[source], dataset)
    counts = Counter(row["source"] for row in plan["draws"])
    pool = sum(catalog["counts"].values())
    pdb_probability = cfg.get("pdb_probability")
    if pdb_probability is None:
        pdb_probability = catalog["counts"]["pdb"] / pool
    plan.update(counts=dict(counts), catalog_counts=catalog["counts"],
                candidate_probabilities=dict(synthetic=synthetic_probability,
                    pdb=(1-synthetic_probability)*pdb_probability,
                    swissprot=(1-synthetic_probability)*(1-pdb_probability)),
                realized_probabilities={key: counts.get(key, 0)/steps for key in ("pdb", "swissprot", "synthetic")})
    write_json(dataset / "training-plan.json", plan)
    receipt = dict(inputs=inputs, training_plan_sha256=digest(dataset / "training-plan.json"),
                   dataset_dir=str(dataset.resolve()), cache_dir=str(cache.resolve()),
                   catalog_counts=catalog["counts"], counts=dict(counts),
                   prepared_unique={key: len(value) for key, value in selected.items()},
                   cameo=cameo, replacements=replacements, exclusions=list(exclusions.values()),
                   records=[receipt for values in selected.values() for receipt, _ in values.values()],
                   scope=catalog["scope"], crop="General: random <=512 tokens per load; synthetic: full assembly")
    write_json(dataset / "dataset-receipt.json", receipt)
    export_receipts(dataset, receipt_dir)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/general_training_data.yaml")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--receipt-dir", type=Path)
    parser.add_argument("--ccd", type=Path)
    parser.add_argument("--synthetic-selection", type=Path, default=ROOT / "configs/datasets/multimer_refined_v1.json")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--synthetic-probability", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=82301)
    parser.add_argument("--phase", choices=("catalog", "prepare", "all"), default="all")
    args = parser.parse_args()
    if args.phase != "catalog" and args.ccd is None:
        parser.error("--ccd is required for structure preparation")
    receipt = prepare(args.config, args.dataset_dir, args.ccd, args.synthetic_selection,
                      args.steps, args.synthetic_probability, args.seed,
                      args.cache_dir, args.receipt_dir, args.phase)
    print(json.dumps({key: receipt[key] for key in ("counts", "prepared_unique") if key in receipt}, indent=2))


if __name__ == "__main__":
    main()
