#!/usr/bin/env python3
"""Extract observed protein atoms from an mmCIF into mapped input numbering.

This deliberately small parser accepts the one-row-per-atom RCSB mmCIF format
and rejects unsupported alternate conformers/multiple models instead of guessing.
It never reconstructs missing atoms or residues. No third-party dependencies.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex

AA3 = dict(zip("ACDEFGHIKLMNPQRSTVWY", "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atom_rows(path):
    columns = []
    recording = False
    for line in Path(path).read_text().splitlines():
        if line.startswith("_atom_site."):
            columns.append(line.split()[0].split(".", 1)[1])
            recording = True
        elif recording and (line.startswith("ATOM ") or line.startswith("HETATM ")):
            values = shlex.split(line)
            if len(values) != len(columns):
                raise ValueError("Unsupported wrapped mmCIF atom row")
            yield dict(zip(columns, values))
        elif recording and line.startswith("#"):
            return


def prepare(cif_path, mapping_path, fasta_path, output_path):
    mapping = json.loads(Path(mapping_path).read_text())
    chains = mapping["chains"]
    fasta, current = {}, None
    for line in Path(fasta_path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            current = line[1:].split("|", 1)[0]
            if current in fasta:
                raise ValueError("Duplicate FASTA chain")
            fasta[current] = ""
        elif line:
            if current is None:
                raise ValueError("Missing FASTA header")
            fasta[current] += line
    if fasta != {c["input_chain"]: c["sequence"] for c in chains}:
        raise ValueError("FASTA and mapping sequences differ")
    source_map = {c["reference_author_chain"]: c for c in chains}
    if len(source_map) != len(chains):
        raise ValueError("Source author chains must map one-to-one")
    atom_records = []
    residue_atoms = {}
    seen = set()
    excluded = Counter()
    for a in atom_rows(cif_path):
        if a["group_PDB"] != "ATOM":
            excluded["nonprotein_HETATM"] += 1
            continue
        if a["auth_asym_id"] not in source_map:
            raise ValueError("Unmapped source protein chain")
        if a.get("pdbx_PDB_model_num", "1") != "1":
            raise ValueError("Multiple models require an explicit selection policy")
        if a["label_alt_id"] not in (".", "?"):
            raise ValueError("Alternate conformer requires an explicit selection policy")
        if a["pdbx_PDB_ins_code"] not in (".", "?"):
            raise ValueError("Insertion code requires explicit mapping")
        if float(a["occupancy"]) <= 0:
            excluded["zero_occupancy"] += 1
            continue
        c = source_map[a["auth_asym_id"]]
        rows = [r for r in c["residues"] if r["pdb_author_position"] == int(a["auth_seq_id"])]
        if len(rows) != 1:
            raise ValueError("Unmapped/ambiguous source author residue")
        r = rows[0]
        if AA3[r["amino_acid"]] != a["label_comp_id"]:
            raise ValueError("Reference/input amino-acid mismatch")
        key = (c["input_chain"], r["input_index_1based"], a["label_atom_id"])
        if key in seen:
            raise ValueError("Duplicate mapped atom")
        seen.add(key)
        residue_atoms.setdefault(key[:2], []).append(a["label_atom_id"])
        atom_records.append((key, a))
    order = {c["input_chain"]: i for i, c in enumerate(chains)}
    atom_records.sort(key=lambda x: (order[x[0][0]], x[0][1], int(x[1]["id"])))
    lines = ["REMARK 950 OBSERVED PROTEIN ATOMS ONLY; NO UNRESOLVED RESIDUES REBUILT",
             "REMARK 950 CHAINS AND RESIDUE NUMBERS FOLLOW FULL INPUT FASTA",
             "REMARK 950 SOURCE 9N4K; ENGINEERED GLYCOSYLATED RECEPTOR TRAP",
             "REMARK 950 GLYCANS AND HINGE/TAGS NOT REPRESENTED IN THIS REFERENCE"]
    for c in chains:
        seq = [AA3[x] for x in c["sequence"]]
        for i in range(0, len(seq), 13):
            lines.append(f"SEQRES {i//13+1:3d} {c['input_chain']} {len(seq):4d}  {' '.join(seq[i:i+13])}")
    serial, previous = 0, None
    for (chain, pos, name), a in atom_records:
        if previous is not None and previous != chain:
            serial += 1
            lines.append(f"TER   {serial:5d}")
        previous = chain
        serial += 1
        element = a["type_symbol"]
        atom_name = f" {name:<3}" if len(element) == 1 and len(name) < 4 else f"{name:<4}"
        lines.append(f"ATOM  {serial:5d} {atom_name} {a['label_comp_id']:>3} {chain}{pos:4d}    "
                     f"{float(a['Cartn_x']):8.3f}{float(a['Cartn_y']):8.3f}{float(a['Cartn_z']):8.3f}"
                     f"{float(a['occupancy']):6.2f}{float(a['B_iso_or_equiv']):6.2f}          {element:>2}  ")
    lines += ["TER", "END"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    coverage = []
    for c in chains:
        residues = []
        for r in c["residues"]:
            atoms = residue_atoms.get((c["input_chain"], r["input_index_1based"]), [])
            if bool(atoms) != r["has_reference_coordinates"]:
                raise ValueError("Observed coverage disagrees with source mapping")
            residues.append({"input_index_1based": r["input_index_1based"],
                             "amino_acid": r["amino_acid"],
                             "uniprot_position": r["uniprot_position"],
                             "source_author_position": r["pdb_author_position"],
                             "observed_atom_names": atoms})
        coverage.append({"input_chain": c["input_chain"], "gene": c["gene"],
                         "source_author_chain": c["reference_author_chain"],
                         "full_sequence": c["sequence"], "input_length": c["length"],
                         "observed_residue_count": sum(bool(r["observed_atom_names"]) for r in residues),
                         "missing_input_indices": [r["input_index_1based"] for r in residues if not r["observed_atom_names"]],
                         "residues": residues})
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "operation": "Exact extraction and renumbering; observed protein atoms only, no coordinate reconstruction or fitting",
              "source_url": "https://files.rcsb.org/download/9N4K.cif",
              "source_cif": str(cif_path), "source_cif_sha256": sha256(cif_path),
              "mapping_source": str(mapping_path), "mapping_sha256": sha256(mapping_path),
              "fasta_source": str(fasta_path), "fasta_sha256": sha256(fasta_path),
              "output_pdb": str(output_path), "output_pdb_sha256": sha256(output_path),
              "full_input_residues": sum(c["length"] for c in chains),
              "observed_residues": len(residue_atoms), "atom_count": len(atom_records),
              "excluded_atom_counts": dict(excluded), "model": 1,
              "coordinate_precision": "PDB 3 decimal Angstrom precision, same as source atom coordinates",
              "output_to_source_author_chain": {c["input_chain"]: c["reference_author_chain"] for c in chains},
              "coverage": coverage,
              "limitations": ["Reference is an engineered, glycosylated receptor trap; protein-only comparison does not establish native signaling.",
                              "Unresolved sequence is recorded but never assigned invented coordinates.",
                              "Same-reference guided recovery and residual contacts are not independent experimental validation."]}
    provenance = output_path.with_suffix(".provenance.json")
    provenance.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = prepare(args.cif, args.mapping, args.fasta, args.output)
    print(json.dumps({k: r[k] for k in ("output_pdb", "output_pdb_sha256", "full_input_residues", "observed_residues", "atom_count")}))


if __name__ == "__main__":
    main()
