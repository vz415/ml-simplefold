"""Check that each predicted PDB is finite and covers the input sequence."""
import argparse
import json
from pathlib import Path

import gemmi
import numpy as np
from Bio import SeqIO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta_path", type=Path, required=True)
    parser.add_argument("--prediction_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()
    fastas = sorted(args.fasta_path.iterdir()) if args.fasta_path.is_dir() else [args.fasta_path]
    results = []
    expected_paths = set()
    for fasta in fastas:
        if fasta.suffix not in {".fa", ".fas", ".fasta"}:
            continue
        records = list(SeqIO.parse(fasta, "fasta"))
        expected_sequence = "".join(str(record.seq) for record in records)
        for sample in range(args.num_samples):
            path = args.prediction_dir / f"{fasta.stem}_sampled_{sample}.pdb"
            expected_paths.add(path)
            if not path.is_file():
                raise RuntimeError(f"Missing prediction: {path}")
            structure = gemmi.read_structure(str(path))
            residues = [residue for chain in structure[0] for residue in chain]
            sequence = "".join(gemmi.find_tabulated_residue(residue.name).one_letter_code for residue in residues)
            if sequence != expected_sequence:
                raise RuntimeError(f"Sequence mismatch in {path}: {sequence}")
            coords = np.array([[atom.pos.x, atom.pos.y, atom.pos.z]
                               for residue in residues for atom in residue])
            if not len(coords) or not np.isfinite(coords).all():
                raise RuntimeError(f"Invalid coordinates in {path}")
            if not all(sum(atom.name == "CA" for atom in residue) == 1 for residue in residues):
                raise RuntimeError(f"Missing or duplicate CA atoms in {path}")
            results.append({"path": str(path), "residues": len(residues), "atoms": len(coords),
                            "finite_coordinates": True, "sequence_matches": True})
    if not results or set(args.prediction_dir.glob("*.pdb")) != expected_paths:
        raise RuntimeError("Unexpected prediction count")
    report = args.prediction_dir.parent / "validation.json"
    report.write_text(json.dumps(results, indent=2) + "\n")
    print(report.read_text())


if __name__ == "__main__":
    main()
