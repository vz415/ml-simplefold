#!/usr/bin/env python3
"""Run official DockQ in an isolated analysis interpreter; no installations."""
import argparse
import importlib.metadata
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model")
    p.add_argument("--native")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    try:
        from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
        version = importlib.metadata.version("DockQ")
    except ImportError as exc:
        result = {"status": "unassessed", "reason": str(exc), "version": None}
    else:
        result = {"status": "available", "version": version}
        if args.model and args.native:
            interfaces, total = run_on_all_native_interfaces(
                load_PDB(args.model), load_PDB(args.native),
                chain_map={"X": "X", "Y": "Y"}, no_align=True)
            if len(interfaces) != 1:
                raise ValueError(f"Expected one merged-ligand/receptor interface, got {interfaces.keys()}")
            result.update(status="assessed", interface=next(iter(interfaces.values())),
                          total=total, native_to_model_chain_map={"X": "X", "Y": "Y"},
                          no_align=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
