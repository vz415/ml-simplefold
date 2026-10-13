"""Build the full-construct BMP6 joint recipe and a separate held-out mask.

This deterministic local preparation does not launch inference. The generated
specification is an execution artifact; the scoring registry owns recipe history.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    prep = root / "artifacts/bmp90/preparation/bmp6"
    examples = root / "examples/bmp90"
    outputs = root / "artifacts/bmp90/specifications"
    examples.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((prep / "deposited_native_no_tags_mapping.json").read_text())
    original_fasta = prep / "deposited_native_no_tags.fasta"
    fasta = examples / "BMP6_ACVR1_ACVR2B_deposited_native_no_tags.fasta"
    fasta.write_bytes(original_fasta.read_bytes())
    by_chain = {c["input_chain"]: c for c in mapping["chains"]}
    by_auth = {c["reference_author_chain"]: c for c in mapping["chains"]}
    assert mapping["total_residues"] == 716
    for chain in mapping["chains"]:
        assert len(chain["sequence"]) == chain["length"]
        assert hashlib.sha256(chain["sequence"].encode()).hexdigest() == chain["sequence_sha256"]
    mapping.update({"active": False, "construct": "deposited_native_no_tags",
                    "usage": "Prepared for the authorized finite target sequence; this file does not submit jobs.",
                    "input_mapping_status": "verified deposited/canonical identities; output chain order must still be validated for each run",
                    "fasta_sha256": sha(fasta)})
    map_path = fasta.with_suffix(".mapping.json")
    map_path.write_text(json.dumps(mapping, indent=2) + "\n")
    reference = prep / "9N4K_deposited_native_reference.pdb"
    if sha(reference) != "b00134800a16d1bb300b96ebf87eb0ddfcb0eaed25e507b35cd1f5520afb77f5":
        raise ValueError("Unexpected full-numbered reference checksum; review preparation first")
    ca = {}
    for line in reference.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            ca[line[21], int(line[22:26])] = tuple(float(line[i:i + 8]) for i in (30, 38, 46))
    assert len(ca) == 552

    def point(chain, canonical):
        data = by_chain[chain]
        position = canonical - data["range_1based_inclusive"][0] + 1
        return {"chain": chain, "position": position,
                "residue": data["sequence"][position - 1], "atom": "CA",
                "canonical_position": canonical, "uniprot": data["uniprot"]}

    def pair(name, left, right, sigma=3.0, evidence="experimental_contact"):
        a, b = point(*left), point(*right)
        return {"id": name, "left": a, "right": b,
                "distance_angstrom": round(math.dist(ca[a["chain"], a["position"]],
                                                       ca[b["chain"], b["position"]]), 6),
                "sigma_angstrom": sigma, "weight": 1.0,
                "evidence_labels": [evidence, "9N4K", "engineered_glycosylated_reference"],
                "evidence_allocation": "guidance"}

    pairs, interfaces = [], []
    for receptor, near, far in (("C", "A", "B"), ("D", "B", "A")):
        members = [pair(f"ALK2_{receptor}_S41_N450", (receptor, 41), (near, 450)),
                   pair(f"ALK2_{receptor}_N60_E444", (receptor, 60), (near, 444)),
                   pair(f"ALK2_{receptor}_T83_L424", (receptor, 83), (far, 424))]
        pairs.extend(members)
        interfaces.append({"id": f"ACVR1_{receptor}", "receptor_chain": receptor,
                           "type": "type_I", "weight": 1.0, "pair_ids": [p["id"] for p in members]})
    for receptor, ligand in (("E", "B"), ("F", "A")):
        members = [pair(f"ActRIIB_{receptor}_Y60_L499", (receptor, 60), (ligand, 499)),
                   pair(f"ActRIIB_{receptor}_W78_S487", (receptor, 78), (ligand, 487)),
                   pair(f"ActRIIB_{receptor}_F101_I431", (receptor, 101), (ligand, 431))]
        pairs.extend(members)
        interfaces.append({"id": f"ACVR2B_{receptor}", "receptor_chain": receptor,
                           "type": "type_II", "weight": 1.0, "pair_ids": [p["id"] for p in members]})
    integrity = [pair("BMP6_dimer_C477_C477", ("A", 477), ("B", 477), 3.0, "reference_geometry_prior"),
                 pair("BMP6_dimer_N450_Y490", ("A", 450), ("B", 490), 4.0, "reference_geometry_prior"),
                 pair("BMP6_dimer_Y490_N450", ("A", 490), ("B", 450), 4.0, "reference_geometry_prior")]
    for chains, residues in (("AB", (412, 512)), ("CD", (35, 99)), ("EF", (29, 109))):
        for chain in chains:
            integrity.append(pair(f"fold_span_{chain}", (chain, residues[0]), (chain, residues[1]),
                                  3.0, "reference_geometry_prior"))
    all_guidance = pairs + integrity
    excluded = {}
    for p in all_guidance:
        for side in ("left", "right"):
            q = p[side]
            excluded.setdefault(by_chain[q["chain"]]["gene"], set()).add(q["canonical_position"])
    contacts = json.loads((prep / "9N4K_coordinate_analysis.json").read_text())["contacts_within_5A"]
    heldout = []
    for contact in contacts:
        receptor = by_auth[contact["receptor_auth_chain"]]
        ligand = by_auth[contact["ligand_auth_chain"]]
        if (contact["receptor_auth_residue"] in excluded[receptor["gene"]]
                or contact["ligand_auth_residue"] in excluded[ligand["gene"]]):
            continue
        heldout.append({**contact,
                        "left": point(receptor["input_chain"], contact["receptor_auth_residue"]),
                        "right": point(ligand["input_chain"], contact["ligand_auth_residue"])})
    heldout_path = outputs / "bmp6_joint_v1.heldout.json"
    heldout_path.write_text(json.dumps({
        "recipe_id": "bmp6_joint_v1", "reference": "9N4K", "contacts": heldout,
        "excluded_canonical_positions_by_gene": {k: sorted(v) for k, v in excluded.items()},
        "allocation": "Prespecified held-out reference contacts: not used for scoring or deterministic-decoy selection.",
        "exclusion": "Exclude contacts touching any guided interface or integrity endpoint, in either symmetry-equivalent copy.",
        "independence_limit": "Same experimental assembly and C2-related copies; held-out features are correlated and are not independent experiments.",
        "retirement_rule": "If used to tune this recipe, relabel as tuning and create a new evaluation allocation."}, indent=2) + "\n")
    required = []
    for c in mapping["chains"]:
        present = [r["input_index_1based"] for r in c["residues"] if r["has_reference_coordinates"]]
        assert present == list(range(min(present), max(present) + 1))
        required.append({"chain": c["input_chain"], "start": min(present), "end": max(present)})
    spec = {
        "schema_version": 2, "active": True, "mapping_status": "verified", "recipe_id": "bmp6_joint_v1",
        "target_id": "BMP6_ACVR1_ACVR2B", "fasta_sha256": sha(fasta),
        "scope": "Joint-interface protein-only reconstruction using full deposited native sequences without engineered tags. 9N4K glycan and receptor tether omitted; no native affinity or signaling claim.",
        "evidence_labels": ["https://www.pnas.org/doi/10.1073/pnas.2502788122", "https://www.rcsb.org/structure/9N4K"],
        "chains": [{"input_chain": c["input_chain"], "saved_chain": c["expected_saved_chain"],
                    "length": c["length"], "sequence_sha256": c["sequence_sha256"]} for c in mapping["chains"]],
        "equivalent_chain_groups": [["A", "B"], ["C", "D"], ["E", "F"]],
        "pairs": pairs, "interface_groups": interfaces, "aggregation": {"minimum_fraction": .5},
        "integrity_pairs": integrity,
        "safeguards": {"ca_clash_distance_angstrom": 3.0, "ca_clash_weight": 1.,
                       "backbone_min_angstrom": 2.8, "backbone_max_angstrom": 4.2,
                       "backbone_sigma_angstrom": .5, "backbone_weight": .2,
                       "missing_core_weight": 2., "maximum_penalty_exponent": 6.,
                       "integrity_factor_floor": .5, "required_ca_ranges": required},
        "calibration": {"status": "engineering_hypothesis_pending_sampling_and_benchmark_calibration",
                        "interface_sigma_angstrom": 3.0, "rationale": "Broad soft CA geometry preserves tolerance; widths are not atomic contact cutoffs or measured energies.",
                        "heldout_path": str(heldout_path.relative_to(root)), "heldout_sha256": sha(heldout_path),
                        "heldout_contact_count": len(heldout),
                        "score_never_estimates": ["DockQ", "binding energy", "affinity", "probability of correctness"]},
        "parent_relationships": [{"recipe_id": "2h62_contacts_l61_trimmed", "relationship": "conceptual_type_II_module_transfer",
                                  "retained": "ActRIIB aromatic engagement, joint receptor guidance and distinct copy occupancy",
                                  "numbering": "Successful BMP2 construct W60/Y42 map to canonical ACVR2B W78/Y60; BMP6 distances are measured anew in 9N4K.",
                                  "limitations": "No transplant of BMP2 ligand contacts, BMPR1A interface, L61/S88 geometry, numerical weights, or receptor trimming."},
                                 {"recipe_id": "bmp6_alk2_sparse_v1", "relationship": "type_I_evidence_parent_only",
                                  "change": "Re-map to716 residues, add both ActRIIB copies, use per-copy aggregation and physical/integrity safeguards; old552-aa sampling is not a matched control."}],
        "evidence_allocation": {"guidance": "12 interface CA distances and9 sparse dimer/fold geometry distances",
                                "context": "9N4K engineered receptor tethering, glycosylated BMP6, distinct ALK2/ALK3 mechanism",
                                "tuning": "Deterministic reference and incorrect-pose fixtures; not held-out assessment",
                                "heldout": "Separate residue-disjoint reference-contact file; symmetric copies are correlated"},
        "construct_manifest": str(map_path.relative_to(root)), "construct_manifest_sha256": sha(map_path),
        "reference_source": {"path": str(reference.relative_to(root)), "sha256": sha(reference),
                             "coordinate_coverage": "552 observed residues of716 total; missing tails are not modeled in the reference",
                             "source_url": "https://files.rcsb.org/download/9N4K.cif"},
        "physical_limits": ["CA safeguards do not replace all-atom clashes, rotamers, hydrogen bonds, protonation or disulfide checks.",
                            "Three distances per receptor do not uniquely determine its pose; retain plausible competing orientations.",
                            "Sparse fold-span distances do not prove native folds; compare full core geometry after sampling.",
                            "Dimer C477 CA spacing is a geometry prior, not an asserted interchain SG bond.",
                            "Full tails remain sampled; only experimentally observed core coordinates are required by the missing-core guard."],
        "expected_improvement": "Recover both receptor classes while limiting weak-copy masking and contact-satisfaction through severe CA distortions.",
        "falsification": "Independent populations repeatedly yield one failed copy or high-reward implausible/incorrect orientations after full physical and reference evaluation.",
        "activation_scope": "Prepared local scoring specification; parent owns finite-budget GPU submission and receipts."}
    spec_path = outputs / "bmp6_joint_v1.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({"fasta": str(fasta.relative_to(root)), "spec": str(spec_path.relative_to(root)),
                      "total_residues": 716, "interface_pairs": len(pairs), "integrity_pairs": len(integrity),
                      "heldout_contacts": len(heldout), "spec_sha256": sha(spec_path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    build(parser.parse_args().root)
