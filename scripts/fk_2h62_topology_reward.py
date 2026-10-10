"""Typed 3D interface placement for completion of the partial 2H62 assembly.

Observed receptor-to-BMP2 Cα distance patterns define one wrist and one
knuckle site. Exchanging both BMP2 identities transfers each pattern to its
opposite site, without constructing or fitting missing receptor coordinates.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

try:
    from .fk_2h62_reward import DEFAULT_SETTINGS, PartialComplexReward
except ImportError:
    from fk_2h62_reward import DEFAULT_SETTINGS, PartialComplexReward


DEFAULT_TOPOLOGY_SETTINGS = {
    "distance_tolerance_angstrom": 2.,
    "distance_scale_angstrom": 5.,
    "receptor_anchor_count": 6,
    "ligand_anchor_count_per_chain": 4,
    "worst_copy_weight": .5,
    "missing_anchor_error_angstrom": 50.,
}


def huber_error(value):
    """Quadratic near the target and linear, without saturation, farther away."""
    return .5 * value * value if value <= 1. else value - .5


def _spread_anchors(native, keys, count, first=None):
    """Deterministic farthest-point selection from the declared Cα candidates."""
    keys = sorted(keys)
    if not keys:
        return []
    xyz = np.array([native[key] for key in keys])
    if first is None:
        first = keys[int(np.argmin(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1)))]
    selected = [first]
    remaining = [key for key in keys if key != first]
    while remaining and len(selected) < count:
        next_key = max(remaining, key=lambda key: min(
            float(np.sum((native[key] - native[anchor]) ** 2)) for anchor in selected))
        selected.append(next_key)
        remaining.remove(next_key)
    return selected


class TopologyComplexReward(PartialComplexReward):
    def __init__(self, reference_cif, reference_pdb, settings=None):
        configured = copy.deepcopy(DEFAULT_SETTINGS)
        configured["topology"] = dict(DEFAULT_TOPOLOGY_SETTINGS)
        configured["penalties"]["typed_interface_error"] = {
            "weight": 1., "scale": 1., "cap": None}
        for key, value in (settings or {}).items():
            if key in ("topology", "penalties", "positive_weights"):
                configured[key].update(value)
            else:
                configured[key] = value
        super().__init__(reference_cif, reference_pdb, configured)
        self.interface_patterns = {
            component: self._interface_pattern(chain)
            for component, chain in (("BMPR1A", "B"), ("ACVR2B", "C"))}
        self.metadata.update(
            reward_kind="2h62_partial_topology",
            topology_settings=self.settings["topology"],
            typed_interface_patterns=self.interface_patterns,
            topology_policy=(
                "Each receptor is assigned to its own type's observed or ligand-swapped "
                "Cα distance pattern. Ligand anchors span BOTH BMP2 monomers. The "
                "inferred opposite site swaps BOTH ligand identities, with no mirrored "
                "receptor coordinates. Identical copies may exchange labels in the "
                "same coherent eight-assignment search used for all other terms."),
            topology_error_policy=(
                "For each copy, RMS of absolute receptor–ligand distance errors outside "
                "the tolerance, divided by distance scale, then Huber loss. Aggregate "
                "(1-worst_copy_weight)*mean + worst_copy_weight*max across four copies. "
                "Missing anchor pairs contribute the fixed missing_anchor_error_angstrom "
                "rather than being dropped. The new penalty has no cap; broad errors "
                "remain distinguishable. Precise hotspot terms retain their original "
                "native-centered distance and angle windows."))

    def _interface_pattern(self, receptor):
        topology = self.settings["topology"]
        contacts = self.native_contacts[receptor]
        contact_keys = sorted({(receptor, position, "CA") for position, _, _ in contacts
                               if (receptor, position, "CA") in self.native})
        all_rec = sorted(key for key in self.native if key[0] == receptor and key[2] == "CA")
        receptor_keys = _spread_anchors(self.native, contact_keys,
                                       topology["receptor_anchor_count"])
        # If the observed interface has too few Cα positions, retain them and
        # obtain remaining anchors from the same observed receptor chain.
        if len(receptor_keys) < topology["receptor_anchor_count"]:
            for key in _spread_anchors(self.native, all_rec, len(all_rec)):
                if key not in receptor_keys:
                    receptor_keys.append(key)
                if len(receptor_keys) == topology["receptor_anchor_count"]:
                    break
        receptor_center = np.mean([self.native[key] for key in receptor_keys], axis=0)
        ligand_keys = []
        for chain in ("A", "D"):
            chain_keys = sorted(key for key in self.native if key[0] == chain and key[2] == "CA")
            interface_keys = sorted({(chain, position, "CA") for _, c, position in contacts
                                     if c == chain and (chain, position, "CA") in self.native})
            # Start at an interface position when available; spread the other
            # landmarks over the whole monomer, including the contextual half.
            first = min(interface_keys or chain_keys,
                        key=lambda key: float(np.linalg.norm(self.native[key] - receptor_center)))
            ligand_keys.extend(_spread_anchors(
                self.native, chain_keys, topology["ligand_anchor_count_per_chain"], first))
        targets = np.linalg.norm(
            np.array([self.native[key] for key in receptor_keys])[:, None] -
            np.array([self.native[key] for key in ligand_keys])[None], axis=-1)
        return dict(
            reference_receptor_chain=receptor,
            receptor_anchor_keys=[list(key) for key in receptor_keys],
            ligand_anchor_keys=[list(key) for key in ligand_keys],
            target_distances_angstrom=targets.tolist(),
            anchor_selection=(
                "Receptor: spatially spread observed native-contact Cα positions, "
                "with same-chain observed Cα fallback if necessary. BMP2: one "
                "interface/nearest anchor followed by farthest-point landmarks "
                "across each monomer separately."))

    def _copy_features(self, atoms, receptor, ligand_mapping, site, component):
        result = super()._copy_features(atoms, receptor, ligand_mapping, site, component)
        pattern = self.interface_patterns[component]
        convert = lambda key: self._key(key, receptor, ligand_mapping, site)
        receptor_keys = [convert(key) for key in pattern["receptor_anchor_keys"]]
        ligand_keys = [convert(key) for key in pattern["ligand_anchor_keys"]]
        targets = np.array(pattern["target_distances_angstrom"])
        topology = self.settings["topology"]
        excess = np.full(targets.shape, topology["missing_anchor_error_angstrom"], dtype=float)
        measured = [[None for _ in ligand_keys] for _ in receptor_keys]
        present_pairs = 0
        for i, rec in enumerate(receptor_keys):
            for j, lig in enumerate(ligand_keys):
                if rec not in atoms or lig not in atoms:
                    continue
                distance = float(np.linalg.norm(atoms[rec] - atoms[lig]))
                measured[i][j] = distance
                excess[i, j] = max(abs(distance - targets[i, j]) -
                                   topology["distance_tolerance_angstrom"], 0.)
                present_pairs += 1
        error = float(np.sqrt(np.mean(excess ** 2)))
        normalized = error / topology["distance_scale_angstrom"]
        huber = huber_error(normalized)
        result["features"].update(
            typed_interface_copy_error_angstrom=error,
            typed_interface_copy_error_normalized=normalized,
            typed_interface_copy_huber_error=huber)
        result["details"]["typed_interface"] = dict(
            site_evidence="experimental distance pattern" if site == 0 else "inferred ligand-swapped distance pattern",
            receptor_anchor_keys=[list(key) for key in receptor_keys],
            ligand_anchor_keys=[list(key) for key in ligand_keys],
            target_distances_angstrom=targets.tolist(),
            measured_distances_angstrom=measured,
            post_tolerance_errors_angstrom=excess.tolist(),
            pair_count=int(targets.size), present_pair_count=present_pairs,
            anchor_pair_coverage=present_pairs / targets.size,
            error_rms_angstrom=error, normalized_error=normalized, huber_error=huber)
        return result

    def _assignment_features(self, copies):
        result = super()._assignment_features(copies)
        errors = [copy["features"]["typed_interface_copy_huber_error"] for copy in copies]
        weight = self.settings["topology"]["worst_copy_weight"]
        result["typed_interface_error"] = float((1. - weight) * np.mean(errors) + weight * max(errors))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cif", type=Path, required=True)
    parser.add_argument("--reference-pdb", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--settings-json", type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings_json.read_text()) if args.settings_json else None
    reward = TopologyComplexReward(args.reference_cif, args.reference_pdb, settings)
    samples = [dict(source=str(path), **reward.score(path))
               for path in sorted(args.prediction_dir.glob("*.pdb"))]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(dict(metadata=reward.metadata, samples=samples),
                                          indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
