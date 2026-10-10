"""Transfer the 2GOO BMPR1A contact objective to the observed 2H62 wrist.

ACVR2B packing and the original partial-complex penalties remain unchanged.
The unobserved wrist exchanges both BMP2 identities in the experimental
contact pattern; it is an inferred restraint, not a completed native structure.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

try:
    from .complex_interfaces import HOTSPOTS, _contacts
    from .fk_2h62_reward import DEFAULT_SETTINGS, PartialComplexReward
except ImportError:
    from complex_interfaces import HOTSPOTS, _contacts
    from fk_2h62_reward import DEFAULT_SETTINGS, PartialComplexReward


DEFAULT_CONTACT_SETTINGS = copy.deepcopy(DEFAULT_SETTINGS)
DEFAULT_CONTACT_SETTINGS["positive_weights"] = {
    "W60_packing": 2., "Y42_engagement": 1., "Q86_geometry": 0.,
    "BMPR1A_hotspot_contact_recall": 1.,
    "BMPR1A_native_contact_recall": 1.,
    "BMPR1A_native_contact_precision": .5,
}


class ContactComplexReward(PartialComplexReward):
    def __init__(self, reference_cif, reference_pdb, settings=None):
        configured = copy.deepcopy(DEFAULT_CONTACT_SETTINGS)
        for key, value in (settings or {}).items():
            if key in ("positive_weights", "penalties"):
                configured[key].update(value)
            else:
                configured[key] = value
        super().__init__(reference_cif, reference_pdb, configured)
        self.wrist_atom_keys = sorted(key for key in self.native if key[0] in ("A", "B", "D"))
        reference = self.references["B"]
        positions = dict(zip(reference.author_residues, reference.full_positions))
        self.wrist_hotspots = []
        for author, expected in HOTSPOTS["BMPR1A"]:
            position = positions.get(author)
            identity = reference.full_sequence[position] if position is not None else None
            status = ("unresolved_reference_residue" if position is None else
                      "ok" if identity == expected else "reference_residue_identity_mismatch")
            native = {contact for contact in self.native_contacts["B"]
                      if contact[0] == position} if status == "ok" else set()
            self.wrist_hotspots.append(dict(
                name=expected + author, reference_author_residue=author,
                expected_residue=expected, reference_residue_identity=identity,
                reference_full_sequence_position=position + 1 if position is not None else None,
                status=status, native_contacts=len(native),
                native_contact_pairs=[list(pair) for pair in sorted(native)]))
        self.metadata.update(
            reward_kind="2h62_partial_contacts",
            positive_weights=self.settings["positive_weights"],
            inactive_positive_terms=[key for key, weight in self.settings["positive_weights"].items()
                                     if weight == 0.],
            BMPR1A_hotspots=self.wrist_hotspots,
            BMPR1A_native_contact_count=len(self.native_contacts["B"]),
            BMPR1A_native_contact_pairs=[list(pair) for pair in sorted(self.native_contacts["B"])],
            BMPR1A_contact_policy=(
                "The 2GOO positive feature definitions and weights are reused, but native "
                "contacts come only from the observed 2H62 BMPR1A wrist and both BMP2 "
                "chains. Residue-pair contact means any occupied protein heavy-atom "
                "pair <=4 Å. Predictions are restricted to reference-observed sequence "
                "positions and atom names. Missing atoms do not shrink the full native "
                "contact denominator. Each copy averages defined F85 and Q86 recalls; "
                "a hotspot with no native contacts has null diagnostic recall. If "
                "neither recall is defined, the ranking feature is zero. Aggregate "
                "features take the worse BMPR1A copy under the coherent assignment."),
            BMPR1A_precision_policy=(
                "No predicted contacts gives undefined diagnostic precision (null) "
                "and a zero precision feature for ranking only."),
            BMPR1A_opposite_site_policy=(
                "Exchange BOTH ligand identities through the original _key mapping; "
                "the second wrist is an inferred contact restraint. No 2GOO coordinates, "
                "mirrored receptor coordinates, or fabricated native contact evidence."),
            Q86_policy=(
                "Original Q86 distances and angles remain per-copy diagnostics; "
                "Q86_geometry has zero weight in this profile. BMPR1A steering uses "
                "F85/Q86 contact recovery, full native-contact recovery and precision."))

    def _copy_features(self, atoms, receptor, ligand_mapping, site, component):
        result = super()._copy_features(atoms, receptor, ligand_mapping, site, component)
        if component != "BMPR1A":
            return result
        convert = lambda key: self._key(key, receptor, ligand_mapping, site)
        paired = {key: atoms[convert(key)] for key in self.wrist_atom_keys if convert(key) in atoms}
        native = self.native_contacts["B"]
        predicted = _contacts(paired, "B", ("A", "D"))
        recovered = native & predicted
        recall = len(recovered) / len(native) if native else 0.
        precision = len(recovered) / len(predicted) if predicted else None
        hotspots = []
        for annotation in self.wrist_hotspots:
            row = dict(annotation)
            position = row["reference_full_sequence_position"]
            reference_keys = {key for key in self.wrist_atom_keys
                              if key[0] == "B" and position is not None and key[1] == position - 1}
            native_site = {tuple(pair) for pair in row["native_contact_pairs"]}
            recovered_site = native_site & predicted
            hotspot_recall = len(recovered_site) / len(native_site) if native_site else None
            matched_atoms = len(reference_keys & set(paired))
            row.update(
                recovered_contacts=len(recovered_site), contact_recall=hotspot_recall,
                reference_heavy_atoms=len(reference_keys), matched_heavy_atoms=matched_atoms,
                heavy_atom_coverage=matched_atoms / len(reference_keys) if reference_keys else None)
            if row["status"] == "ok" and matched_atoms < len(reference_keys):
                row["status"] = "partial_prediction_atoms" if matched_atoms else "missing_prediction_residue_atoms"
            hotspots.append(row)
        defined_recalls = [row["contact_recall"] for row in hotspots
                           if row["contact_recall"] is not None]
        hotspot_recall = float(np.mean(defined_recalls)) if defined_recalls else 0.
        result["features"].update(
            BMPR1A_hotspot_contact_recall=hotspot_recall,
            BMPR1A_native_contact_recall=recall,
            BMPR1A_native_contact_precision=precision if precision is not None else 0.)
        result["details"]["BMPR1A_contacts"] = dict(
            site_evidence="experimental wrist contact pattern" if site == 0 else "inferred ligand-swapped wrist contact pattern",
            native_contacts=len(native), predicted_contacts=len(predicted), recovered_contacts=len(recovered),
            native_contact_recall=recall, native_contact_precision=precision,
            hotspot_contact_recall=hotspot_recall, hotspots=hotspots,
            reference_heavy_atoms=len(self.wrist_atom_keys), matched_heavy_atoms=len(paired),
            heavy_atom_coverage=len(paired) / len(self.wrist_atom_keys),
            recovered_contact_pairs=[list(pair) for pair in sorted(recovered)],
            predicted_contact_pairs=[list(pair) for pair in sorted(predicted)],
            ligand_reference_to_prediction={chain: convert((chain, 0, "CA"))[0] for chain in ("A", "D")})
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
    reward = ContactComplexReward(args.reference_cif, args.reference_pdb, settings)
    samples = [dict(source=str(path), **reward.score(path))
               for path in sorted(args.prediction_dir.glob("*.pdb"))]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(dict(metadata=reward.metadata, samples=samples),
                                          indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
