"""Cooperative ACVR2B hotspots and whole-interface contacts for 2H62.

Keep the BMPR1A contact block and four structural penalties. W60 and Y42
only earn hotspot credit jointly; L61 geometry is retained as a diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

try:
    from .complex_interfaces import _contacts
    from .fk_2h62_l61_reward import DEFAULT_L61_SETTINGS, L61ComplexReward
except ImportError:
    from complex_interfaces import _contacts
    from fk_2h62_l61_reward import DEFAULT_L61_SETTINGS, L61ComplexReward


DEFAULT_COOPERATIVE_SETTINGS = copy.deepcopy(DEFAULT_L61_SETTINGS)
DEFAULT_COOPERATIVE_SETTINGS["positive_weights"].update(
    W60_packing=0., Y42_engagement=0., ACVR2B_interface_score=3.)
DEFAULT_COOPERATIVE_SETTINGS["penalties"]["ACVR2B_L61_geometry_error"]["weight"] = 0.


def harmonic_mean(first, second):
    total = first + second
    return 2. * first * second / total if total else 0.


class CooperativeComplexReward(L61ComplexReward):
    def __init__(self, reference_cif, reference_pdb, settings=None):
        configured = copy.deepcopy(DEFAULT_COOPERATIVE_SETTINGS)
        for key, value in (settings or {}).items():
            if key in ("positive_weights", "penalties", "l61_geometry"):
                configured[key].update(value)
            else:
                configured[key] = value
        super().__init__(reference_cif, reference_pdb, configured)
        self.knuckle_atom_keys = sorted(key for key in self.native
                                        if key[0] in ("A", "C", "D"))
        self.metadata.update(
            reward_kind="2h62_partial_cooperative",
            ACVR2B_native_contact_count=len(self.native_contacts["C"]),
            ACVR2B_native_contact_pairs=[list(pair) for pair in sorted(self.native_contacts["C"])],
            ACVR2B_contact_policy=(
                "Observed 2H62 ACVR2B contacts against both BMP2 chains; residue-pair "
                "contact means an occupied protein heavy-atom pair <=4 Å. Restrict "
                "predictions to reference-observed positions and atom names. Missing "
                "atoms do not reduce the native denominator. Site 1 exchanges BOTH "
                "ligand identities; that contact pattern is an inferred restraint."),
            ACVR2B_score_formula=(
                "J_i=2*W_i*Y_i/(W_i+Y_i); F_i=2*C_i*P_i/(C_i+P_i); "
                "A_i=0.5*(J_i+F_i). Zero denominators give zero. "
                "ACVR2B contribution=3*min(A_1,A_2) under one coherent assignment. "
                "The configured coefficient may override 3."),
            ACVR2B_aggregation_policy=(
                "Combine J and F within each copy BEFORE taking the worse copy. "
                "Separate diagnostic minima are not used to reconstruct the reward."),
            ACVR2B_L61_policy="Distance and approach angles remain diagnostics; default penalty weight is zero.",
            ACVR2B_interpretation=(
                "Harmonic hotspot agreement and whole-interface contact F1 are "
                "additive, bounded geometry scores, not binding energies. W60 and "
                "Y42 are equally weighted within J. F can earn credit when J is zero."))

    def _copy_features(self, atoms, receptor, ligand_mapping, site, component):
        result = super()._copy_features(atoms, receptor, ligand_mapping, site, component)
        if component != "ACVR2B":
            return result
        convert = lambda key: self._key(key, receptor, ligand_mapping, site)
        paired = {key: atoms[convert(key)] for key in self.knuckle_atom_keys
                  if convert(key) in atoms}
        native = self.native_contacts["C"]
        predicted = _contacts(paired, "C", ("A", "D"))
        recovered = native & predicted
        recall = len(recovered) / len(native) if native else 0.
        precision = len(recovered) / len(predicted) if predicted else None
        features = result["features"]
        joint = harmonic_mean(features["W60_packing"], features["Y42_engagement"])
        contact_f1 = harmonic_mean(recall, precision if precision is not None else 0.)
        features.update(
            ACVR2B_native_contact_recall=recall,
            ACVR2B_native_contact_precision=precision if precision is not None else 0.,
            ACVR2B_joint_hotspot_score=joint,
            ACVR2B_contact_f1=contact_f1,
            ACVR2B_interface_score=.5 * (joint + contact_f1))
        result["details"]["ACVR2B_contacts"] = dict(
            site_evidence="experimental knuckle contact pattern" if site == 0
                          else "inferred ligand-swapped knuckle contact pattern",
            native_contacts=len(native), predicted_contacts=len(predicted),
            recovered_contacts=len(recovered), native_contact_recall=recall,
            native_contact_precision=precision, contact_f1=contact_f1,
            joint_hotspot_score=joint, interface_score=features["ACVR2B_interface_score"],
            reference_heavy_atoms=len(self.knuckle_atom_keys), matched_heavy_atoms=len(paired),
            heavy_atom_coverage=len(paired) / len(self.knuckle_atom_keys),
            recovered_contact_pairs=[list(pair) for pair in sorted(recovered)],
            predicted_contact_pairs=[list(pair) for pair in sorted(predicted)],
            ligand_reference_to_prediction={chain: convert((chain, 0, "CA"))[0]
                                            for chain in ("A", "D")})
        return result

    def _assignment_features(self, copies):
        features = super()._assignment_features(copies)
        for key in ("ACVR2B_native_contact_recall", "ACVR2B_native_contact_precision",
                    "ACVR2B_joint_hotspot_score", "ACVR2B_contact_f1"):
            features[key] = min(row["features"][key] for row in copies
                                if row["component"] == "ACVR2B")
        return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cif", type=Path, required=True)
    parser.add_argument("--reference-pdb", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--settings-json", type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings_json.read_text()) if args.settings_json else None
    reward = CooperativeComplexReward(args.reference_cif, args.reference_pdb, settings)
    samples = [dict(source=str(path), **reward.score(path))
               for path in sorted(args.prediction_dir.glob("*.pdb"),
                                  key=lambda path: int(path.stem.rsplit("_", 1)[1]))]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(dict(metadata=reward.metadata, samples=samples),
                                          indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
