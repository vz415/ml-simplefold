"""ACVR2B L61 local backbone geometry added to the 2H62 contact reward.

The observed L61-N / BMP2-S88-OG interaction constrains packing around W60.
The second site's geometry is inferred by exchanging both BMP2 identities;
this local restraint does not establish whole-receptor orientation or affinity.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

try:
    from .fk_2h62_contact_reward import DEFAULT_CONTACT_SETTINGS, ContactComplexReward
    from .fk_2h62_reward import angle
except ImportError:
    from fk_2h62_contact_reward import DEFAULT_CONTACT_SETTINGS, ContactComplexReward
    from fk_2h62_reward import angle


DEFAULT_L61_SETTINGS = copy.deepcopy(DEFAULT_CONTACT_SETTINGS)
DEFAULT_L61_SETTINGS["l61_geometry"] = {
    "distance_tolerance_angstrom": .75,
    "distance_scale_angstrom": 3.,
    "angle_tolerance_degrees": 20.,
    "angle_scale_degrees": 30.,
    "missing_geometry_error": 10.,
}
DEFAULT_L61_SETTINGS["penalties"]["ACVR2B_L61_geometry_error"] = {
    "weight": .5, "scale": 1., "cap": None,
}


def huber_excess(value, target, tolerance, scale):
    """Huber loss after a native-centered flat tolerance, dimensionless."""
    excess = max(abs(value - target) - tolerance, 0.) / scale
    return .5 * excess ** 2 if excess <= 1. else excess - .5


class L61ComplexReward(ContactComplexReward):
    def __init__(self, reference_cif, reference_pdb, settings=None):
        configured = copy.deepcopy(DEFAULT_L61_SETTINGS)
        for key, value in (settings or {}).items():
            if key in ("positive_weights", "penalties", "l61_geometry"):
                configured[key].update(value)
            else:
                configured[key] = value
        super().__init__(reference_cif, reference_pdb, configured)
        self.l61_residues = [self._l61_reference_residue("C", "60", "W"),
                             self._l61_reference_residue("C", "61", "L"),
                             self._l61_reference_residue("A", "88", "S")]
        w60, l61, s88 = self.l61_residues
        key = lambda row, name: (row["reference_chain"], row["full_position"], name)
        self.l61_distance_keys = (key(l61, "N"), key(s88, "OG"))
        self.l61_angle_keys = (
            (key(w60, "C"), key(l61, "N"), key(s88, "OG")),
            (key(l61, "CA"), key(l61, "N"), key(s88, "OG")),
            (key(s88, "CB"), key(s88, "OG"), key(l61, "N")),
        )
        self.l61_distance_target = self._geometry_measurement(self.native, self.l61_distance_keys)
        self.l61_angle_targets = [self._geometry_measurement(self.native, keys)
                                  for keys in self.l61_angle_keys]
        self.metadata.update(
            reward_kind="2h62_partial_contacts_l61",
            L61_geometry=dict(
                reference_residues=self.l61_residues,
                atom_key_positions="zero-based full-sequence indices",
                native_distance_atom_keys=self.l61_distance_keys,
                native_distance_angstrom=self.l61_distance_target,
                native_approach_angle_atom_keys=self.l61_angle_keys,
                native_approach_angles_degrees=self.l61_angle_targets,
                settings=self.settings["l61_geometry"],
                penalty=self.settings["penalties"]["ACVR2B_L61_geometry_error"],
                formula=("Per copy: Huber(distance excess / distance scale) + mean "
                         "Huber(approach-angle excess / angle scale). Excess is the "
                         "absolute native deviation outside the declared tolerance. "
                         "Aggregate the larger error of the two ACVR2B copies. "
                         "Subtract the declared weight times that error without a cap."),
                missing_geometry_policy=("Missing or degenerate native/predicted geometry "
                                         "has at least missing_geometry_error; partial "
                                         "measurements never shrink the three-angle denominator."),
                evidence=("Experimental occupied 2H62 local interaction for site 0; "
                          "inferred ligand-swapped local interaction for site 1."),
                interpretation=("Local heavy-atom backbone approach geometry near W60, "
                                "not an explicit-hydrogen hydrogen-bond test, a "
                                "whole-receptor orientation guarantee, or affinity.")))

    def _l61_reference_residue(self, chain, author, expected):
        reference = self.references[chain]
        position = dict(zip(reference.author_residues, reference.full_positions)).get(author)
        identity = reference.full_sequence[position] if position is not None else None
        status = ("unresolved_reference_residue" if position is None else
                  "ok" if identity == expected else "reference_residue_identity_mismatch")
        return dict(reference_chain=chain, reference_author_residue=author,
                    expected_residue=expected, reference_residue_identity=identity,
                    reference_full_sequence_position=position + 1 if position is not None else None,
                    full_position=position if status == "ok" else None, status=status)

    @staticmethod
    def _geometry_measurement(atoms, keys):
        if any(key not in atoms for key in keys):
            return None
        coordinates = [atoms[key] for key in keys]
        value = (float(np.linalg.norm(coordinates[0] - coordinates[1]))
                 if len(keys) == 2 else angle(*coordinates))
        return value if value is not None and np.isfinite(value) else None

    def _copy_features(self, atoms, receptor, ligand_mapping, site, component):
        result = super()._copy_features(atoms, receptor, ligand_mapping, site, component)
        if component != "ACVR2B":
            return result
        settings = self.settings["l61_geometry"]
        convert = lambda keys: tuple(self._key(key, receptor, ligand_mapping, site) for key in keys)
        distance_keys = convert(self.l61_distance_keys)
        value = self._geometry_measurement(atoms, distance_keys)
        distance_error = (huber_excess(value, self.l61_distance_target,
                                      settings["distance_tolerance_angstrom"],
                                      settings["distance_scale_angstrom"])
                          if value is not None and self.l61_distance_target is not None else None)
        distance = dict(atom_keys=distance_keys, distance_angstrom=value,
                        target_angstrom=self.l61_distance_target, error=distance_error)
        angles = []
        for native_keys, target in zip(self.l61_angle_keys, self.l61_angle_targets):
            keys = convert(native_keys)
            value = self._geometry_measurement(atoms, keys)
            error = (huber_excess(value, target, settings["angle_tolerance_degrees"],
                                 settings["angle_scale_degrees"])
                     if value is not None and target is not None else None)
            angles.append(dict(atom_keys=keys, angle_degrees=value, target_degrees=target, error=error))
        error = (distance_error or 0.) + sum(row["error"] or 0. for row in angles) / len(angles)
        complete = distance_error is not None and all(row["error"] is not None for row in angles)
        if not complete:
            error = max(error, settings["missing_geometry_error"])
        keys = set(distance_keys).union(*(row["atom_keys"] for row in angles))
        result["features"]["ACVR2B_L61_geometry_error"] = float(error)
        result["details"]["L61_geometry"] = dict(
            site_evidence="experimental local geometry" if site == 0 else "inferred ligand-swapped local geometry",
            distance=distance, approach_angles=angles, geometry_error=float(error),
            status="ok" if complete else "missing_or_degenerate_geometry",
            required_heavy_atoms=len(keys), matched_heavy_atoms=sum(key in atoms for key in keys),
            heavy_atom_coverage=sum(key in atoms for key in keys) / len(keys))
        return result

    def _assignment_features(self, copies):
        features = super()._assignment_features(copies)
        features["ACVR2B_L61_geometry_error"] = max(
            row["features"]["ACVR2B_L61_geometry_error"] for row in copies
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
    reward = L61ComplexReward(args.reference_cif, args.reference_pdb, settings)
    samples = [dict(source=str(path), **reward.score(path))
               for path in sorted(args.prediction_dir.glob("*.pdb"))]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(dict(metadata=reward.metadata, samples=samples),
                                          indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
