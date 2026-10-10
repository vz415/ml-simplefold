"""Reference-guided terminal/proxy rewards for the BMP2 receptor FK pilot.

Coordinate analysis only. The scalar is a declared experimental objective,
not a binding-energy estimate. Existing native-interface and clash definitions
are reused unchanged; each receptor feature uses its worse copy.
"""
from __future__ import annotations

import math

from analyze_complex_ensemble import REF_ORDER, read_prediction
from compare_structures import reference_ca
from complex_interfaces import InterfaceReference
from structure_clashes import analyze_clashes


def combine_reward(interfaces, clashes, settings):
    rows = interfaces['receptors']
    features = {}
    for component in ('BMPR1A', 'ACVR2A'):
        copies = [r for r in rows if r['component'] == component]
        if len(copies) != 2:
            raise ValueError(f'Expected two {component} copies')
        for field in ('hotspot_contact_recall', 'native_contact_recall',
                      'native_contact_precision'):
            # A disconnected receptor has undefined reported precision. For
            # ranking only, assign zero when there are explicitly no contacts.
            values = [0. if field == 'native_contact_precision' and r[field] is None
                      and r.get('predicted_contacts') == 0 else r[field] for r in copies]
            if any(v is None or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError(f'Undefined or invalid {component} {field}')
            features[f'{component}_{field}'] = min(values)
    features['BMP2_dimer_CA_RMSD_angstrom'] = interfaces['metrics']['BMP2_dimer_CA_RMSD_angstrom']
    features['BMPR1A_worst_placement_CA_RMSD_angstrom'] = max(
        r['ligand_frame_CA_RMSD_angstrom'] for r in rows if r['component'] == 'BMPR1A')
    features['clashes_per_1000_heavy_atoms'] = clashes['clashes_per_1000_heavy_atoms']
    if any(v is None or not math.isfinite(v) for v in features.values()):
        raise ValueError('Reward features must be finite')
    contributions = {}
    for key, weight in settings['positive_weights'].items():
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Reward weights must be finite and nonnegative')
        contributions[key] = weight * features[key]
    for key, spec in settings['penalties'].items():
        weight, scale, cap = spec['weight'], spec['scale'], spec['cap']
        if (not all(math.isfinite(v) for v in (weight, scale, cap))
                or weight < 0 or scale <= 0 or cap <= 0):
            raise ValueError('Penalty weight/scale/cap are invalid')
        contributions[key] = -weight * min(max(features[key], 0) / scale, cap)
    return {'reward': sum(contributions.values()), 'features': features,
            'contributions': contributions}


class ComplexReward:
    def __init__(self, reference_cif, reference_pdb, settings):
        self.references = {c: reference_ca(reference_cif, c) for c in REF_ORDER}
        if sum(len(c.full_sequence) for c in self.references.values()) != 694:
            raise ValueError('Expected the six-chain 694-residue complex reference')
        self.interface_reference = InterfaceReference(reference_pdb, self.references)
        self.settings = settings

    def score(self, path):
        chains = read_prediction(path, self.references)
        interfaces = self.interface_reference.analyze(path, chains)
        if interfaces['metrics']['reference_heavy_atom_coverage'] < 0.99:
            raise ValueError('Prediction is missing reference-observed heavy atoms')
        clashes = analyze_clashes(path)
        result = combine_reward(interfaces, clashes, self.settings)
        result['interfaces'] = interfaces
        result['clashes'] = clashes
        return result
