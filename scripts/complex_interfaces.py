"""Native BMP2 receptor-interface diagnostics from coordinates, without inference.

Reference chains A/D are BMP2, B/E BMPR1A and C/F ACVR2A. Prediction
chains A/B, C/D and E/F respectively. All residue correspondences come from
the reference's full-sequence positions, never guessed author-number offsets.
"""
from __future__ import annotations

import gemmi
import numpy as np

try:
    from .compare_structures import amino_acid, altloc_priority, kabsch
except ImportError:
    from compare_structures import amino_acid, altloc_priority, kabsch

GROUPS = (("BMP2", ("A", "D"), ("A", "B")),
          ("BMPR1A", ("B", "E"), ("C", "D")),
          ("ACVR2A", ("C", "F"), ("E", "F")))
HOTSPOTS = {"BMPR1A": (("85", "F"), ("86", "Q")),
            "ACVR2A": (("42", "F"), ("60", "W"), ("83", "F"))}
BACKBONE = frozenset(("N", "CA", "C", "O"))


def _rmsd(first, second):
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1)))) if len(first) else None


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _read_atoms(path, chains):
    """Return heavy atoms keyed by (chain, full-sequence index, atom name)."""
    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError("Interface analysis requires exactly one coordinate model")
    authors = {c: dict(zip(data.author_residues, data.full_positions)) for c, data in chains.items()}
    selected = {}
    for chain in structure[0]:
        if chain.name not in chains:
            continue
        for residue in chain:
            letter = amino_acid(residue.name)
            if letter is None or str(residue.seqid) not in authors[chain.name]:
                continue
            position = authors[chain.name][str(residue.seqid)]
            if letter != chains[chain.name].full_sequence[position]:
                raise ValueError(f"Residue identity differs from sequence at {chain.name}:{residue.seqid}")
            for atom in residue:
                if atom.occ <= 0 or atom.element.name in ("H", "D"):
                    continue
                key = (chain.name, position, atom.name)
                priority = altloc_priority(atom.occ, atom.altloc.strip("\x00"))
                xyz = np.array((atom.pos.x, atom.pos.y, atom.pos.z))
                if not np.isfinite(xyz).all():
                    raise ValueError(f"Nonfinite atom at {chain.name}:{residue.seqid}:{atom.name}")
                if key not in selected or priority < selected[key][0]:
                    selected[key] = (priority, xyz)
    return {key: value[1] for key, value in selected.items()}


def _contacts(atoms, receptor, ligand, cutoff=4.0):
    """Set of (receptor sequence index, BMP2 chain, BMP2 sequence index)."""
    rec = [(key, xyz) for key, xyz in atoms.items() if key[0] == receptor]
    lig = [(key, xyz) for key, xyz in atoms.items() if key[0] in ligand]
    if not rec or not lig:
        return set()
    rec_xyz, lig_xyz = np.array([v for _, v in rec]), np.array([v for _, v in lig])
    result = set()
    # Chunks avoid allocating a full all-atom complex distance matrix.
    for start in range(0, len(rec), 128):
        delta = rec_xyz[start:start + 128, None] - lig_xyz[None]
        rows, columns = np.where(np.sum(delta * delta, axis=-1) <= cutoff * cutoff)
        for i, j in zip(rows, columns):
            result.add((rec[start + int(i)][0][1], lig[int(j)][0][0], lig[int(j)][0][1]))
    return result


def _dihedral(a, b, c, d):
    b0, b1, b2 = a - b, c - b, d - c
    length = np.linalg.norm(b1)
    if length < 1e-10:
        return None
    b1 = b1 / length
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    if min(np.linalg.norm(v), np.linalg.norm(w)) < 1e-10:
        return None
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _angles(atoms, chain, position):
    keys = ((chain, position - 1, "C"), (chain, position, "N"),
            (chain, position, "CA"), (chain, position, "C"), (chain, position + 1, "N"))
    if any(key not in atoms for key in keys):
        return None
    coordinates = [atoms[key] for key in keys]
    # Breaks in the backbone are not assigned secondary-structure propensity.
    if np.linalg.norm(coordinates[0] - coordinates[1]) > 2.0 or np.linalg.norm(coordinates[3] - coordinates[4]) > 2.0:
        return None
    phi, psi = _dihedral(*coordinates[:4]), _dihedral(*coordinates[1:])
    return None if phi is None or psi is None else (phi, psi)


class InterfaceReference:
    """Prepare one native reference, then score many full-sequence predictions.

    ``analyze`` returns serializable ``metrics``, ``receptors``, ``mapping``,
    ``rotation`` and ``translation``. Coordinates use the row-vector convention
    ``prediction @ rotation + translation``. ``metadata`` provides reference
    per-copy baselines and reproducible method definitions.
    """

    def __init__(self, reference_pdb, references):
        self.references = references
        self.atoms = _read_atoms(reference_pdb, references)
        self.contacts = {rc: _contacts(self.atoms, rc, ("A", "D"))
                         for _, refs, _ in GROUPS[1:] for rc in refs}
        self.symmetry_rotation, self.symmetry_translation, symmetry_distances = self._symmetry_operator()
        self.metadata = {
            "methods": {
                "alignment": "BMP2-dimer-only proper Kabsch CA fit, trying both identical ligand-copy assignments; receptor copies minimize CA RMSD in that fixed ligand frame. No receptor fit changes placement scores.",
                "contacts": "Residue-pair contact if any positive-occupancy nonhydrogen atom pair is <=4 Å. Reference denominator is the full native contact set. Prediction contacts use reference-observed residues and atom-name intersection; absent reference atoms cannot produce false contacts. Missing prediction atoms lower reported coverage and can leave native contacts unrecovered. Undefined ratios are null.",
                "interface_rmsd": "Single proper Kabsch fit to matched N/CA/C/O atoms of the reference contacting receptor and BOTH BMP2 chains for each receptor; all native-contact residues included. This is a backbone interface RMSD diagnostic, not DockQ.",
                "orientation": "Angle of the receptor-only best-fit proper rotation after BMP2 alignment. This separate fit diagnoses orientation only; it does not rescue placement RMSD. Fold changes can also affect the angle.",
                "symmetry": "CA consistency with the fixed proper-rotation operator fitted from reference ABC to DEF and reverse, at identical observed sequence positions. Symmetry alone does not establish native docking: inspect receptor placement and wrist/knuckle contact recovery too.",
                "alpha1": "BMPR1A author residues 82–88 (Gly82–Lys88 bound α1). Fraction with phi in [-100,-30] and psi in [-80,-5] degrees among residues with connected, available backbone dihedrals. Dihedral-based helix propensity, not DSSP or a full secondary-structure assignment. Higher propensity alone need not be more native-like: the reference Gly82 is outside this helical-angle window.",
                "hotspots": "BMPR1A F85/Q86 and ACVR2A F42/W60/F83 use reference author numbering mapped explicitly to full polymer sequence positions, with residue-identity checks. Missing or mismatched annotations yield null, never guessed numbering.",
            },
            "contact_cutoff_angstrom": 4.0,
            "reference_heavy_atoms": len(self.atoms),
            "reference_C2_operator_CA_RMSD_angstrom": float(np.sqrt(np.mean(symmetry_distances ** 2))),
            "reference_C2_rotation_degrees": self._rotation_angle(self.symmetry_rotation),
        }
        # Reference baseline rows reuse the exact per-copy metric definitions.
        self.metadata["reference_receptors"] = [self._receptor(rc, rc, component, self.atoms, self.atoms,
                                                                self.references[rc].coordinates)
            for component, refs, _ in GROUPS[1:] for rc in refs]
        baseline = {"BMP2_dimer_CA_RMSD_angstrom": 0.0,
                    "reference_anchored_receptor_placement_CA_RMSD_angstrom": 0.0,
                    "reference_heavy_atom_coverage": 1.0,
                    "reference_anchored_C2_CA_deviation_angstrom": self.metadata["reference_C2_operator_CA_RMSD_angstrom"]}
        for component, _, _ in GROUPS[1:]:
            rows = [row for row in self.metadata["reference_receptors"] if row["component"] == component]
            for field in ("ligand_frame_CA_RMSD_angstrom", "centroid_displacement_angstrom", "orientation_error_degrees",
                          "native_contact_recall", "native_contact_precision", "interface_backbone_RMSD_angstrom",
                          "hotspot_contact_recall"):
                baseline[f"{component}_{field}"] = _mean([row[field] for row in rows])
        alpha1 = [row["alpha1"] for row in self.metadata["reference_receptors"] if row["component"] == "BMPR1A"]
        baseline["BMPR1A_alpha1_helix_dihedral_fraction"] = _mean([row["helix_dihedral_fraction"] for row in alpha1])
        baseline["BMPR1A_alpha1_backbone_RMSD_angstrom"] = _mean([row["backbone_RMSD_angstrom"] for row in alpha1])
        self.metadata["metrics"] = baseline

    @staticmethod
    def _rotation_angle(rotation):
        return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))

    def _symmetry_operator(self):
        first, second = [], []
        for _, refs, _ in GROUPS:
            left, right = (self.references[c] for c in refs)
            lpos = dict(zip(left.full_positions, left.coordinates))
            rpos = dict(zip(right.full_positions, right.coordinates))
            common = sorted(set(lpos) & set(rpos))
            first.extend(lpos[p] for p in common)
            second.extend(rpos[p] for p in common)
        a, b = np.asarray(first), np.asarray(second)
        # Both halves enforce an approximately involutory reference operator.
        return kabsch(np.concatenate((a, b)), np.concatenate((b, a)))

    def _ligand_alignment(self, chains):
        fixed = np.concatenate([self.references[c].coordinates for c in ("A", "D")])
        candidates = []
        for ligand in (("A", "B"), ("B", "A")):
            mapping = dict(zip(("A", "D"), ligand))
            moving = np.concatenate([chains[mapping[c]].coordinates[self.references[c].full_positions]
                                     for c in ("A", "D")])
            rotation, translation, distances = kabsch(moving, fixed)
            receptor_error = 0.0
            for _, refs, predictions in GROUPS[1:]:
                assignment = []
                for names in (predictions, predictions[::-1]):
                    error = sum(float(np.sum((chains[pc].coordinates[self.references[rc].full_positions]
                                              @ rotation + translation - self.references[rc].coordinates) ** 2))
                                for rc, pc in zip(refs, names))
                    assignment.append((error, names))
                error, names = min(assignment, key=lambda item: item[0])
                mapping.update(zip(refs, names))
                receptor_error += error
            candidates.append((float(np.sqrt(np.mean(distances ** 2))), receptor_error,
                               mapping, rotation, translation))
        minimum = min(row[0] for row in candidates)
        return min((row for row in candidates if row[0] <= minimum + 1e-8), key=lambda row: row[1])

    def _hotspots(self, rc, pc, component, native, predicted, raw_mapped, chains):
        reference = self.references[rc]
        lookup = dict(zip(reference.author_residues, reference.full_positions))
        result = []
        for author, expected in HOTSPOTS[component]:
            position = lookup.get(author)
            item = dict(name=expected + author, reference_author_residue=author,
                        reference_full_sequence_position=position + 1 if position is not None else None,
                        prediction_author_residue=None, status="unresolved_reference_residue",
                        native_contacts=None, recovered_contacts=None, contact_recall=None,
                        reference_heavy_atoms=None, matched_heavy_atoms=None, heavy_atom_coverage=None,
                        nearest_ligand_heavy_atom_distance_angstrom=None)
            if position is not None and reference.full_sequence[position] != expected:
                item["status"] = "reference_residue_identity_mismatch"
            elif position is not None:
                item["status"] = "ok"
                prediction = chains[pc]
                plookup = dict(zip(prediction.full_positions, prediction.author_residues))
                item["prediction_author_residue"] = plookup.get(position)
                reference_keys = {key for key in self.atoms if key[:2] == (rc, position)}
                matched_keys = reference_keys & set(raw_mapped)
                item.update(reference_heavy_atoms=len(reference_keys), matched_heavy_atoms=len(matched_keys),
                            heavy_atom_coverage=len(matched_keys) / len(reference_keys) if reference_keys else None)
                if len(matched_keys) < len(reference_keys):
                    item["status"] = "partial_prediction_atoms" if matched_keys else "missing_prediction_residue_atoms"
                native_site = {contact for contact in native if contact[0] == position}
                recovered = native_site & predicted
                item.update(native_contacts=len(native_site), recovered_contacts=len(recovered),
                            contact_recall=len(recovered) / len(native_site) if native_site else None)
                rec_atoms = np.array([xyz for key, xyz in raw_mapped.items() if key[:2] == (rc, position)])
                lig_atoms = np.array([xyz for key, xyz in raw_mapped.items() if key[0] in ("A", "D")])
                if len(rec_atoms) and len(lig_atoms):
                    item["nearest_ligand_heavy_atom_distance_angstrom"] = float(np.sqrt(
                        np.min(np.sum((rec_atoms[:, None] - lig_atoms[None]) ** 2, axis=-1))))
                elif not len(rec_atoms):
                    item["status"] = "missing_prediction_residue_atoms"
            result.append(item)
        return result

    def _alpha1(self, rc, common, raw_mapped):
        reference = self.references[rc]
        lookup = dict(zip(reference.author_residues, reference.full_positions))
        positions = [lookup[str(author)] for author in range(82, 89) if str(author) in lookup]
        # Require the documented endpoint identities, so synthetic numbering
        # cannot accidentally be presented as the bound α1 helix.
        valid = ("82" in lookup and "88" in lookup
                 and reference.full_sequence[lookup["82"]] == "G"
                 and reference.full_sequence[lookup["88"]] == "K")
        result = dict(reference_author_range="82–88", status="ok" if valid else "unavailable_or_identity_mismatch",
                      expected_residues=7, mapped_reference_residues=len(positions),
                      evaluated_residues=0, helix_dihedral_fraction=None,
                      backbone_RMSD_angstrom=None, reference_backbone_atoms=0,
                      matched_backbone_atoms=0, residues=[])
        if not valid:
            return result
        for position in positions:
            angles = _angles(raw_mapped, rc, position)
            author = reference.author_residues[reference.full_positions.index(position)]
            row = dict(reference_author_residue=author, phi_degrees=None, psi_degrees=None,
                       helix_like=None)
            if angles is not None:
                phi, psi = angles
                row.update(phi_degrees=phi, psi_degrees=psi,
                           helix_like=bool(-100 <= phi <= -30 and -80 <= psi <= -5))
            result["residues"].append(row)
        flags = [row["helix_like"] for row in result["residues"] if row["helix_like"] is not None]
        result["evaluated_residues"] = len(flags)
        result["helix_dihedral_fraction"] = float(np.mean(flags)) if flags else None
        ref_keys = [key for key in self.atoms if key[0] == rc and key[1] in positions and key[2] in BACKBONE]
        keys = [key for key in ref_keys if key in common]
        result["reference_backbone_atoms"], result["matched_backbone_atoms"] = len(ref_keys), len(keys)
        if len(keys) >= 3 and any(key[2] in ("N", "C", "O") for key in keys):
            _, _, distances = kabsch(np.array([common[key] for key in keys]),
                                     np.array([self.atoms[key] for key in keys]))
            result["backbone_RMSD_angstrom"] = float(np.sqrt(np.mean(distances ** 2)))
        return result

    def _receptor(self, rc, pc, component, common, raw_mapped, placed_ca, chains=None):
        chains = chains or self.references
        fixed = self.references[rc].coordinates
        rotation, _, _ = kabsch(placed_ca, fixed)
        native, predicted = self.contacts[rc], _contacts(common, rc, ("A", "D"))
        recovered = native & predicted
        row = dict(component=component, epitope="wrist" if component == "BMPR1A" else "knuckle",
                   reference_chain=rc, prediction_chain=pc, matched_CA=len(fixed),
                   ligand_frame_CA_RMSD_angstrom=_rmsd(placed_ca, fixed),
                   centroid_displacement_angstrom=float(np.linalg.norm(placed_ca.mean(axis=0) - fixed.mean(axis=0))),
                   orientation_error_degrees=self._rotation_angle(rotation),
                   native_contact_recall=len(recovered) / len(native) if native else None,
                   native_contact_precision=len(recovered) / len(predicted) if predicted else None,
                   reference_contacts=len(native), predicted_contacts=len(predicted), recovered_contacts=len(recovered),
                   interface_backbone_RMSD_angstrom=None,
                   monomer_contacts=[], hotspots=[])
        for ligand in ("A", "D"):
            native_subset = {contact for contact in native if contact[1] == ligand}
            predicted_subset = {contact for contact in predicted if contact[1] == ligand}
            recovered_subset = native_subset & predicted_subset
            row["monomer_contacts"].append(dict(reference_chain=ligand,
                reference_count=len(native_subset), predicted_count=len(predicted_subset),
                recovered_count=len(recovered_subset),
                recall=len(recovered_subset) / len(native_subset) if native_subset else None))
        row["contacted_BMP2_monomer_count"] = sum(item["predicted_count"] > 0 for item in row["monomer_contacts"])
        if component == "BMPR1A":
            row["contacts_both_BMP2_monomers"] = row["contacted_BMP2_monomer_count"] == 2
        else:
            native_monomer = max(row["monomer_contacts"], key=lambda item: item["reference_count"])
            row["native_knuckle_BMP2_chain"] = native_monomer["reference_chain"] if native else None
            row["native_monomer_contact_fraction"] = native_monomer["predicted_count"] / len(predicted) if predicted and native else None
        receptor_positions = {contact[0] for contact in native}
        ligand_positions = {(contact[1], contact[2]) for contact in native}
        selected = {key for key in self.atoms if key[2] in BACKBONE and
                    ((key[0] == rc and key[1] in receptor_positions) or key[:2] in ligand_positions)}
        keys = sorted(selected & set(common))
        row["reference_interface_backbone_atoms"] = len(selected)
        row["matched_interface_backbone_atoms"] = len(keys)
        row["interface_backbone_atom_coverage"] = len(keys) / len(selected) if selected else None
        # CA-only fixtures do not masquerade as all-backbone measurements.
        if len(keys) >= 3 and any(key[2] in ("N", "C", "O") for key in keys):
            _, _, distances = kabsch(np.array([common[key] for key in keys]),
                                     np.array([self.atoms[key] for key in keys]))
            row["interface_backbone_RMSD_angstrom"] = float(np.sqrt(np.mean(distances ** 2)))
        row["hotspots"] = self._hotspots(rc, pc, component, native, predicted, raw_mapped, chains)
        row["hotspot_contact_recall"] = _mean([item["contact_recall"] for item in row["hotspots"]])
        if component == "BMPR1A":
            row["alpha1"] = self._alpha1(rc, common, raw_mapped)
        return row

    def analyze(self, prediction_pdb, prediction_chains):
        for _, refs, predictions in GROUPS:
            for rc in refs:
                for pc in predictions:
                    chain = prediction_chains[pc]
                    if chain.sequence != self.references[rc].full_sequence:
                        raise ValueError(f"Interface sequence mismatch for prediction chain {pc}")
                    if chain.full_positions != list(range(len(chain.sequence))):
                        raise ValueError("Interface predictions must contain complete sequence-position CA mapping")
        ligand_rmsd, _, mapping, rotation, translation = self._ligand_alignment(prediction_chains)
        source_atoms = _read_atoms(prediction_pdb, prediction_chains)
        reverse = {pc: rc for rc, pc in mapping.items()}
        raw_mapped = {(reverse[c], pos, atom): xyz for (c, pos, atom), xyz in source_atoms.items()}
        common = {key: xyz for key, xyz in raw_mapped.items() if key in self.atoms}
        receptors = [self._receptor(rc, mapping[rc], component, common, raw_mapped,
                     prediction_chains[mapping[rc]].coordinates[self.references[rc].full_positions]
                     @ rotation + translation, prediction_chains)
                     for component, refs, _ in GROUPS[1:] for rc in refs]
        # Contact calculations and internal conformation fits above use raw
        # coordinates; only frame-dependent CA scores use ligand alignment.
        symmetry_errors = []
        receptor_placement_errors = []
        for component, refs, _ in GROUPS:
            first, second = refs
            positions = sorted(set(self.references[first].full_positions) & set(self.references[second].full_positions))
            a = prediction_chains[mapping[first]].coordinates[positions] @ rotation + translation
            b = prediction_chains[mapping[second]].coordinates[positions] @ rotation + translation
            symmetry_errors.extend(np.sum((a @ self.symmetry_rotation + self.symmetry_translation - b) ** 2, axis=1))
            symmetry_errors.extend(np.sum((b @ self.symmetry_rotation + self.symmetry_translation - a) ** 2, axis=1))
            if component != "BMP2":
                for rc in refs:
                    xyz = prediction_chains[mapping[rc]].coordinates[self.references[rc].full_positions] @ rotation + translation
                    receptor_placement_errors.extend(np.sum((xyz - self.references[rc].coordinates) ** 2, axis=1))
        metrics = {"BMP2_dimer_CA_RMSD_angstrom": ligand_rmsd,
                   "reference_anchored_C2_CA_deviation_angstrom": float(np.sqrt(np.mean(symmetry_errors))),
                   "reference_anchored_receptor_placement_CA_RMSD_angstrom": float(np.sqrt(np.mean(receptor_placement_errors))),
                   "reference_heavy_atom_coverage": len(common) / len(self.atoms) if self.atoms else None}
        fields = ("ligand_frame_CA_RMSD_angstrom", "centroid_displacement_angstrom", "orientation_error_degrees",
                  "native_contact_recall", "native_contact_precision", "interface_backbone_RMSD_angstrom",
                  "hotspot_contact_recall")
        for component, _, _ in GROUPS[1:]:
            rows = [row for row in receptors if row["component"] == component]
            for field in fields:
                metrics[f"{component}_{field}"] = _mean([row[field] for row in rows])
            metrics[f"{component}_worst_copy_ligand_frame_CA_RMSD_angstrom"] = max(row["ligand_frame_CA_RMSD_angstrom"] for row in rows)
            recall = [row["native_contact_recall"] for row in rows if row["native_contact_recall"] is not None]
            metrics[f"{component}_worst_copy_native_contact_recall"] = min(recall) if recall else None
            if component == "BMPR1A":
                metrics["BMPR1A_both_BMP2_monomers_contact_fraction"] = _mean([row["contacts_both_BMP2_monomers"] for row in rows])
            else:
                metrics["ACVR2A_native_knuckle_monomer_contact_fraction"] = _mean([row["native_monomer_contact_fraction"] for row in rows])
        bmpr = [row["alpha1"] for row in receptors if row["component"] == "BMPR1A"]
        metrics["BMPR1A_alpha1_helix_dihedral_fraction"] = _mean([row["helix_dihedral_fraction"] for row in bmpr])
        metrics["BMPR1A_alpha1_backbone_RMSD_angstrom"] = _mean([row["backbone_RMSD_angstrom"] for row in bmpr])
        return dict(metrics=metrics, receptors=receptors, mapping=mapping,
                    rotation=rotation.tolist(), translation=translation.tolist(),
                    matched_reference_heavy_atoms=len(common), reference_heavy_atoms=len(self.atoms))
