import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.run_bmp6_refinement import prepare_input, protocol_xml, verified_file


def fixture(tmp_path):
    lines, serial = [], 0
    for chain in 'AB':
        for position in (1, 2):
            for atom, element in [('N', 'N'), ('CA', 'C'), ('C', 'C'), ('O', 'O'), ('CB', 'C'), ('SG', 'S')]:
                serial += 1
                lines.append(f'ATOM  {serial:5d} {atom:^4s} CYS {chain}{position:4d}    '
                             f'{serial * 0.1:8.3f}{0:8.3f}{0:8.3f}{1:6.2f}{0:6.2f}          {element:>2s}\n')
        lines.append('TER   \n')
    path = tmp_path / 'original.pdb'
    path.write_text(''.join(lines) + 'END\n')
    topology = dict(construct=dict(chain_order=list('AB'), total_residues=4, source_observed_residues=2),
                    chain_masks={chain: dict(length=2, sequence_sha256=hashlib.sha256(b'CC').hexdigest(),
                                              observed_core_positions_1based=[2]) for chain in 'AB'},
                    native_disulfides=[dict(id=str(pos), left=dict(chain='A', input_position_1based=pos),
                                             right=dict(chain='B', input_position_1based=pos)) for pos in (1, 2)],
                    disulfide_count=2)
    return path, topology


def test_coordinates_topology_and_core(tmp_path):
    path, topology = fixture(tmp_path)
    out = tmp_path / 'prepared'
    result = prepare_input(path, topology, out)
    records = lambda p: [line for line in p.read_text().splitlines() if line.startswith('ATOM  ')]
    assert records(path) == records(out / 'input.pdb')
    assert (out / 'disulfides.txt').read_text() == '1 3\n2 4\n'
    assert result['core_pose_indices'] == [2, 4]
    assert result['unrestrained_backbone_residues'] == 2


def test_missing_native_sulfur_rejected(tmp_path):
    path, topology = fixture(tmp_path)
    path.write_text(''.join(line for line in path.read_text().splitlines(keepends=True)
                            if not (line.startswith('ATOM') and line[12:16].strip() == 'SG')))
    with pytest.raises(ValueError, match='disulfide endpoint'):
        prepare_input(path, topology, tmp_path / 'prepared')


def test_construct_mutation_rejected(tmp_path):
    path, topology = fixture(tmp_path)
    path.write_text(path.read_text().replace('CYS A   1', 'SER A   1'))
    with pytest.raises(ValueError, match='sequence mismatch'):
        prepare_input(path, topology, tmp_path / 'prepared')


def test_inputs_are_hash_locked(tmp_path):
    path, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='checksum mismatch'):
        verified_file(dict(path=path.name, sha256='0' * 64), tmp_path)


def test_constraint_selection_does_not_freeze_tail_repacking():
    xml = ET.fromstring(protocol_xml([2, 4], Path('/tmp/relax.script')))
    cst = xml.find('./MOVERS/AtomCoordinateCstMover')
    relax = xml.find('./MOVERS/FastRelax')
    assert cst.attrib['task_operations'] == 'core_for_constraints'
    assert relax.attrib['task_operations'] == 'repack'
    assert relax.attrib['disable_design'] == 'true'
    assert relax.attrib['ramp_down_constraints'] == 'false'
    assert relax.find('MoveMap').attrib['jump'] == 'false'
    assert xml.find('./RESIDUE_SELECTORS/Index').attrib['resnums'] == '2,4'
