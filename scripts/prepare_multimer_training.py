"""Build the all-training synthetic dataset from hash-pinned refined PDBs."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
from dataclasses import asdict
import gemmi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/simplefold'))
from boltz_data_pipeline.types import ChainInfo, InterfaceInfo, Record
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from utils.mmcif_utils import parse_mmcif


def prepare(selection_path, source, output, ccd_path):
    selection = json.loads(Path(selection_path).read_text())
    for name in ['structures', 'records', 'tokens', 'cif', 'fasta', 'esm']:
        (output / name).mkdir(parents=True, exist_ok=True)
    with open(ccd_path, 'rb') as f:
        ccd = pickle.load(f)
    records, receipts = [], []
    for item in selection['records']:
        path = source / item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
        structure = gemmi.read_structure(str(path))
        structure.remove_hydrogens(); structure.remove_waters(); structure.remove_alternative_conformations()
        structure.setup_entities(); structure.assign_label_seq_id()
        # Already assembled coordinates: retain exactly these six chains, no symmetry expansion.
        for chain in structure[0]:
            for residue in chain:
                residue.subchain = chain.name
        cif = output / 'cif' / (item['id'] + '.cif')
        structure.make_mmcif_document().write_file(str(cif))
        parsed = parse_mmcif(str(cif), ccd, use_assembly=False)
        target = parsed.data
        assert len(target.chains) == 6
        sequences, entities, copies = [], {}, {}
        for chain in target.chains:
            start, count = int(chain['res_idx']), int(chain['res_num'])
            residues = target.residues[start:start+count]
            seq = ''.join(gemmi.find_tabulated_residue(str(r['name'])).one_letter_code for r in residues)
            sequences.append(seq)
            entity = entities.setdefault(seq, len(entities))
            chain['entity_id'] = entity
            chain['sym_id'] = copies.get(entity, 0)
            copies[entity] = int(chain['sym_id']) + 1
        assert sorted(copies.values()) == [2, 2, 2], copies
        assert all(target.mask)
        target.dump(output / 'structures' / (item['id'] + '.npz'))
        info = [ChainInfo(chain_id=i, chain_name=str(c['name']), mol_type=int(c['mol_type']),
                          cluster_id=-1, msa_id='', num_residues=int(c['res_num']), entity_id=int(c['entity_id']))
                for i,c in enumerate(target.chains)]
        interfaces = [InterfaceInfo(int(x['chain_1']), int(x['chain_2'])) for x in target.interfaces]
        record = Record(item['id'], parsed.info, info, interfaces)
        record.dump(output / 'records' / (item['id'] + '.json'))
        from boltz_data_pipeline.types import Input
        tokens = BoltzTokenizer().tokenize(Input(target, {}))
        assert len(tokens.tokens) == sum(len(s) for s in sequences)
        with (output / 'tokens' / (item['id'] + '.pkl')).open('wb') as f:
            pickle.dump(tokens, f)
        (output / 'fasta' / (item['id'] + '.fasta')).write_text(''.join(
            f'>{str(c["name"])}|{item["groups"][str(c["name"])]}\n{seq}\n' for c,seq in zip(target.chains,sequences)))
        records.append(asdict(record))
        receipts.append(dict(**item, residues=sum(map(len,sequences)), chains=[dict(chain=str(c['name']), sequence=seq,
            entity_id=int(c['entity_id']), sym_id=int(c['sym_id'])) for c,seq in zip(target.chains,sequences)]))
    (output / 'manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    (output / 'train.txt').write_text(''.join(r['id']+'\n' for r in records))
    (output / 'dataset-receipt.json').write_text(json.dumps(dict(selection=selection,records=receipts,
        split='all training; no held-out validation',crop='none; full six-chain assemblies'),indent=2)+'\n')
    return receipts
