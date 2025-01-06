#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
import shutil
import argparse
from functools import partial

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from utils.logger import print_log
from data.converter.blocks_to_data import blocks_to_data
from data.converter.sdf_to_list_blocks import sdf_to_list_blocks
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_interface
from data.mmap_dataset import create_mmap

from .non_redundant_pdb import clustering


def parse():
    parser = argparse.ArgumentParser(description='Process PDBBind')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory of raw data of general set and refined set')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--interface_dist_th', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()


def parse_index(fpath, quality):
    pdb_dir = os.path.dirname(fpath)
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    
    data = {}
    for line in lines:
        name, status = line.strip().split('\t')
        if status == 'failed':
            continue
        metadata = json.load(open(os.path.join(pdb_dir, name, 'metadata.json'), 'r'))
        if 'pos' not in metadata['dock']:
            continue # the positive sample failed to dock
        metadata['quality'] = quality
        data[name] = (metadata, os.path.join(pdb_dir, name))
    return data



def process_iterator_PL(indexes, if_th):
    for pdb_id in indexes:
        metadata, data_dir = indexes[pdb_id]
        prot_fname = os.path.join(data_dir, 'receptor.pdb')
        sm_fname = os.path.join(data_dir, 'ligands.sdf')

        prot_list_blocks = pdb_to_list_blocks(prot_fname)
        sm_dicts = sdf_to_list_blocks(sm_fname, dict_form=True, silent=True)
        rec_blocks = []
        for blocks in prot_list_blocks:
            rec_blocks.extend(blocks)

        all_data, len_dict = {}, {}
        for name in sorted(list(sm_dicts.keys())):
            pocket_blocks, _ = blocks_interface(rec_blocks, sm_dicts[name], if_th)
            if len(pocket_blocks) == 0:
                print_log(f'{pdb_id}, {name} no interaction detected', level='WARN')
            data = blocks_to_data(pocket_blocks, sm_dicts[name])
            for key in data:
                if isinstance(data[key], np.ndarray):
                    data[key] = data[key].tolist()
            all_data[name] = data
            len_dict[name] = len(pocket_blocks) + len(sm_dicts[name])

        yield pdb_id, all_data, [len_dict, metadata]


def _extract_id_to_seqs(lines, filter_refined=False):
    id2seqs = {}
    for line in lines:
        line = line.strip().split('\t')
        prop = json.loads(line[-1])
        if prop['quality'] == 'general' and filter_refined:
            continue
        rec_seq = 'X'.join(prop['receptor_seqs'])
        id2seqs[line[0]] = (rec_seq, prop['dock']['pos']['smiles'])
    return id2seqs


def _cluster_seq_id(id2seqs, seq_id):
    # make temporary directory
    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    else:
        raise ValueError(f'Working directory {tmp_dir} exists!')
    
    # 1. get non-redundant dimer by 90% seq-id
    fasta = os.path.join(tmp_dir, 'seq.fasta')
    # receptor
    with open(fasta, 'w') as fout:
        for _id in id2seqs:
            fout.write(f'>{_id}\n{id2seqs[_id][0]}\n')
    id2clu, clu2id = clustering(fasta, tmp_dir, seq_id)
    shutil.rmtree(tmp_dir)
    return id2clu, clu2id


def _generate_scaffold(smiles, include_chirality=False):
    """return scaffold string of target molecule"""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold\
        .MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def _cluster_scaffold(id2seqs):
    id2clu, clu2id = {}, {}
    for _id in id2seqs:
        smi = id2seqs[_id][1]
        scaffold = _generate_scaffold(smi, include_chirality=True)
        id2clu[_id] = scaffold
        if scaffold not in clu2id:
            clu2id[scaffold] = []
        clu2id[scaffold].append(_id)
    return id2clu, clu2id


def create_test_set(mmap_dir, max_size=500):
    index_file = os.path.join(mmap_dir, 'index.txt')
    with open(index_file, 'r') as fin:
        lines = fin.readlines()
    id2lines = { line.split('\t')[0]: line for line in lines }
    id2seqs = _extract_id_to_seqs(lines, filter_refined=True) # only use high-quality data for testing
    max_cluster_size = 5 # use 

    # delete similar recptors (above 30% sequence identity)
    id2clu_seq, clu2id_seq = _cluster_seq_id(id2seqs, 0.3)

    # delete similar scaffolds
    id2clu_scaffold, clu2id_scaffold = _cluster_scaffold(id2seqs)
    test_ids = []
    for _id in id2seqs:
        clu_size1 = len(clu2id_seq[id2clu_seq[_id]])
        clu_size2 = len(clu2id_scaffold[id2clu_scaffold[_id]])
        if clu_size1 < max_cluster_size and clu_size2 < max_cluster_size:
            test_ids.append(_id)
    
    np.random.shuffle(test_ids)
    test_ids = test_ids[:max_size]

    # write results
    with open(os.path.join(mmap_dir, 'test.txt'), 'w') as fout:
        for _id in test_ids:
            fout.write(id2lines[_id])
    
    return test_ids


def _split_by_cluster(id2clu, clu2id, test_ids, train_ratio):
    test_clus = { id2clu[_id]: True for _id in test_ids }
    available_clus = sorted([ c for c in clu2id if c not in test_clus ])
    train_size = int(len(available_clus) * train_ratio)
    train_ids, valid_ids = [], []
    np.random.shuffle(available_clus)
    for c in available_clus[:train_size]:
        train_ids.extend(clu2id[c])
    for c in available_clus[train_size:]:
        valid_ids.extend(clu2id[c])
    return train_ids, valid_ids, available_clus[:train_size], available_clus[train_size:]


def split_by_func(mmap_dir, test_ids, name, func, train_ratio=0.9):
    index_file = os.path.join(mmap_dir, 'index.txt')
    with open(index_file, 'r') as fin:
        lines = fin.readlines()
    id2lines = { line.split('\t')[0]: line for line in lines }
    id2seqs = _extract_id_to_seqs(lines, filter_refined=False)

    # cluster
    id2clu, clu2id = func(id2seqs)

    # split
    train_ids, valid_ids, train_clus, valid_clus = _split_by_cluster(id2clu, clu2id, test_ids, train_ratio)
    print_log(f'Train set: {len(train_ids)} entries, {len(train_clus)} clusters')
    print_log(f'Validation set: {len(valid_ids)} entries, {len(valid_clus)} clusters')

    # write results
    out_dir = os.path.join(mmap_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'train.txt'), 'w') as fout:
        for _id in train_ids: fout.write(id2lines[_id])
    with open(os.path.join(out_dir, 'train_cluster.txt'), 'w') as fout:
        for _id in train_ids: fout.write(f'{_id}\t{id2clu[_id]}\n')
    
    with open(os.path.join(out_dir, 'valid.txt'), 'w') as fout:
        for _id in valid_ids: fout.write(id2lines[_id])
    with open(os.path.join(out_dir, 'valid_cluster.txt'), 'w') as fout:
        for _id in valid_ids: fout.write(f'{_id}\t{id2clu[_id]}\n')



def main(args):

    if not os.path.exists(args.out_dir):
        print_log(f'Generating data from {args.data_dir}')
        # refined set
        indexes = parse_index(os.path.join(args.data_dir, 'processed_refined_set', 'done.log'), 'refined')
        indexes2 = parse_index(os.path.join(args.data_dir, 'processed_general_set', 'done.log'), 'general')
        for name in indexes2:
            assert name not in indexes, name
            indexes[name] = indexes2[name]

        create_mmap(
            process_iterator_PL(
                indexes, args.interface_dist_th
            ), args.out_dir, len(indexes)
        )

    # create splits
    non_redundant_test = create_test_set(args.out_dir)
    print_log(f'Size of non-redundant test set: {len(non_redundant_test)}')
    split_funcs = {
        'seqid_30': partial(_cluster_seq_id, seq_id=0.3),
        'seqid_60': partial(_cluster_seq_id, seq_id=0.6),
        'scaffold': _cluster_scaffold,
    }

    for name in split_funcs:
        print()
        print_log(f'Processing split {name}...')
        split_by_func(args.out_dir, non_redundant_test, name, split_funcs[name])

    print_log('Finished!')


if __name__ == '__main__':
    np.random.seed(12)
    main(parse())