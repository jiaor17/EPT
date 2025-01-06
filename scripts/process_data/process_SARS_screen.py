#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

import numpy as np
import pandas as pd

from utils.logger import print_log
from data.converter.blocks_to_data import blocks_to_data
from data.converter.sdf_to_list_blocks import sdf_to_list_blocks
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_interface
from data.mmap_dataset import create_mmap


def parse():
    parser = argparse.ArgumentParser(description='Process PDBBind')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory of raw data of general set and refined set')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--interface_dist_th', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()


def parse_actives(fpath):
    
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    
    data = {}
    for line in lines[1:]: # no heads
        _id, glide_score, _, smiles = line.strip().split(',')
        data[_id] = {
            'id': _id,
            'name': _id,
            'GlideScore': glide_score,
            'smiles': smiles,
            'labels': ['active']
        }
    return data


def parse_FDA(fpath, name_file):
    id2info = {}
    with open(name_file, 'r') as fin:
        lines = fin.readlines()
    for i, line in enumerate(lines):
        line = line.strip().split('\t')
        name, smiles = line[0], line[1]
        if len(line) == 3:
            assert line[2] == 'antiviral'
            antiviral = True
        else:
            antiviral = False
        id2info[i] = (name, smiles, antiviral)
    
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    
    data = {}
    for line in lines[1:]: # no heads
        _id, glide_score, _, smiles = line.strip().split(',')
        fetch_name, fetch_smiles, antiviral = id2info[int(_id)]
        assert fetch_smiles == smiles
        data[_id] = {
            'id': _id,
            'name': fetch_name,
            'GlideScore': glide_score,
            'smiles': smiles,
            'labels': ['antiviral'] if antiviral else []
        }
    return data


def process_iterator_screen(data_dir, active_index, FDA_index, if_th):
    prot_fname = None
    for f in os.listdir(data_dir):
        if f.endswith('.pdb'):
            prot_fname = os.path.join(data_dir, f)
            break
    for dirname, indexes in zip(['actives', 'FDA'], [active_index, FDA_index]):
        sm_fname = os.path.join(data_dir, dirname, 'ligands.sdf')
        prot_list_blocks = pdb_to_list_blocks(prot_fname)
        sm_dicts = sdf_to_list_blocks(sm_fname, dict_form=True, silent=True)
        rec_blocks = []
        for blocks in prot_list_blocks:
            rec_blocks.extend(blocks)

        for name in sorted(list(sm_dicts.keys())):
            pocket_blocks, _ = blocks_interface(rec_blocks, sm_dicts[name], if_th)
            if len(pocket_blocks) == 0:
                print_log(f'{name} no interaction detected', level='WARN')
            data = blocks_to_data(pocket_blocks, sm_dicts[name])
            for key in data:
                if isinstance(data[key], np.ndarray):
                    data[key] = data[key].tolist()
            metadata = indexes[name]
            length = len(pocket_blocks) + len(sm_dicts[name])
            yield name, data, [length, metadata]


def main(args):

    print_log(f'Generating data from {args.data_dir}')
    # refined set
    active_index = parse_actives(os.path.join(args.data_dir, 'actives', 'final_result.csv'))
    FDA_index = parse_FDA(
        os.path.join(args.data_dir, 'FDA', 'final_result.csv'),
        os.path.join(args.data_dir, 'FDA', 'FDA_approv.txt')
    )

    create_mmap(
        process_iterator_screen(
            args.data_dir, active_index, FDA_index, args.interface_dist_th
        ), args.out_dir, len(active_index) + len(FDA_index)
    )

    print_log('Finished!')


if __name__ == '__main__':
    np.random.seed(12)
    main(parse())