#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import json
import pickle
import argparse

import numpy as np

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from utils.logger import print_log
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.mol2_to_blocks import mol2_to_blocks
from data.converter.blocks_interface import blocks_interface
from data.converter.blocks_to_data import blocks_to_data
from data.mmap_dataset import create_mmap



def parse():
    parser = argparse.ArgumentParser(description='Process PDBbind benchmark of protein-ligand interaction')
    parser.add_argument('--benchmark_dir', type=str, required=True,
                        help='Directory of the benchmark containing metadata and pdb_files')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--interface_dist_th', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()


def process_iterator(benchmark_dir, interface_dist_th):
    labels = json.load(open(os.path.join(benchmark_dir, 'metadata', 'affinities.json'), 'r'))
    for pdb_id in labels:
        pdb_dir = os.path.join(benchmark_dir, 'pdb_files')

        prot_fname = os.path.join(pdb_dir, pdb_id, pdb_id + '.pdb')
        sm_fname = os.path.join(pdb_dir, pdb_id, f'{pdb_id}_ligand.mol2')

        list_blocks1 = pdb_to_list_blocks(prot_fname)
        blocks2 = mol2_to_blocks(sm_fname)
        # try:
        #     list_blocks1 = pdb_to_list_blocks(prot_fname)
        # except Exception as e:
        #     print_log(f'{pdb_id} protein parsing failed: {e}', level='ERROR')
        #     continue
        # try:
        #     blocks2 = mol2_to_blocks(sm_fname)
        # except Exception as e:
        #     print_log(f'{pdb_id} ligand parsing failed: {e}', level='ERROR')
        #     continue
        blocks1 = []
        for b in list_blocks1:
            blocks1.extend(b)

        data = blocks_to_data(blocks1, blocks2)
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()


        # construct pockets
        blocks1, _ = blocks_interface(blocks1, blocks2, interface_dist_th)
        if len(blocks1) == 0:  # no interface (if len(interface1) == 0 then we must have len(interface2) == 0)
            print_log(f'{pdb_id} has no interface', level='ERROR')
            continue

        data_interface = blocks_to_data(blocks1, blocks2)
        for key in data_interface:
            if isinstance(data_interface[key], np.ndarray):
                data_interface[key] = data_interface[key].tolist()

        result = {
            'complex': data,
            'interface': data_interface
        }

        yield pdb_id, result, [len(data['B']), len(data_interface['B']), labels[pdb_id]]

def main(args):

    # TODO: 1. preprocess PDBbind into json summaries and complex pdbs
    labels = json.load(open(os.path.join(args.benchmark_dir, 'metadata', 'affinities.json'), 'r'))
    print_log(f'Processing data from directory: {args.benchmark_dir}.')
    create_mmap(
        process_iterator(args.benchmark_dir, args.interface_dist_th),
        args.out_dir, len(labels))
    
    print_log('Finished database construction!')

    id2line = {}
    with open(os.path.join(args.out_dir, 'index.txt'), 'r') as fin:
        for line in fin.readlines():
            _id = line.split('\t')[0]
            id2line[_id] = line

    for split in ['identity30', 'identity60', 'scaffold']:
        split_info = json.load(open(os.path.join(args.benchmark_dir, 'metadata', f'{split}_split.json'), 'r'))
        out_dir = os.path.join(args.out_dir, split)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for name in ['train', 'valid', 'test']:
            data_out_path = os.path.join(out_dir, name + '.txt')
            data_out = []
            miss_cnt = 0
            for pdb_id in split_info[name]:
                if pdb_id in id2line:
                    data_out.append(id2line[pdb_id])
                else:
                    miss_cnt += 1
            print_log(f'Obtained {len(data_out)}, missing {miss_cnt}, saving to {data_out_path}...')
            with open(data_out_path, 'w') as fout:
                fout.writelines(data_out)

    print_log('Finished splitting!')


if __name__ == '__main__':
    main(parse())
