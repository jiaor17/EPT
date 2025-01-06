#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import argparse

import numpy as np

from utils.logger import print_log
from data.converter.blocks_to_data import blocks_to_data
from data.converter.mol2_to_blocks import mol2_to_blocks
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.mmap_dataset import create_mmap


def parse():
    parser = argparse.ArgumentParser(description='Process PDBBind')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory of scPDB data')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    return parser.parse_args()


def parse_index(fpath):
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    
    data = {}
    for line in lines:
        if line.startswith('#'):
            continue
        line = re.split(r'\s+', line)
        pdb_id, resolution, year, kd = line[:4]
        data[pdb_id] = kd
    return data


def process_iterator_PP(data_dir):
    indexes = parse_index(os.path.join(data_dir, 'index', 'INDEX_general_PP.2020'))
    for pdb_id in indexes:
        list_blocks = pdb_to_list_blocks(os.path.join(data_dir, f'{pdb_id}.ent.pdb'))

        data = blocks_to_data(*list_blocks)
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()

        yield pdb_id, data, [len(data['B']), indexes[pdb_id]]


def process_iterator_PL(data_dir, index_file):
    indexes = parse_index(index_file)
    for pdb_id in indexes:
        if not os.path.exists(os.path.join(data_dir, pdb_id)):
            continue

        prot_fname = os.path.join(data_dir, pdb_id, f'{pdb_id}_protein.pdb')
        sm_fname = os.path.join(data_dir, pdb_id, f'{pdb_id}_ligand.mol2')

        list_blocks1 = pdb_to_list_blocks(prot_fname)
        blocks2 = mol2_to_blocks(sm_fname)

        data = blocks_to_data(*(list_blocks1 + [blocks2]))
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()

        yield pdb_id, data, [len(data['B']), indexes[pdb_id]]
        



def main(args):
    
    print_log(f'Processing PP')
    PP_index_file = os.path.join(args.data_dir, 'PP', 'index', 'INDEX_general_PP.2020')
    PP_index = parse_index(PP_index_file)
    create_mmap(
        process_iterator_PP(os.path.join(args.data_dir, 'PP')),
        os.path.join(args.out_dir, 'PP'), len(PP_index)
    )

    print_log(f'Processing PL refined set')
    PL_refine_index_file = os.path.join(args.data_dir, 'refined-set', 'index', 'INDEX_refined_set.2020')
    PL_refine_index = parse_index(PL_refine_index_file)
    create_mmap(
        process_iterator_PL(
            os.path.join(args.data_dir, 'refined-set'),
            PL_refine_index_file,
        ), os.path.join(args.out_dir, 'refined-set'), len(PL_refine_index)
    )

    print_log(f'Processing PL others')
    PL_other_index_file = os.path.join(args.data_dir, 'v2020-other-PL', 'index', 'INDEX_general_PL.2020')
    PL_other_index = parse_index(PL_other_index_file)
    create_mmap(
        process_iterator_PL(
            os.path.join(args.data_dir, 'v2020-other-PL'),
            PL_other_index_file,
        ), os.path.join(args.out_dir, 'v2020-other-PL'), len(PL_other_index) - len(PL_refine_index)
    )

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())
