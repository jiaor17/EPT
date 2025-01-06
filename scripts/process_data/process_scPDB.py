#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

import numpy as np

from utils.logger import print_log
from data.converter.blocks_to_data import blocks_to_data
from data.converter.mol2_to_blocks import mol2_to_blocks
from data.mmap_dataset import create_mmap


def parse():
    parser = argparse.ArgumentParser(description='Process scPDB')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory of scPDB data')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    return parser.parse_args()
    

def process_iterator(data_dir):

    for item_id in os.listdir(data_dir):
        prot_fname = os.path.join(data_dir, item_id, 'protein.mol2')
        sm_fname = os.path.join(data_dir, item_id, 'ligand.mol2')
        
        blocks1 = mol2_to_blocks(prot_fname)
        blocks2 = mol2_to_blocks(sm_fname)

        if len(blocks1) == 0 or len(blocks2) == 0:
            continue
        
        data = blocks_to_data(blocks1, blocks2)
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()

        # id, data, [len] (only save the lengths as the properties)
        yield item_id, data, [len(data['B'])]


def main(args):
    
    cnt = 0
    for _ in os.listdir(args.data_dir):
        cnt += 1

    print_log(f'Processing data from directory: {args.data_dir}')
    create_mmap(
        process_iterator(args.data_dir),
        args.out_dir, cnt)
    
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())