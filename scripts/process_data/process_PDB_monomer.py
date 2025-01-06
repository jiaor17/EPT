#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import gzip
import shutil
import argparse

import numpy as np

from utils.logger import print_log
from data.format import VOCAB
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_to_data import blocks_to_data
from data.mmap_dataset import create_mmap


def parse():
    parser = argparse.ArgumentParser(description='Process PDB to monomers')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory of pdb database')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    return parser.parse_args()
    

def process_iterator(data_dir):

    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        for pdb_file in os.listdir(category_dir):
            path = os.path.join(category_dir, pdb_file)
            tmp_file = os.path.join(tmp_dir, f'{pdb_file}.decompressed')

            try:
                # uncompress the file to the tmp file
                with gzip.open(path, 'rb') as fin:
                    with open(tmp_file, 'wb') as fout:
                        shutil.copyfileobj(fin, fout)
        
                list_blocks, chains = pdb_to_list_blocks(tmp_file, return_chain_ids=True)
            except Exception as e:
                print_log(f'Parsing {pdb_file} failed: {e}', level='WARN')
                continue

            for blocks, chain in zip(list_blocks, chains):

                item_id = chain + '_' + pdb_file
                # data = blocks_to_data(blocks)
                num_blocks = len(blocks)
                num_units = sum([len(block.units) for block in blocks])

                seq = ''.join([block.symbol for block in blocks])

                data = blocks_to_data(blocks)
                for key in data:
                    if isinstance(data[key], np.ndarray):
                        data[key] = data[key].tolist()

                # id, data, properties, whether this entry is finished for producing data 
                yield item_id, data, [num_blocks, num_units, chain, seq]
            
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    shutil.rmtree(tmp_dir)

def main(args):
    
    print_log(f'Processing data from directory: {args.pdb_dir}.')
    create_mmap(
        process_iterator(args.pdb_dir),
        args.out_dir)
    
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())