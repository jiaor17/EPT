#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

import numpy as np
import pandas as pd

from rdkit import Chem

from utils.logger import print_log
from data.converter.blocks_to_data import blocks_to_data
from data.converter.rdkit_to_blocks import rdkit_to_blocks
from data.converter.sdf_to_list_blocks import sdf_to_list_blocks
from data.mmap_dataset import create_mmap


def parse():
    parser = argparse.ArgumentParser(description='Process molecule data from PCQM4Mv2 dataset.')
    parser.add_argument('--sdf_file', type=str, required=True,
                        help='Input sdf file')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--using_hydrogen', action='store_true',
                        help='Whether to preserve hydrogen atoms')
    parser.add_argument('--hydrogen_as_block', action='store_true',
                        help='Whether to consider hydrogen atoms as blocks')
    return parser.parse_args()

def process_iterator(sdf_file, using_hydrogen, hydrogen_as_block):

    # Read SDF file
    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)

    for mol in supplier:

        if mol is None:
            continue

        try:
            smiles = Chem.MolToSmiles(mol)

        except:
            continue

        blocks = rdkit_to_blocks(mol, using_hydrogen, hydrogen_as_block)

        if blocks is None:
            continue
        
        data = blocks_to_data(blocks)
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()

        # id, data, [len] (only save the lengths as the properties)
        yield smiles, data, [len(data['B'])]


def main(args):

    print_log('Processing sdf dataset ...')

    if not args.using_hydrogen:
        ret_name = 'woH'
    elif args.hydrogen_as_block:
        ret_name = 'blockH'
    else:
        ret_name = 'atomH'

    create_mmap(
        process_iterator(args.sdf_file, args.using_hydrogen, args.hydrogen_as_block),
        os.path.join(args.out_dir, ret_name))
    
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())