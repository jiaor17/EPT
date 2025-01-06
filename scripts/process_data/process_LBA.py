#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from rdkit import Chem

import json
import pickle

from utils.logger import print_log
from data.format import Block, Atom
from data.format import VOCAB as VOCAB
from data.converter.df_to_blocks import df_to_blocks
from data.converter.blocks_to_data import blocks_to_data
from data.mmap_dataset import create_mmap

from data.atom3d_lmdb import LMDBDataset
from data.tokenizer.tokenize_3d import TOKENIZER, tokenize_3d



def parse():
    parser = argparse.ArgumentParser(description='Process molecule data from LBA dataset.')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Directory of rdkit_folder')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    return parser.parse_args()

def process_iterator(base_data):


    for item in base_data:
        
        # receptor
        blocks1 = df_to_blocks(item['atoms_pocket'], key_atom_name='name')
        
        # ligand (each block is an atom)
        blocks2 = []
        for row in item['atoms_ligand'].itertuples():
            atom = Atom(
                atom_name=getattr(row, 'name'),  # e.g. C1, C2, ..., these position code will be a unified encoding such as <sm> (small molecule) in our framework
                coordinate=[getattr(row, axis) for axis in ['x', 'y', 'z']],
                element=getattr(row, 'element'),
                pos_code=VOCAB.atom_pos_sm
            )
            blocks2.append(Block(
                symbol=atom.element.lower(),
                units=[atom]
            ))


        data = blocks_to_data(blocks1, blocks2)

        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()

        # id, data, [len] (only save the lengths as the properties)
        yield item['id'], data, [item['scores']['neglog_aff']]

def main(args):    

    for split in ['train', 'val', 'test']:

        print_log(f'Processing {split} dataset ...')

        base_data = LMDBDataset(os.path.join(args.base_path, split))

        create_mmap(
            process_iterator(base_data),
            os.path.join(args.out_dir,split),
            len(base_data)
        )
    
        print_log('Finished!')


if __name__ == '__main__':
    main(parse())