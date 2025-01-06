#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

import numpy as np
import pandas as pd

from rdkit import Chem

import json
import pickle

from utils.logger import print_log
from data.converter.blocks_to_data import blocks_to_data
from data.converter.rdkit_to_blocks import rdkit_to_blocks
from data.converter.sdf_to_list_blocks import sdf_to_list_blocks
from data.mmap_dataset import create_mmap


def parse():
    parser = argparse.ArgumentParser(description='Process molecule data from GEOM dataset.')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Directory of rdkit_folder')
    parser.add_argument('--dataset', type=str, required=True,
                        help='qm9 or drugs')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--conf_per_mol', type=int, default=5,
                        help='Number of conformers to preserve, selected by boltzmannweight')
    parser.add_argument('--using_hydrogen', action='store_true',
                        help='Whether to preserve hydrogen atoms')
    parser.add_argument('--hydrogen_as_block', action='store_true',
                        help='Whether to consider hydrogen atoms as blocks')
    return parser.parse_args()

def preprocess_GEOM_dataset(base_path, dataset_name, conf_per_mol):
    """
    base_path: directory that contains GEOM dataset
    dataset_name: dataset name in [qm9, drugs]
    """
    

    # read summary file
    assert dataset_name in ['qm9', 'drugs']
    summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
    with open(summary_path, 'r') as f:
        summ = json.load(f)

    # filter valid pickle path
    pickle_path_list = [] 
    num_confs = 0
    for smiles, meta_mol in summ.items():
        u_conf = meta_mol.get('uniqueconfs')
        if u_conf is None:
            continue
        if u_conf <= 0:
            continue
        pickle_path = meta_mol.get('pickle_path')
        if pickle_path is None:
            continue
        pickle_path_list.append(pickle_path)
        num_confs += min(u_conf, conf_per_mol)

    return pickle_path_list, num_confs

def process_iterator(base_path, pickle_path_list, conf_per_mol, using_hydrogen, hydrogen_as_block):


    for i in range(len(pickle_path_list)):
        
        with open(os.path.join(base_path, pickle_path_list[i]), 'rb') as fin:
            mol = pickle.load(fin)
        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            continue
        if mol.get('uniqueconfs') <= 0:
            continue

        smiles = mol.get('smiles')

        if mol.get('uniqueconfs') <= conf_per_mol:
            # use all confs
            conf_ids = np.arange(mol.get('uniqueconfs'))
        else:
            # filter the most probable 'conf_per_mol' confs
            all_weights = np.array([_.get('boltzmannweight', -1.) for _ in mol.get('conformers')])
            descend_conf_id = (-all_weights).argsort()
            conf_ids = descend_conf_id[:conf_per_mol]

        for conf_id in conf_ids:
            conf_meta = mol.get('conformers')[conf_id]
            blocks = rdkit_to_blocks(conf_meta.get('rd_mol'), using_hydrogen, hydrogen_as_block)

            if blocks is None:
                continue
            
            data = blocks_to_data(blocks)
            for key in data:
                if isinstance(data[key], np.ndarray):
                    data[key] = data[key].tolist()

            # id, data, [len] (only save the lengths as the properties)
            yield f'{smiles}_{conf_id}', data, [len(data['B'])]

def main(args):

    print_log(f'Processing {args.dataset} dataset ...')

    pickle_path_list, num_confs = preprocess_GEOM_dataset(args.base_path, args.dataset, args.conf_per_mol)

    if not args.using_hydrogen:
        ret_name = 'woH'
    elif args.hydrogen_as_block:
        ret_name = 'blockH'
    else:
        ret_name = 'atomH'

    create_mmap(
        process_iterator(args.base_path, pickle_path_list, args.conf_per_mol, args.using_hydrogen, args.hydrogen_as_block),
        os.path.join(args.out_dir, args.dataset, ret_name),
        num_confs)
    
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())