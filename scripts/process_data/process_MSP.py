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
from data.format import Block, Atom, VOCAB
from data.converter.df_to_blocks import df_to_blocks
from data.converter.blocks_to_data import blocks_to_data
from data.converter.blocks_interface import blocks_interface
from data.mmap_dataset import create_mmap

from data.atom3d_lmdb import LMDBDataset

import pdb


def parse():
    parser = argparse.ArgumentParser(
        description="Process molecule data from MSP dataset."
    )
    parser.add_argument(
        "--base_path", type=str, required=True, help="Directory of dataset"
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    return parser.parse_args()


def process_iterator(base_data):
    for item in base_data:
        chains = item['original_atoms'].chain.unique()
        blocks_original, chain_ids_original = df_to_blocks(item["original_atoms"], key_atom_name="name", return_chain_ids=True)
        assert len(blocks_original) == len(chain_ids_original)
        blocks_mutated, chain_ids_mutated = df_to_blocks(item["mutated_atoms"], key_atom_name="name", return_chain_ids=True)
        assert len(blocks_mutated) == len(chain_ids_mutated)

        assert len(blocks_original) == len(blocks_mutated)
        ori_seq = ''.join([block.symbol for block in blocks_original])
        mut_seq = ''.join([block.symbol for block in blocks_mutated])
        is_mut = [0 if a == b else 1 for a, b in zip(ori_seq, mut_seq)]
        assert sum(is_mut) == 1
        mut_pos = is_mut.index(1)
        _, select_index = blocks_interface([blocks_original[mut_pos]], blocks_original, 8.0, return_index=True)
        chain2blocks_original = { c: [] for c in chains }
        for i in select_index:
            chain2blocks_original[chain_ids_original[i]].append(blocks_original[i])
        chain2blocks_mutated = { c: [] for c in chains }
        for i in select_index:
            chain2blocks_mutated[chain_ids_mutated[i]].append(blocks_mutated[i])
        
        label = item["label"]  # '1' for better binding, '0' for worse or equal
        list_blocks_original = [chain2blocks_original[c] for c in chain2blocks_original if len(chain2blocks_original[c]) > 0]
        data1 = blocks_to_data(*list_blocks_original)
        list_blocks_mutated = [chain2blocks_mutated[c] for c in chain2blocks_mutated if len(chain2blocks_mutated[c]) > 0]
        data2 = blocks_to_data(*list_blocks_mutated)

        for key in data1:
            if isinstance(data1[key], np.ndarray):
                data1[key] = data1[key].tolist()
        for key in data2:
            if isinstance(data2[key], np.ndarray):
                data2[key] = data2[key].tolist()
        data = (data1, data2)
        # id, data, label
        yield item["id"], data, [len(data1['B']) + len(data2['B']), label]


def main(args):
    for split in ["train", "val", "test"]:
        print_log(f"Processing {split} dataset ...")

        base_data = LMDBDataset(os.path.join(args.base_path, split))

        create_mmap(
            process_iterator(base_data),
            os.path.join(args.out_dir, split),
            len(base_data),
        )

        print_log("Finished!")


if __name__ == "__main__":
    main(parse())
