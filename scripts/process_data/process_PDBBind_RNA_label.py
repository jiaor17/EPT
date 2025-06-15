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
from data.format import VOCAB, Atom, Block

import copy


import math

def parse():
    parser = argparse.ArgumentParser(description='Process PDBBind')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory of scPDB data')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    return parser.parse_args()


def kd_to_dg(kd, temperature=25.0):
    """Conversion of Kd to DG"""
    R = 0.0019872043
    dg_rt = math.log(kd)
    temp_in_k = temperature + 273.15
    rt = R * temp_in_k
    return dg_rt * rt

def parse_index(fpath):
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    
    data = {}
    for line in lines:
        if line.startswith('#'):
            continue
        line = re.split(r'\s+', line)
        pdb_id, resolution, year, kd = line[:4]
        # data[pdb_id] = kd

        if (not kd.startswith('Kd')) and (not kd.startswith('Ki')):  # IC50 is very different from Kd and Ki, therefore discarded
            print_log(f'{pdb_id} not measured by Kd or Ki, dropped.', level='ERROR')
            # return None
            continue
        
        if '=' not in kd:  # some data only provide a threshold, e.g. Kd<1nM, discarded
            print_log(f'{pdb_id} Kd only has threshold: {kd}', level='ERROR')
            # return None
            continue

        kd = kd.split('=')[-1].strip()
        aff, unit = float(kd[:-2]), kd[-2:]
        if unit == 'mM':
            aff *= 1e-3
        elif unit == 'nM':
            aff *= 1e-9
        elif unit == 'uM':
            aff *= 1e-6
        elif unit == 'pM':
            aff *= 1e-12
        elif unit == 'fM':
            aff *= 1e-15
        else:
            # return None   # unrecognizable unit
            continue
        
        # affinity data
        data[pdb_id] = {
            'Kd': aff,
            'dG': kd_to_dg(aff, 25.0),   # regard as measured under the standard condition
            'neglog_aff': -math.log(aff, 10)  # pK = -log_10 (Kd)
        }


    return data



def blocks_to_coords(blocks: List[Block]):
    max_n_unit = 0
    coords, masks = [], []
    for block in blocks:
        coords.append([unit.get_coord() for unit in block.units])
        max_n_unit = max(max_n_unit, len(coords[-1]))
        masks.append([1 for _ in coords[-1]])
    
    for i in range(len(coords)):
        num_pad =  max_n_unit - len(coords[i])
        coords[i] = coords[i] + [[0, 0, 0] for _ in range(num_pad)]
        masks[i] = masks[i] + [0 for _ in range(num_pad)]
    
    return np.array(coords), np.array(masks).astype('bool')  # [N, M, 3], [N, M], M == max_n_unit, in mask 0 is for padding


def dist_matrix_from_coords(coords1, masks1, coords2, masks2):
    dist = np.linalg.norm(coords1[:, None] - coords2[None, :], axis=-1)  # [N1, N2, M]
    dist = dist + np.logical_not(masks1[:, None] * masks2[None, :]) * 1e6  # [N1, N2, M]
    dist = np.min(dist, axis=-1)  # [N1, N2]
    return dist


def dist_matrix_from_residues(residue_list1, residue_list2):
    coords, mask = blocks_to_coords(residue_list1 + residue_list2)
    midpoint = len(residue_list1)
    coords1, masks1 = coords[:midpoint], mask[:midpoint]
    coords2, masks2 = coords[midpoint:], mask[midpoint:]
    return dist_matrix_from_coords(coords1, masks1, coords2, masks2)

def blocks_interface(blocks1, blocks2, dist_th):
    blocks_coord, blocks_mask = blocks_to_coords(blocks1 + blocks2)
    blocks1_coord, blocks1_mask = blocks_coord[:len(blocks1)], blocks_mask[:len(blocks1)]
    blocks2_coord, blocks2_mask = blocks_coord[len(blocks1):], blocks_mask[len(blocks1):]
    dist = dist_matrix_from_coords(blocks1_coord, blocks1_mask, blocks2_coord, blocks2_mask)
    
    on_interface = dist < dist_th
    indexes1 = np.nonzero(on_interface.sum(axis=1) > 0)[0]
    indexes2 = np.nonzero(on_interface.sum(axis=0) > 0)[0]

    blocks1 = [blocks1[i] for i in indexes1]
    blocks2 = [blocks2[i] for i in indexes2]

    return blocks1, blocks2

def break_blocks_into_atoms(blocks):

    block_list = []

    for block in blocks:
        for atom in block.units:
            atom_new = copy.deepcopy(atom)
            atom_new.pos_code = VOCAB.atom_pos_sm
            block_list.append(
                Block(symbol=atom_new.element.lower(), units = [atom_new])
            )

    return block_list


def process_iterator_PP(data_dir, index_file, dist_th = 6.0):
    indexes = parse_index(index_file)
    for pdb_id in indexes:
        list_blocks, chains = pdb_to_list_blocks(os.path.join(data_dir, f'{pdb_id}.ent.pdb'), return_chain_ids=True)

        if len(list_blocks) != 2:
            continue

        rec_residues, lig_residues = list_blocks # [Q] I can't visit FASTA. Swapping the order of the two chains may not actually change the logic of the following codes?

        rec_inter, lig_inter = blocks_interface(rec_residues, lig_residues, dist_th)

        list_blocks = [rec_inter, lig_inter]


        data = blocks_to_data(*list_blocks)
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()

        Kd, dG, neglog_aff = indexes[pdb_id]['Kd'], indexes[pdb_id]['dG'], indexes[pdb_id]['neglog_aff']

        yield pdb_id, data, [len(data['B']), Kd, dG, neglog_aff]


def process_iterator_PL(data_dir, index_file, dist_th = 6.0):
    indexes = parse_index(index_file)
    for pdb_id in indexes:
        if not os.path.exists(os.path.join(data_dir, pdb_id)):
            continue

        prot_fname = os.path.join(data_dir, pdb_id, f'{pdb_id}_protein.pdb')
        sm_fname = os.path.join(data_dir, pdb_id, f'{pdb_id}_ligand.mol2')

        list_blocks1 = pdb_to_list_blocks(prot_fname)
        blocks2 = mol2_to_blocks(sm_fname)

        blocks1 = []
        for b in list_blocks1:
            blocks1.extend(b)

        # construct pockets
        blocks1, _ = blocks_interface(blocks1, blocks2, dist_th)

        list_blocks = [blocks1, blocks2]

        data = blocks_to_data(*list_blocks)
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()


        Kd, dG, neglog_aff = indexes[pdb_id]['Kd'], indexes[pdb_id]['dG'], indexes[pdb_id]['neglog_aff']

        yield pdb_id, data, [len(data['B']), Kd, dG, neglog_aff]

def process_iterator_NL(data_dir, index_file, dist_th = 6.0):
    indexes = parse_index(index_file)
    for pdb_id in indexes:
        if not os.path.exists(os.path.join(data_dir, pdb_id)):
            continue

        list_blocks, chains = pdb_to_list_blocks(os.path.join(data_dir, f'{pdb_id}.ent.pdb'), return_chain_ids=True)

        bases = ['DA', 'DG', 'DC', 'DT', 'R-A', 'R-G', 'R-C', 'R-U']

        rec_blocks = []
        lig_blocks = []
        for chain in list_blocks:
            split_point = None
            for i in range(len(chain)):
                residue = chain[len(chain) - i - 1]
                if residue.symbol in bases:
                    split_point = len(chain) - i
                    break
            rec_blocks.extend(chain[:split_point])
            lig_blocks.extend(break_blocks_into_atoms(chain[split_point:]))

        # construct pockets
        rec_interface, _ = blocks_interface(rec_blocks, lig_blocks, dist_th)

        list_blocks = [rec_interface, lig_blocks]

        data = blocks_to_data(*list_blocks)
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()


        Kd, dG, neglog_aff = indexes[pdb_id]['Kd'], indexes[pdb_id]['dG'], indexes[pdb_id]['neglog_aff']

        yield pdb_id, data, [len(data['B']), Kd, dG, neglog_aff]

        



def main(args):
    
    print_log(f'Processing PP')
    PP_index_file = os.path.join(args.data_dir, 'PP', 'index', 'INDEX_general_PP.2020')
    PP_index = parse_index(PP_index_file)
    create_mmap(
        process_iterator_PP(os.path.join(args.data_dir, 'PP')),
        PP_index_file,
        os.path.join(args.out_dir, 'PP-aff'), len(PP_index)
    )

    print_log(f'Processing PL refined set')
    PL_refine_index_file = os.path.join(args.data_dir, 'refined-set', 'index', 'INDEX_refined_set.2020')
    PL_refine_index = parse_index(PL_refine_index_file)
    create_mmap(
        process_iterator_PL(
            os.path.join(args.data_dir, 'refined-set'),
            PL_refine_index_file,
        ), os.path.join(args.out_dir, 'refined-set-aff'), len(PL_refine_index)
    )

    print_log(f'Processing PL others')
    PL_other_index_file = os.path.join(args.data_dir, 'v2020-other-PL', 'index', 'INDEX_general_PL.2020')
    PL_other_index = parse_index(PL_other_index_file)
    create_mmap(
        process_iterator_PL(
            os.path.join(args.data_dir, 'v2020-other-PL'),
            PL_other_index_file,
        ), os.path.join(args.out_dir, 'v2020-other-PL-aff'), len(PL_other_index) - len(PL_refine_index)
    )

    print_log(f'Processing PN')
    PN_index_file = os.path.join(args.data_dir, 'PN', 'index', 'INDEX_general_PN.2020')
    PN_index = parse_index(PN_index_file)
    create_mmap(
        process_iterator_PP(os.path.join(args.data_dir, 'PN')),
        PN_index_file,
        os.path.join(args.out_dir, 'PN-aff'), len(PN_index)
    )

    print_log(f'Processing NL')
    NL_index_file = os.path.join(args.data_dir, 'NL', 'index', 'INDEX_general_NL.2020')
    NL_index = parse_index(PP_index_file)
    create_mmap(
        process_iterator_NL(os.path.join(args.data_dir, 'NL')),
        NL_index_file,
        os.path.join(args.out_dir, 'NL-aff'), len(NL_index)
    )

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())
