import numpy as np
import torch

import logging
import os
import urllib

import tarfile

import pickle

from os.path import join as join
import urllib.request

from rdkit import Chem

# from data.qm9.data.prepare.process import process_xyz_files, process_xyz_gdb9
# from data.qm9.data.prepare.utils import download_data, is_int, cleanup_file
# from data.converter.rdkit_to_blocks import rdkit_to_blocks
from data.format import Block, Atom, VOCAB
from data.converter.xyz2mol import xyz2mol, __ATOM_LIST__
from data.converter.blocks_to_data import blocks_to_data

from utils.logger import print_log
from data.mmap_dataset import create_mmap

import argparse

import pdb

def parse():
    parser = argparse.ArgumentParser(description='Process molecule data from QM9 dataset.')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--using_hydrogen', action='store_true',
                        help='Whether to preserve hydrogen atoms')
    parser.add_argument('--hydrogen_as_block', action='store_true',
                        help='Whether to consider hydrogen atoms as blocks')
    parser.add_argument('--download', action='store_true',
                        help='Whether to download the dataset')    
    return parser.parse_args()

def is_int(str):
    try:
        int(str)
        return True
    except:
        return False

# Cleanup. Use try-except to avoid race condition.
def cleanup_file(file, cleanup=True):
    if cleanup:
        try:
            os.remove(file)
        except OSError:
            pass

charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}


def process_iterator(data, process_fn, file_idx_list=None):
    """
    Take a set of datafiles and apply a predefined data processing script to each
    one. Data can be stored in a directory, tarfile, or zipfile. An optional
    file extension can be added.

    Parameters
    ----------
    data : str
        Complete path to datafiles. Files must be in a directory, tarball, or zip archive.
    file_idx_list : ?????, optional
        Optionally add a file filter to check a file index is in a
        predefined list, for example, when constructing a train/valid/test split.
    """
    print_log('Processing data file: {}'.format(data))
    if tarfile.is_tarfile(data):
        tardata = tarfile.open(data, 'r')
        files = tardata.getmembers()

        readfile = lambda data_pt: tardata.extractfile(data_pt)

    elif os.is_dir(data):
        files = os.listdir(data)
        files = [os.path.join(data, file) for file in files]

        readfile = lambda data_pt: open(data_pt, 'r')

    else:
        raise ValueError('Can only read from directory or tarball archive!')


    # Use only files that match desired filter.
    files = [(idx, file) for idx, file in enumerate(files) if idx in file_idx_list]

    # Now loop over files using readfile function defined above
    # Process each file accordingly using process_file_fn


    used_props = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']

    for file in files:

        idx, f = file
        with readfile(f) as openfile:
            molecule_dict = process_fn(idx, openfile)

        yield molecule_dict['smiles'], molecule_dict['data'], [molecule_dict[pr] for pr in used_props]


def xyz_to_blocks(atoms, pos, using_hydrogen, hydrogen_as_block):
    
    pos = np.array(pos)
    p_dist = np.sqrt(np.sum((pos[None, :, :] - pos[:, None, :]) ** 2, axis = -1))
    sbs = np.array(atoms)
    h_idx = np.where(sbs == 'H')[0]
    nh_idx = np.where(sbs != 'H')[0]
    belong = nh_idx[np.argmin(p_dist[h_idx, :][:, nh_idx].reshape(len(h_idx), len(nh_idx)), axis = 1)]

    rev_dict = {j:[] for j in nh_idx}

    for i,j in enumerate(belong):
        rev_dict[j].append(h_idx[i])

    blocks = []

    for i in nh_idx:
        symbol = atoms[i].lower()
        pos_nh = pos[i]
        centor = Atom(atom_name=symbol, coordinate=pos_nh, element=symbol, pos_code=VOCAB.atom_pos_sm)
        units = [centor]
        if using_hydrogen:
            for neighbor in rev_dict[i]:
                pos_h = pos[neighbor]
                assert atoms[neighbor] == 'H'
                at_h = Atom(atom_name='h', coordinate=pos_h, element='h', pos_code=VOCAB.atom_pos_sm)
                if hydrogen_as_block:
                    block_h = Block(symbol='h', units=[at_h])
                    blocks.append(block_h)
                else:
                    units.append(at_h)
        block = Block(symbol=symbol, units=units)
        blocks.append(block)
    


    return blocks


def process_xyz_gdb9(idx, datafile, using_hydrogen, hydrogen_as_block, therm_energy_dict):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    mol_xyz = xyz_lines[2:num_atoms+2]
    mol_freq = xyz_lines[num_atoms+2]

    atoms = []
    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
        atoms.append(atom)
        atom_charges.append(charge_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])

    prop_strings = ['index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
    molecule.update(mol_props)
    # rdmol = xyz2mol(atom_charges, atom_positions, charge=0, use_graph=True, allow_charged_fragments=True, embed_chiral=True, use_huckel=False)[0]
    # blocks = rdkit_to_blocks(rdmol, using_hydrogen, hydrogen_as_block)

    blocks = xyz_to_blocks(atoms, atom_positions, using_hydrogen, hydrogen_as_block)

    
    data = blocks_to_data(blocks)
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()

    molecule.update({
        'smiles': idx,
        'data': data
    })

    molecule = add_thermo_targets(molecule, therm_energy_dict)

    return molecule



def gen_splits_gdb9(gdb9dir, cleanup=True):
    """
    Generate GDB9 training/validation/test splits used.

    First, use the file 'uncharacterized.txt' in the GDB9 figshare to find a
    list of excluded molecules.

    Second, create a list of molecule ids, and remove the excluded molecule
    indices.

    Third, assign 100k molecules to the training set, 10% to the test set,
    and the remaining to the validation set.

    Finally, generate torch.tensors which give the molecule ids for each
    set.
    """
    print_log('Splits were not specified! Automatically generating.')
    gdb9_url_excluded = 'https://springernature.figshare.com/ndownloader/files/3195404'
    gdb9_txt_excluded = join(gdb9dir, 'uncharacterized.txt')
    urllib.request.urlretrieve(gdb9_url_excluded, filename=gdb9_txt_excluded)

    # First get list of excluded indices
    excluded_strings = []
    with open(gdb9_txt_excluded) as f:
        lines = f.readlines()
        excluded_strings = [line.split()[0]
                            for line in lines if len(line.split()) > 0]

    excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

    assert len(excluded_idxs) == 3054, 'There should be exactly 3054 excluded atoms. Found {}'.format(
        len(excluded_idxs))

    # Now, create a list of indices
    Ngdb9 = 133885
    Nexcluded = 3054

    included_idxs = np.array(
        sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

    # Now generate random permutations to assign molecules to training/validation/test sets.
    Nmols = Ngdb9 - Nexcluded

    Ntrain = 110000
    Nvalid = 10000
    Ntest = Nmols - (Ntrain + Nvalid)

    # Generate random permutation
    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)

    # Now use the permutations to generate the indices of the dataset splits.
    # train, valid, test, extra = np.split(included_idxs[data_perm], [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    train, valid, test, extra = np.split(
        data_perm, [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    assert(len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
        len(train), len(valid), len(test), len(extra))

    train = included_idxs[train]
    valid = included_idxs[valid]
    test = included_idxs[test]

    splits = {'train': train, 'valid': valid, 'test': test}

    # Cleanup
    cleanup_file(gdb9_txt_excluded, cleanup)

    return splits


def get_thermo_dict(gdb9dir, cleanup=True):
    """
    Get dictionary of thermochemical energy to subtract off from
    properties of molecules.

    Probably would be easier just to just precompute this and enter it explicitly.
    """
    # Download thermochemical energy
    print_log('Downloading thermochemical energy.')
    gdb9_url_thermo = 'https://springernature.figshare.com/ndownloader/files/3195395'
    gdb9_txt_thermo = join(gdb9dir, 'atomref.txt')

    urllib.request.urlretrieve(gdb9_url_thermo, filename=gdb9_txt_thermo)

    # Loop over file of thermochemical energies
    therm_targets = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']

    # Dictionary that
    id2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    # Loop over file of thermochemical energies
    therm_energy = {target: {} for target in therm_targets}
    with open(gdb9_txt_thermo) as f:
        for line in f:
            # If line starts with an element, convert the rest to a list of energies.
            split = line.split()

            # Check charge corresponds to an atom
            if len(split) == 0 or split[0] not in id2charge.keys():
                continue

            # Loop over learning targets with defined thermochemical energy
            for therm_target, split_therm in zip(therm_targets, split[1:]):
                therm_energy[therm_target][id2charge[split[0]]
                                           ] = float(split_therm)

    # Cleanup file when finished.
    cleanup_file(gdb9_txt_thermo, cleanup)

    return therm_energy


def add_thermo_targets(data, therm_energy_dict):
    """
    Adds a new molecular property, which is the thermochemical energy.

    Parameters
    ----------
    data : ?????
        QM9 dataset split.
    therm_energy : dict
        Dictionary of thermochemical energies for relevant properties found using :get_thermo_dict:
    """

    # Now, loop over the targets with defined thermochemical energy
    for target, target_therm in therm_energy_dict.items():

        # Loop over each charge, and multiplicity of the charge
        thermo = sum([target_therm[z] for z in data['charges']])

        # Now add the thermochemical energy as a property
        data[target] = data[target] - thermo

    return data


# def download_dataset_qm9(datadir, dataname, splits=None, calculate_thermo=True, exclude=True, cleanup=True):
def main(args):
    """
    Download and prepare the QM9 (GDB9) dataset.
    """
    # Define directory for which data will be output.
    gdb9dir = args.out_dir


    # Important to avoid a race condition
    os.makedirs(gdb9dir, exist_ok=True)
    gdb9_url_data = 'https://springernature.figshare.com/ndownloader/files/3195389'
    gdb9_tar_data = join(gdb9dir, 'dsgdb9nsd.xyz.tar.bz2')
    if args.download:
        print_log(
            'Downloading and processing GDB9 dataset. Output will be in directory: {}.'.format(gdb9dir))

        print_log('Beginning download of GDB9 dataset!')
        urllib.request.urlretrieve(gdb9_url_data, filename=gdb9_tar_data)
        print_log('GDB9 dataset downloaded successfully!')

    split_file = os.path.join(gdb9dir, 'split.p')
    if os.path.exists(split_file):
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)
    # If splits are not specified, automatically generate them.
    else:
        splits = gen_splits_gdb9(gdb9dir, cleanup = True)
        with open(split_file, 'wb') as f:
            pickle.dump(splits, f)

    therm_energy = get_thermo_dict(gdb9dir, cleanup = True)

    process_fn = lambda idx, datafile: process_xyz_gdb9(idx, datafile, args.using_hydrogen, args.hydrogen_as_block, therm_energy)

    if not args.using_hydrogen:
        ret_name = 'woH'
    elif args.hydrogen_as_block:
        ret_name = 'blockH'
    else:
        ret_name = 'atomH'

    for split, split_idx in splits.items():
        
        create_mmap(
            process_iterator(gdb9_tar_data, process_fn, split_idx),
            os.path.join(args.out_dir, ret_name, split), len(split_idx))



    print_log('Processing/saving complete!')


if __name__ == '__main__':
    main(parse())