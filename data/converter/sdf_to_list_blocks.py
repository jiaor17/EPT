#!/usr/bin/python
# -*- coding:utf-8 -*-
from rdkit import Chem
from typing import List
from data.format import Block, Atom, VOCAB
from p_tqdm import p_map
from .rdkit_to_blocks import rdkit_to_blocks

def sdf_to_list_blocks(sdf_file: str, using_hydrogen: bool = False) -> List[List[Block]]:
    '''
        Convert an SDF file to a list of lists of blocks for each molecule in parallel.
        
        Parameters:
            sdf_file: Path to the SDF file
            using_hydrogen: Whether to preserve hydrogen atoms, default false

        Returns:
            A list of lists of blocks. Each inner list represents the blocks for a molecule.
    '''
    # Read SDF file
    supplier = Chem.SDMolSupplier(sdf_file)

    # Define function to process a single molecule
    def process_molecule(mol):
        if mol is not None:
            blocks = rdkit_to_blocks(mol, using_hydrogen=using_hydrogen)
            return blocks
        else:
            return None

    # Parallel processing of molecules
    results = p_map(process_molecule, supplier)

    # Remove None results
    results = [result for result in results if result is not None]

    # Return the final list of lists of blocks
    return results

if __name__ == '__main__':
    import sys
    list_blocks = sdf_to_list_blocks(sys.argv[1])
    print(f'{sys.argv[1]} parsed')
    print(f'number of molecules: {len(list_blocks)}')