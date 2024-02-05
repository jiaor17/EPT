#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List

from data.format import Block, Atom, VOCAB

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol


def rdkit_to_blocks(mol: Mol, using_hydrogen: bool = False, hydrogen_as_block: bool = False) -> List[Block]:
    '''
        Convert a single rdkit mol to a list of blocks.
        Each block contains only one atom, and the entire list represent the origin molecule.
        
        Parameters:
            rdmol: Input molecule
            using_hydrogen: Whether to preserve hydrogen atoms, default false. Note that removing H atoms may degrade performance for small mol datasets.

        Returns:
            A list of blocks. Each block contains only one atom.
    '''
    if mol.GetNumConformers() != 1:
        return None
    N = mol.GetNumAtoms()
    conf = mol.GetConformer(0)
    blocks = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        symbol = atom.GetSymbol().lower()
        pos = conf.GetAtomPosition(atom.GetIdx())
        centor = Atom(atom_name=symbol, coordinate=pos, element=symbol, pos_code=VOCAB.atom_pos_sm)
        units = [centor]
        if using_hydrogen:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    pos_h = conf.GetAtomPosition(neighbor.GetIdx())
                    at_h = Atom(atom_name='h', coordinate=pos_h, element='h', pos_code=VOCAB.atom_pos_sm)
                    if hydrogen_as_block:
                        block_h = Block(symbol='h', units=[at_h])
                        blocks.append(block_h)
                    else:
                        units.append(at_h)
        block = Block(symbol=symbol, units=units)
        blocks.append(block)


    return blocks
