#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np

import utils.register as R

from .mmap_dataset import MMAPDataset


@R.register('ProtSampleDataset')
class ProtSampleDataset(MMAPDataset):
    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str]=None,
            specify_index: Optional[str]=None,
            approx_length: int=1,
            name: Optional[str]=None,
            local_scope_n_block: int=3
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index, approx_length, name)
        self.local_scope_n_block = local_scope_n_block

    def __getitem__(self, idx: int):
        '''
        an example of the returned data
        {
            'X': [Natom, 3],
            'B': [Nblock],
            'A': [Natom],
            'atom_positions': [Natom],
            'block_lengths': [Nblock]
            'segment_ids': [Nblock],
        }
        '''
        item = super().__getitem__(idx)
        if len(item['B']) <= self.local_scope_n_block:
            return item
        
        start = np.random.randint(0, len(item['B']) - self.local_scope_n_block + 1)
        end = start + self.local_scope_n_block
        atom_start = 0
        for i in range(start):
            atom_start += item['block_lengths'][i]
        atom_end = atom_start
        for i in range(start, end):
            atom_end += item['block_lengths'][i]
        new_item = {
            'X': item['X'][atom_start:atom_end],
            'B': item['B'][start:end],
            'A': item['A'][atom_start:atom_end],
            'atom_positions': item['atom_positions'][atom_start:atom_end],
            'block_lengths': item['block_lengths'][start:end],
            'segment_ids': item['segment_ids'][start:end]
        }
        
        return new_item
    
    @classmethod
    def collate_fn(cls, batch):
        results = {
            'X': torch.cat([torch.tensor(item['X'], dtype=torch.float) for item in batch], dim=0),
            'B': torch.cat([torch.tensor(item['B'], dtype=torch.long) for item in batch], dim=0),
            'A': torch.cat([torch.tensor(item['A'], dtype=torch.long) for item in batch], dim=0),
            'atom_positions': torch.cat([torch.tensor(item['atom_positions'], dtype=torch.long) for item in batch], dim=0),
            'block_lengths': torch.cat([torch.tensor(item['block_lengths'], dtype=torch.long) for item in batch], dim=0),
            'segment_ids': torch.cat([torch.tensor(item['segment_ids'], dtype=torch.long) for item in batch], dim=0),
            'lengths': torch.tensor([len(item['B']) for item in batch], dtype=torch.long),
        }

        results['X'] = results['X'].unsqueeze(-2)  # number of channel is 1
        return results



if __name__ == '__main__':
    import sys
    dataset = ProtSampleDataset(sys.argv[1])
    print(dataset[0])