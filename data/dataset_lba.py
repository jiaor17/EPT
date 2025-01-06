#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np

import utils.register as R

from .mmap_dataset import MMAPDataset

@R.register('LBADataset')
class LBADataset(MMAPDataset):


    def __init__(self, mmap_dir: str) -> None:
        super().__init__(mmap_dir)

        self._properties = [float(x[0]) for x in self._properties] # number of blocks in each data
        self.unit = 1
    
    def get_item_len(self, idx: int):
        return self._properties[idx]

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
        item['label'] = [self._properties[idx] * self.unit]
        return item
    
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
            'label': torch.cat([torch.tensor(item['label'], dtype=torch.float) for item in batch], dim=0),
        }

        results['X'] = results['X'].unsqueeze(-2)  # number of channel is 1
        return results



if __name__ == '__main__':
    import sys
    dataset = LBADataset(sys.argv[1])
    print(dataset[0])