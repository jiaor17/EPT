#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np

from .mmap_dataset import MMAPDataset, decompress
import utils.register as R

@R.register("MSPDataset")
class MSPDataset(MMAPDataset):
    def __init__(self, mmap_dir: str) -> None:
        super().__init__(mmap_dir)

        self._properties = [
            int(x[0]) for x in self._properties
        ]  # '1' for better binding, '0' for worse or equal
        self.unit = 1

    def __getitem__(self, idx: int):
        """
        an example of the returned data
        {
            'X': [Natom, 3],
            'B': [Nblock],
            'A': [Natom],
            'atom_positions': [Natom],
            'block_lengths': [Nblock]
            'segment_ids': [Nblock],
        }
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        _, start, end = self._indexes[idx]
        data: list = decompress(self._mmap[start:end])  # two blocks:
        for item in data:
            if "label" not in item:
                item["label"] = self._properties[idx]
        # if tuple cannot be accepted,change the index into idx/2
        return data[0], data[1]

    @classmethod
    def collate_fn(cls, batch):
        results = {
            "X_r": torch.cat([torch.tensor(data1["X"], dtype=torch.float).unsqueeze(-2) for data1, _ in batch], dim=0),
            "B_r": torch.cat([torch.tensor(data1["B"], dtype=torch.long) for data1, _ in batch], dim=0),
            "A_r": torch.cat([torch.tensor(data1["A"], dtype=torch.long) for data1, _ in batch], dim=0),
            "atom_positions_r": torch.cat([torch.tensor(data1["atom_positions"], dtype=torch.long) for data1, _ in batch], dim=0),
            "block_lengths_r": torch.cat([torch.tensor(data1["block_lengths"], dtype=torch.long) for data1, _ in batch], dim=0),
            "lengths_r": torch.cat([torch.tensor(len(data1["B"]), dtype=torch.long).unsqueeze(0) for data1, _ in batch], dim=0),
            "segment_ids_r": torch.cat([torch.tensor(data1["segment_ids"], dtype=torch.long) for data1, _ in batch], dim=0),
            "X_l": torch.cat([torch.tensor(data2["X"], dtype=torch.float).unsqueeze(-2) for _, data2 in batch], dim=0),
            "B_l": torch.cat([torch.tensor(data2["B"], dtype=torch.long) for _, data2 in batch], dim=0),
            "A_l": torch.cat([torch.tensor(data2["A"], dtype=torch.long) for _, data2 in batch], dim=0),
            "atom_positions_l": torch.cat([torch.tensor(data2["atom_positions"], dtype=torch.long) for _, data2 in batch], dim=0),
            "block_lengths_l": torch.cat([torch.tensor(data2["block_lengths"], dtype=torch.long) for _, data2 in batch], dim=0),
            "lengths_l": torch.cat([torch.tensor(len(data2["B"]), dtype=torch.long).unsqueeze(0) for _, data2 in batch], dim=0),
            "segment_ids_l": torch.cat([torch.tensor(data2["segment_ids"], dtype=torch.long) for _, data2 in batch], dim=0),
            "label": torch.cat([torch.tensor(data1["label"], dtype=torch.float).unsqueeze(0) for data1, _ in batch], dim=0),
        }
        return results


@R.register("MSPDataset2")
class MSPDataset2(MMAPDataset):

    def __init__(self, mmap_dir: str) -> None:
        super().__init__(mmap_dir)
        self._lengths = [int(x[0]) for x in self._properties]
        self._properties = [int(x[1]) for x in self._properties] # number of blocks in each data
        self.unit = 1
    
    def get_item_len(self, idx: int):
        return self._lengths[idx]

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
        wild, mutant = super().__getitem__(idx)
        return {
            'wild': wild,
            'mutant': mutant,
            'label': self._properties[idx]
        }
    
    @classmethod
    def collate_fn(cls, batch):
        # batch[0::2] are wild types, batch[1::2] are mutants
        results = {
            'X': torch.cat([torch.tensor(item['wild']['X'] + item['mutant']['X'], dtype=torch.float) for item in batch], dim=0),
            'B': torch.cat([torch.tensor(item['wild']['B'] + item['mutant']['B'], dtype=torch.long) for item in batch], dim=0),
            'A': torch.cat([torch.tensor(item['wild']['A'] + item['mutant']['A'], dtype=torch.long) for item in batch], dim=0),
            'atom_positions': torch.cat([torch.tensor(item['wild']['atom_positions'] + item['mutant']['atom_positions'], dtype=torch.long) for item in batch], dim=0),
            'block_lengths': torch.cat([torch.tensor(item['wild']['block_lengths'] + item['mutant']['block_lengths'], dtype=torch.long) for item in batch], dim=0),
            'segment_ids': torch.cat([torch.tensor(item['wild']['segment_ids'] + item['mutant']['segment_ids'], dtype=torch.long) for item in batch], dim=0),
            'label': torch.tensor([item['label'] for item in batch], dtype=torch.long),
        }
        lengths = []
        for item in batch:
            lengths.append(len(item['wild']['B']))
            lengths.append(len(item['mutant']['B']))
        results['lengths'] = torch.tensor(lengths, dtype=torch.long)

        results['X'] = results['X'].unsqueeze(-2)  # number of channel is 1
        return results



if __name__ == "__main__":
    import sys

    dataset = MSPDataset(sys.argv[1])
    print(dataset.__getitem__(0))
