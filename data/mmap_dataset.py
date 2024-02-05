#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import io
import gzip
import json
import mmap
from tqdm import tqdm
from typing import Optional

import torch

import utils.register as R


def compress(x):
    serialized_x = json.dumps(x).encode()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as f:
        f.write(serialized_x)
    compressed = buf.getvalue()
    return compressed


def decompress(compressed_x):
    buf = io.BytesIO(compressed_x)
    with gzip.GzipFile(fileobj=buf, mode="rb") as f:
        serialized_x = f.read().decode()
    x = json.loads(serialized_x)
    return x


def _prop_to_str(properties: list):
    prop_s = []
    for prop in properties:
        if isinstance(prop, list) or isinstance(prop, dict):
            prop_s.append(json.dumps(prop))
        else:
            prop_s.append(str(prop))
    return prop_s


def create_mmap(iterator, out_dir, total_len=None, commit_batch=10000):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_file = open(os.path.join(out_dir, 'data.bin'), 'wb')
    index_file = open(os.path.join(out_dir, 'index.txt'), 'w')

    i, offset = 0, 0
    for _id, x, properties in tqdm(iterator, total=total_len):
        compressed_x = compress(x)
        bin_length = data_file.write(compressed_x)
        properties = '\t'.join(_prop_to_str(properties))
        index_file.write(f'{_id}\t{offset}\t{offset + bin_length}\t{properties}\n') # tuple of (_id, start, end), data slice is [start, end)
        offset += bin_length
        i += 1

        if i % commit_batch == 0:
            data_file.flush()  # save from memory to disk
            index_file.flush()
        
    data_file.close()
    index_file.close()


@R.register('MMAPDataset')
class MMAPDataset(torch.utils.data.Dataset):
    
    def __init__(self, mmap_dir: str, specify_data: Optional[str]=None, specify_index: Optional[str]=None, approx_length: int=1, name: Optional[str]=None) -> None:
        super().__init__()

        self._indexes = []
        self._properties = []
        _index_path = os.path.join(mmap_dir, 'index.txt') if specify_index is None else specify_index 
        with open(_index_path, 'r') as f:
            for line in f.readlines():
                messages = line.strip().split('\t')
                _id, start, end = messages[:3]
                _property = messages[3:]
                self._indexes.append((_id, int(start), int(end)))
                self._properties.append(_property)
        _data_path = os.path.join(mmap_dir, 'data.bin') if specify_data is None else specify_data
        self._data_file = open(_data_path, 'rb')
        self._mmap = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.approx_length = approx_length
        self.name = name or ""
    
    def __del__(self):
        self._mmap.close()
        self._data_file.close()

    def __len__(self):
        return len(self._indexes)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        
        _, start, end = self._indexes[idx]
        data = decompress(self._mmap[start:end])
        if 'label' not in data:
            data['label'] = 0

        return data
    
    @classmethod
    def collate_fn(cls, batch):
        keys = ['X', 'B', 'A', 'atom_positions', 'block_lengths', 'segment_ids']
        types = [torch.float, torch.long, torch.long, torch.long, torch.long, torch.long]
        res = {}
        for key, _type in zip(keys, types):
            val = []
            for item in batch:
                val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)
        res['label'] = torch.tensor([item['label'] for item in batch], dtype=torch.float)
        lengths = [len(item['B']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)
        res['X'] = res['X'].unsqueeze(-2)  # number of channel is 1
        # res['ids'] = [item['id'] for item in batch]
        return res