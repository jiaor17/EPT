#!/usr/bin/python
# -*- coding:utf-8 -*-
from .mmap_dataset import MMAPDataset
from .dataset_wrapper import MixDatasetWrapper, PretrainDatasetWrapper
from .dataset_prot_sample import ProtSampleDataset


import utils.register as R

def create_dataset(config: dict):
    splits = []
    for split_name in ['train', 'valid', 'test']:
        split_config = config.get(split_name, None)
        if split_config is None:
            splits.append(None)
            continue
        if isinstance(split_config, dict) and 'datasets' in split_config:
            dataset = PretrainDatasetWrapper(split_config['max_n_vertex'],
                *[R.construct(cfg) for cfg in split_config['datasets']]
            )
        else:
            dataset = R.construct(split_config)
        splits.append(dataset)
    return splits  # train/valid/test
