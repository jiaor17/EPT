#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np

from data import create_dataset
from utils.override_parser import OverrideParser
from utils.logger import print_log
from scipy.stats import pearsonr, spearmanr

def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs(y_true - y_pred).mean()


import os


def parse():
    parser = argparse.ArgumentParser(description='inference dG')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configure')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    return parser.parse_known_args()


def get_best_ckpt(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'checkpoint', 'topk_map.txt'), 'r') as f:
        ls = f.readlines()
    ckpts = []
    for l in ls:
        k,v = l.strip().split(':')
        k = float(k)
        v = v.split('/')[-1]
        ckpts.append((k,v))

    # ckpts = sorted(ckpts, key=lambda x:x[0])
    best_ckpt = ckpts[0][1]
    return os.path.join(ckpt_dir, 'checkpoint', best_ckpt)

def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def main(args, overrides):
    config = yaml.safe_load(open(args.config, 'r'))    
    OP = OverrideParser(config, overrides)
    config = OP.parse()
    task = config['task']
    # load model
    b_ckpts = get_best_ckpt(args.ckpt)

    for idx, b_ckpt in enumerate(b_ckpts):
        ckpt_dir = os.path.split(os.path.split(b_ckpt)[0])[0]
        print(f'Using checkpoint {b_ckpt}')
        model = torch.load(b_ckpt, map_location='cpu')
        device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
        model.to(device)
        model.eval()

        # load data
        train_set, _, test_set = create_dataset(config['dataset'])
        dataloder_cfg = config.get('dataloader', {})
        batch_size = dataloder_cfg.get('batch_size', 32)
        num_workers = dataloder_cfg.get('num_workers', 4)
        norm_type = config.get('norm_type', "none")
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                num_workers=num_workers,
                                collate_fn=test_set.collate_fn)

        idx = 0

        train_labels = np.array(train_set._properties) * train_set.unit
        mean = np.mean(train_labels)
        if norm_type == 'mad':
            scale = np.mean(np.abs(train_labels - mean))
        elif norm_type == 'std':
            scale = np.sqrt(np.mean((train_labels - mean) ** 2))
        else:
            scale = 1
        post_trans = lambda x: x.energy*scale + mean

        preds, gts = [], []
        for batch in tqdm(test_loader):
            with torch.no_grad():
                # move data

                batch = to_device(batch, device)


                results = model(
                    Z=batch['X'], B=batch['B'], A=batch['A'],
                    atom_positions=batch['atom_positions'],
                    block_lengths=batch['block_lengths'],
                    lengths=batch['lengths'],
                    segment_ids=batch['segment_ids'],
                    label=batch['label'])

                results = post_trans(results)

                preds += results.detach().cpu().tolist()
                gts += batch['label'].detach().cpu().tolist()

        res_rmse = mae(gts, preds)

        print_log(f'MAE: {res_rmse: .4f}')

if __name__ == '__main__':
    args, overrides = parse()
    main(args, overrides)