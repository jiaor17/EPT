#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from utils.logger import print_log
import utils.register as R
from utils.override_parser import OverrideParser

########### Import your packages below ##########
import trainers         # execute register
import models.wrappers  # execute register
from data import create_dataset
from data.dataset_wrapper import DynamicBatchWrapper
from utils.nn_utils import count_parameters
from utils.multipack_sampler import MultipackDistributedBatchSamplerHybrid

from collections import OrderedDict

import numpy as np
import random

def parse():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--seed', type=int, default=SEED)

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    parser.add_argument('--config', type=str, required=True, help='Path to the yaml configure')
    return parser.parse_known_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


SEED = 12

def gen_state_dict(dic, prefix):
    res = OrderedDict()
    lenk = len(prefix)
    if not prefix[-1] == '.':
        lenk = lenk + 1
    for k,v in dic.items():
        if k.startswith(prefix):
            res[k[lenk:]] = v
    return res

def load_from_ckpt(model, ckpt):

    ori_model = torch.load(ckpt)
    if isinstance(ori_model, dict):
        ori_model_dict = ori_model['model']
        model.graph_constructor.load_state_dict(gen_state_dict(ori_model_dict, 'graph_constructor'), strict=False)
        model.encoder.load_state_dict(gen_state_dict(ori_model_dict, 'encoder'), strict=False)
    else:
        model.graph_constructor.load_state_dict(ori_model.graph_constructor.state_dict(), strict=False)
        model.encoder.load_state_dict(ori_model.encoder.state_dict(), strict=False)


def main(args, overrides):
    config = yaml.safe_load(open(args.config, 'r'))
    OP = OverrideParser(config, overrides)
    config = OP.parse()
    setup_seed(args.seed)
    # torch.autograd.set_detect_anomaly(True)

    model = R.construct(config['model'])
    ########### load your train / valid set ###########
    train_set, valid_set, _ = create_dataset(config['dataset'])
    if valid_set is not None:
        print_log(f'Train: {len(train_set)}, validation: {len(valid_set)}')
    else:
        print_log(f'Train: {len(train_set)}, no validation')

    dataloader_cfg = config['dataloader']
    max_n_vertex_per_gpu = dataloader_cfg.get('max_n_vertex_per_gpu', 0)
    # if max_n_vertex_per_gpu > 0:
    #     train_set = DynamicBatchWrapper(train_set, max_n_vertex_per_gpu)
    #     valid_max_n_vertex_per_gpu = dataloader_cfg.get('valid_max_n_vertex_per_gpu', max_n_vertex_per_gpu)
    #     if valid_set is not None:
    #         valid_set = DynamicBatchWrapper(valid_set, valid_max_n_vertex_per_gpu)
        # dataloader_cfg['batch_size'] = dataloader_cfg['valid_batch_size'] = 1

    ########## set your collate_fn ##########
    collate_fn = train_set.collate_fn

    ########## define your model/trainer/trainconfig #########
    batch_size = dataloader_cfg.get('batch_size', 1)
    valid_batch_size = dataloader_cfg.get('valid_batch_size', batch_size)
    shuffle = dataloader_cfg.get('shuffle', False)
    num_workers = dataloader_cfg.get('num_workers', 1)
    pretrain_ckpt = config.get('pretrain_ckpt', None)

    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=shuffle)
        if not (max_n_vertex_per_gpu > 0):
            batch_size = int(batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {batch_size}')
    else:
        args.local_rank = -1
        train_sampler = None

    ########## multi-pack wrapper #########
    if max_n_vertex_per_gpu > 0:
        train_sampler = MultipackDistributedBatchSamplerHybrid(batch_max_length=max_n_vertex_per_gpu, lengths=train_set._lengths, seed=args.seed)
        valid_max_n_vertex_per_gpu = dataloader_cfg.get('valid_max_n_vertex_per_gpu', max_n_vertex_per_gpu)
        if valid_set is not None:
            valid_sampler = MultipackDistributedBatchSamplerHybrid(batch_max_length=valid_max_n_vertex_per_gpu, lengths=valid_set._lengths, seed=args.seed)
        dataloader_cfg['batch_size'] = 1

    if args.local_rank <= 0:
        if max_n_vertex_per_gpu > 0:
            print_log(f'Dynamic batch enabled. Max number of vertex per GPU: {max_n_vertex_per_gpu}')
        if pretrain_ckpt:
            print_log(f'Loaded pretrained checkpoint from {pretrain_ckpt}')
            load_from_ckpt(model, pretrain_ckpt)
        print(model)
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')


    if max_n_vertex_per_gpu > 0:

        train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                                num_workers=num_workers,
                                collate_fn=collate_fn)

    else:
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=(shuffle and train_sampler is None),
                                sampler=train_sampler,
                                collate_fn=collate_fn)
    if valid_set is not None:
        valid_loader = DataLoader(valid_set, batch_size=valid_batch_size,
                                num_workers=num_workers,
                                collate_fn=collate_fn)
    else:
        valid_loader = None

    trainer = R.construct(
        config['trainer'],
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_config=config)
    trainer.train(args.gpus, args.local_rank)
    
    return trainer.topk_ckpt_map


if __name__ == '__main__':
    args, overrides = parse()
    main(args, overrides)