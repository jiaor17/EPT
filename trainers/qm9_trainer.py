#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import exp, pi, cos, log
import torch
from .abs_trainer import Trainer
import torch.nn.functional as F
import numpy as np
import pdb
import os
from tqdm import tqdm

from utils.logger import print_log
from utils.oom_decorator import OOMReturn
from utils.metric_curve_fit import pred_metric

import utils.register as R


@R.register('QM9Trainer')
class QM9Trainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config, save_config):
        # self.global_step = 0
        # self.epoch = 0
        # self.max_step = config.max_epoch * config.step_per_epoch
        # self.log_alpha = log(config.final_lr / config.lr) / self.max_step
        # self.max_epoch = config.max_epoch
        # self.min_lr = config.final_lr
        super().__init__(model, train_loader, valid_loader, config, save_config)
        train_labels = np.array(train_loader.dataset._properties) * train_loader.dataset.unit
        self.mean = np.mean(train_labels)
        self.mad = np.mean(np.abs(train_labels - self.mean))
        self.loss_type = self.config.loss_type.upper()

        print(self.mean, self.mad)

    def loss(self, pred, label, loss_type):

        if loss_type == 'L1':
            return F.l1_loss(pred, label)
        elif loss_type in ['L2', 'MSE']:
            return F.mse_loss(pred, label)

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    def recover_last_ckpt(self, device):
        load_path = os.path.join(self.model_dir, f'last.ckpt')
        if os.path.exists(load_path):
            last_state_dict = torch.load(load_path, map_location=device)
            module.load_state_dict(last_state_dict['model'])
            self.optimizer.load_state_dict(last_state_dict['optimizer'])
            self.scheduler.load_state_dict(last_state_dict['scheduler'])
            self.global_step = last_state_dict['step']
            self.epoch = last_state_dict['epoch']
            self.topk_ckpt_map = last_state_dict['topk_ckpt_map']
            self.valid_metric_record = last_state_dict['valid_metric_record']
        else:
            print(f"Last Checkpoint File not found ...")

    def save_last_ckpt(self):
        save_path = os.path.join(self.model_dir, f'last.ckpt')
        module_to_save = self.model.module if self.local_rank == 0 else self.model
        state = {
            "model": module_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'step': self.global_step,
            'epoch': self.epoch,
            'topk_ckpt_map': self.topk_ckpt_map,
            'valid_metric_record': self.valid_metric_record,
        }
        torch.save(state, save_path)


    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        ret = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'])
        
        pred_energy = ret.energy

        if not val:
            loss = self.loss(pred_energy * self.mad + self.mean, batch['label'], self.loss_type)
        else:
            loss = self.loss(pred_energy * self.mad + self.mean, batch['label'], 'L1')

        if ret.loss is not None and not val:
            loss = loss + ret.loss

        log_type = 'Validation' if val else 'Train'

        self.log(f'Loss/{log_type}', loss, batch_idx, val, batch_size=len(batch['label']))

        if not val:
            # lr = self.config.lr if self.scheduler is None else self.scheduler.get_last_lr()
            # lr = lr[0]
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)

        return loss