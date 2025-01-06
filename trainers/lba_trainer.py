#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import exp, pi, cos, log
import torch
from .abs_trainer import Trainer

import torch.nn.functional as F

import utils.register as R

from utils.metrics import spearman_correlation

from utils.logger import print_log

import numpy as np

import pdb

@R.register('LBATrainer')
class LBATrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config, save_config):
        
        super().__init__(model, train_loader, valid_loader, config, save_config)
        train_labels = np.array(train_loader.dataset._properties) * train_loader.dataset.unit
        self.mean = np.mean(train_labels)
        # 
        if self.config.norm_type == 'mad':
            self.mad = np.mean(np.abs(train_labels - self.mean))
        elif self.config.norm_type == 'std':
            self.mad = np.sqrt(np.mean((train_labels - self.mean) ** 2))
        else:
            self.mad = 1
        self.loss_type = self.config.loss_type.upper()

    def train_step(self, batch, batch_idx):
        ret = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'], label=batch['label'])
        
        loss = self.loss(ret.energy * self.mad + self.mean, batch['label'], loss_type = self.loss_type)

        if ret.loss is not None:
            loss = loss + ret.loss

        self.log(f'Loss/Train', loss, batch_idx, val=False)

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log('lr', lr, batch_idx, val = False)

        return loss

    def _before_train_epoch_start(self):
        self.val_pred_target = []
        return super()._before_train_epoch_start()

    def valid_step(self, batch, batch_idx):
        ret = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'], label=batch['label'])
        self.val_pred_target.append(((ret.energy * self.mad + self.mean).detach().cpu(), batch['label'].cpu()))
        loss = self.loss(ret.energy * self.mad + self.mean, batch['label'], loss_type = self.loss_type)
        self.log(f'Loss/Validation', loss, batch_idx, val=True, batch_size=len(batch['label']))
        return loss

    ########## Override end ##########
    def loss(self, pred, label, loss_type):

        if loss_type == 'L1':
            return F.l1_loss(pred, label)
        elif loss_type in ['L2', 'MSE']:
            return F.mse_loss(pred, label)

    def _aggregate_val_metric(self, metric_arr, metric_bsz):
        target = torch.cat([pred_target[1] for pred_target in self.val_pred_target], dim=0)
        pred = torch.cat([pred_target[0] for pred_target in self.val_pred_target], dim=0)
        rmse = torch.sqrt(F.mse_loss(pred, target))
        pcc = torch.corrcoef(torch.cat([pred.reshape(1,-1), target.reshape(1,-1)], dim=0))[0, 1]
        spm = spearman_correlation(pred.reshape(-1), target.reshape(-1))
        self.log('RMSE/Validation', rmse, None, val=True)
        self.log('PCC/Validation', pcc, None, val=True)
        self.log('SPCC/Validation', spm, None, val=True)
        print_log(f'Epoch{self.epoch}: RMSE: {rmse: .4f}, PCC: {pcc: .4f}, SPCC: {spm: .4f}') if self._is_main_proc() else 1
        return rmse
