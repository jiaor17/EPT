#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import exp, pi, cos, log
import torch
from .abs_trainer import Trainer
import torch.nn.functional as F
import numpy as np
from torcheval.metrics.functional import binary_auroc

import utils.register as R


@R.register("MSPTrainer")
class MSPTrainer(Trainer):
    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config, save_config):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        train_labels = (
            np.array(train_loader.dataset._properties) * train_loader.dataset.unit
        )
        self.mean = np.mean(train_labels)
        self.mad = np.mean(np.abs(train_labels - self.mean))
        self.loss_type = self.config.loss_type.upper()

        print(self.mean, self.mad)

    def loss(self, pred, label):
        if self.loss_type == "L1":
            return F.l1_loss(pred, label)
        elif self.loss_type in ["L2", "MSE"]:
            return F.mse_loss(pred, label)

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        ret = self.model(
            Z_r=batch["X_r"],
            B_r=batch["B_r"],
            A_r=batch["A_r"],
            atom_positions_r=batch["atom_positions_r"],
            block_lengths_r=batch["block_lengths_r"],
            lengths_r=batch["lengths_r"],
            segment_ids_r=batch["segment_ids_r"],
            Z_l=batch["X_l"],
            B_l=batch["B_l"],
            A_l=batch["A_l"],
            atom_positions_l=batch["atom_positions_l"],
            block_lengths_l=batch["block_lengths_l"],
            lengths_l=batch["lengths_l"],
            segment_ids_l=batch["segment_ids_l"],
            label=batch["label"],
        )

        pred_class = ret

        loss = self.loss(pred_class, batch["label"])

        log_type = "Validation" if val else "Train"

        self.log(
            f"Loss/{log_type}", loss, batch_idx, val, batch_size=len(batch["label"])
        )

        if not val:
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            self.log("lr", lr, batch_idx, val)

        return loss


@R.register('MSPTrainer2')
class MSPTrainer2(Trainer):
    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config, save_config):
        super().__init__(model, train_loader, valid_loader, config, save_config)

    def train_step(self, batch, batch_idx):
        pred_class = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'])
        
        if self.is_oom_return(pred_class): # OOM
            return pred_class
        else:
            if isinstance(pred_class, tuple):
                pred_class, nn_loss = pred_class
                class_loss = F.cross_entropy(pred_class, batch['label'])
                loss = class_loss + nn_loss
                self.log('ClassLoss/Train', loss, batch_idx, val=False)
            else:
                loss = F.cross_entropy(pred_class, batch['label'])

        self.log(f'Loss/Train', loss, batch_idx, val=False)

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log('lr', lr, batch_idx, val=False)

        return loss

    def _before_train_epoch_start(self):
        self.val_pred_target = []
        return super()._before_train_epoch_start()

    def valid_step(self, batch, batch_idx):
        pred_class = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'])
        if isinstance(pred_class, tuple):
            pred_class = pred_class[0]
        self.val_pred_target.append((torch.softmax(pred_class, dim=-1).detach().cpu(), batch['label'].cpu()))
        loss = F.cross_entropy(pred_class, batch['label'])
        self.log(f'Loss/Validation', loss, batch_idx, val=True, batch_size=len(batch['label']))
        return loss
    
    def _aggregate_val_metric(self, metric_arr, metric_bsz):
        target = torch.cat([pred_target[1] for pred_target in self.val_pred_target], dim=0)
        pred = torch.cat([pred_target[0][:, 1] for pred_target in self.val_pred_target], dim=0)
        value = binary_auroc(pred, target)
        self.log('AUROC/Validation', value, None, val=True)
        return value

    ########## Override end ##########