#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import exp, pi, cos, log
import torch
from .abs_trainer import Trainer
import torch.nn.functional as F
import numpy as np
import pdb
import gc
import os
from tqdm import tqdm

from utils.logger import print_log

import inspect

import utils.register as R


@R.register('PreTrainer')
class PreTrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config, save_config):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.set_valid_requires_grad(True)

    def train_step(self, batch, batch_idx, fake_forward=False):
        return self.share_step(batch, batch_idx, val=False, fake_forward=fake_forward)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    def _train_epoch(self, device):
        self._before_train_epoch_start()
        if self.local_rank != -1:
            if self.train_loader.batch_sampler is not None:
                self.train_loader.batch_sampler.set_epoch(self.epoch)
            elif self.train_loader.sampler is not None:
                self.train_loader.sampler.set_epoch(self.epoch)
        t_iter = tqdm(self.train_loader) if self._is_main_proc() else self.train_loader
        for batch in t_iter:
            batch = self.to_device(batch, device)
            oom_flag = False

            try:
                loss = self.train_step(batch, self.global_step)
                update_flag = True
                if loss.isnan() or loss.isinf():
                    print_log(f"Current loss is {loss.item()}, do not update.")
                    update_flag = False
                self.optimizer.zero_grad()
                if update_flag:
                    loss.backward()
                    if self.config.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                else:
                    fake_loss = self.train_step(batch, self.global_step, fake_forward=True)
                    fake_loss.backward()  # Manually perform fake backward            
                self.optimizer.step()
            except torch.cuda.OutOfMemoryError as e:
                print(e)
                oom_flag = True

            if oom_flag:
                fake_loss = self.train_step(batch, self.global_step, fake_forward=True)
                self.optimizer.zero_grad()
                fake_loss.backward()  # Manually perform fake backward
                self.optimizer.step()


            if hasattr(t_iter, 'set_postfix'):
                t_iter.set_postfix(loss=loss.item(), version=self.version)
            self.global_step += 1
            if self.sched_freq == 'batch':
                self.scheduler.step()

            if self.global_step % 5000 == 0:
                if self.detect_nan_and_inf():
                    self.recover_ckpt(self.global_step - 5000, device)
                elif self._is_main_proc():
                    save_path = os.path.join(self.model_dir, f'step{self.global_step}.ckpt')
                    module_to_save = self.model.module if self.local_rank == 0 else self.model
                    state = {
                        "model": module_to_save.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        'cur_step': self.global_step,
                        "epoch": self.epoch
                    }
                    torch.save(state, save_path)
                    rm_path = os.path.join(self.model_dir, f'step{self.global_step - 15000}.ckpt')
                    if os.path.exists(rm_path):
                        os.remove(rm_path)
        if self.sched_freq == 'epoch':
            self.scheduler.step()

    def detect_nan_and_inf(self):
        module = self.model.module if self.local_rank >= 0 else self.model
        is_nan = torch.stack([torch.isnan(p).any() for p in module.parameters()]).any()
        is_inf = torch.stack([torch.isinf(p).any() for p in module.parameters()]).any()
        return (is_nan or is_inf)

    def recover_last_ckpt(self, device):
        cur_checkpoints = os.listdir(self.model_dir)
        max_step = 0
        for ckpt in cur_checkpoints:
            if ckpt.startswith('step'):
                step = int(ckpt[4:-5])
                if step > max_step:
                    max_step = step
        if max_step > 0:
            self.recover_ckpt(max_step, device, recover_epoch=True)

    def recover_ckpt(self, last_step, device, recover_epoch=False):
        module = self.model.module if self.local_rank >= 0 else self.model
        load_path = os.path.join(self.model_dir, f'step{last_step}.ckpt')
        if os.path.exists(load_path):
            last_state_dict = torch.load(load_path, map_location=device)
            module.load_state_dict(last_state_dict['model'])
            self.optimizer.load_state_dict(last_state_dict['optimizer'])
            self.scheduler.load_state_dict(last_state_dict['scheduler'])
            self.global_step = last_state_dict['cur_step']
            if recover_epoch:
                if "epoch" in last_state_dict:
                    self.epoch = last_state_dict['epoch']
                else:
                    self.epoch = self.scheduler.last_epoch

        else:
            print(f"Checkpoint file not found for step {last_step}")


    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False, fake_forward=False):

        if fake_forward:
            ret = self.model.module(
                Z=batch['X'], B=batch['B'], A=batch['A'],
                atom_positions=batch['atom_positions'],
                block_lengths=batch['block_lengths'],
                lengths=batch['lengths'],
                segment_ids=batch['segment_ids'],
                label=batch['label'], fake_forward=fake_forward)

            ret = self.model._post_forward(ret)

            loss = ret.loss

            return loss



        ret = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'], fake_forward=fake_forward)
        
        loss = ret.loss

        log_type = 'Validation' if val else 'Train'

        self.log(f'Loss/{log_type}', loss, batch_idx, val)

        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)

        return loss