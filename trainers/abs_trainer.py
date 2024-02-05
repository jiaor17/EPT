#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
from tqdm import tqdm
from copy import deepcopy

import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.logger import print_log
from utils.oom_decorator import OOMReturn, safe_backward
from utils.metric_curve_fit import pred_metric

if torch.__version__.startswith('2.1'):
    from torch.nn.parallel import DistributedDataParallel
else:
    from .ddp_wrapper import DDPWrapper as DistributedDataParallel

########### Import your packages below ##########
from utils.ema import EMA


class TrainConfig:
    def __init__(self, save_dir, max_epoch, warmup=0,
                 metric_min_better=True, patience=3,
                 grad_clip=None, save_topk=-1,  # -1 for save all
                 grad_interval=1,  # parameter update interval
                 version=-1,
                 **kwargs):
        self.save_dir = save_dir
        self.max_epoch = max_epoch
        self.warmup = warmup
        self.metric_min_better = metric_min_better
        self.patience = patience if patience > 0 else max_epoch
        self.grad_clip = grad_clip
        self.save_topk = save_topk
        self.grad_interval = grad_interval
        self.version = version
        self.__dict__.update(kwargs)

    def add_parameter(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config: dict, save_config: dict):
        self.config = TrainConfig(**config)
        self.save_config = save_config
        self.model = model
        self.ema = self.get_ema()
        self.optimizer = self.get_optimizer()
        sched_config = self.get_scheduler(self.optimizer)
        if sched_config is None:
            sched_config = {
                'scheduler': None,
                'frequency': None
            }
        self.scheduler = sched_config['scheduler']
        self.sched_freq = sched_config['frequency']
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # distributed training
        self.local_rank = -1

        # log
        self.version = self.config.version if self.config.version >= 0 else self._get_version()
        self.config.save_dir = os.path.join(self.config.save_dir, f'version_{self.version}')
        self.model_dir = os.path.join(self.config.save_dir, 'checkpoint')
        self.writer = None  # initialize right before training
        self.writer_buffer = {}

        # training process recording
        self.valid_requires_grad = False
        self.global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.last_valid_metric = None
        self.topk_ckpt_map = []  # smaller index means better ckpt
        self.patience = self.config.patience
        self.valid_metric_record = []
        if self.config.version >= 0:
            self.load_last = True
        else:
            self.load_last = False

    @classmethod
    def to_device(cls, data, device):
        if isinstance(data, dict):
            for key in data:
                data[key] = cls.to_device(data[key], device)
        elif isinstance(data, list) or isinstance(data, tuple):
            res = [cls.to_device(item, device) for item in data]
            data = type(data)(res)
        elif hasattr(data, 'to'):
            data = data.to(device)
        return data

    def save_last_ckpt(self):
        pass

    def recover_last_ckpt(self, device):
        pass
    
    def set_valid_requires_grad(self, require=False):
        self.valid_requires_grad = require

    def _is_main_proc(self):
        return self.local_rank == 0 or self.local_rank == -1

    def _get_version(self):
        version, pattern = -1, r'version_(\d+)'
        if os.path.exists(self.config.save_dir):
            for fname in os.listdir(self.config.save_dir):
                ver = re.findall(pattern, fname)
                if len(ver):
                    version = max(int(ver[0]), version)
        return version + 1

    def is_oom_return(self, value):
        return isinstance(value, OOMReturn)

    def _pred_valid_metric(self):
        if len(self.valid_metric_record) < 3:
            return
        x = [i for i in range(len(self.valid_metric_record))]
        pred_steps = [self.config.max_epoch]
        for _ in range(4):
            pred_steps = [pred_steps[0] // 2] + pred_steps
        pred_steps = [x - 1 for x in pred_steps if x > 0]
        try:
            pred, rel_err, _ = pred_metric(x, self.valid_metric_record, pred_steps)
            rel_err = round(rel_err * 100, 2)
        except Exception:
            return
        print_log(f'Predicted validation metric: {pred} at epochs {pred_steps}, relative error {rel_err}%')

    def _before_train_epoch_start(self):
        return

    def _train_epoch(self, device):
        self._before_train_epoch_start()
        if self.local_rank != -1:
            if self.train_loader.batch_sampler is not None:
                if hasattr(self.train_loader.batch_sampler, 'set_epoch'):
                    self.train_loader.batch_sampler.set_epoch(self.epoch)
            elif self.train_loader.sampler is not None:
                if hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(self.epoch)
        # if self.train_loader.sampler is not None and self.local_rank != -1:  # distributed
        #     self.train_loader.sampler.set_epoch(self.epoch)
        t_iter = tqdm(self.train_loader) if self._is_main_proc() else self.train_loader
        for batch in t_iter:
            batch = self.to_device(batch, device)
            loss = self.train_step(batch, self.global_step)
            update_flag = True
            if self.is_oom_return(loss):
                print_log(f"Out of memory, local rank {self.local_rank}")
                loss = loss.fake_loss
            elif loss.isnan() or loss.isinf():
                print_log(f"Current loss is {loss.item()}, do not update.")
                update_flag = False
            # self.optimizer.zero_grad()
            if update_flag:
                backward_ok = safe_backward((loss / self.config.grad_interval), self.model)
                if not backward_ok:
                    print_log(f'Backward out of memory, skip', level='WARN')
                    loss = loss.detach()  # delete computing graph
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                if (self.global_step + 1) % self.config.grad_interval == 0:
                    # WARN: there might be corner case in ddp
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.ema:
                        self.ema.update()

            if hasattr(t_iter, 'set_postfix'):
                t_iter.set_postfix(loss=loss.item(), version=self.version)
            self.global_step += 1
            if self.sched_freq == 'batch':
                self.scheduler.step()
        if self.sched_freq == 'epoch':
            self.scheduler.step()
    
    def _aggregate_val_metric(self, metric_arr, metric_bsz):
        valid_metric = np.sum(metric_arr) / np.sum(metric_bsz)
        return valid_metric 

    def _valid_epoch(self, device):
        if self.valid_loader is None:
            if self._is_main_proc():
                save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
                module_to_save = self.model.module if self.local_rank == 0 else self.model
                if self.config.save_topk < 0 or (self.config.max_epoch - self.epoch <= self.config.save_topk):
                    print_log(f'No validation, save path: {save_path}')
                    torch.save(module_to_save, save_path)
                else:
                    print_log('No validation')
            return

        metric_arr = []
        metric_bsz = []
        self.model.eval()
        with torch.set_grad_enabled(self.valid_requires_grad):
            t_iter = tqdm(self.valid_loader) if self._is_main_proc() else self.valid_loader
            for batch in t_iter:
                batch = self.to_device(batch, device)
                metric = self.valid_step(batch, self.valid_global_step)
                if self.is_oom_return(metric):
                    print_log(f"Out of memory, local rank {self.local_rank}")
                    continue
                metric_arr.append(metric.cpu().item() * batch['label'].shape[0])
                metric_bsz.append(batch['label'].shape[0])
                self.valid_global_step += 1
        self.model.train()
        # judge
        valid_metric = self._aggregate_val_metric(metric_arr, metric_bsz)
        if self._is_main_proc():
            save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
            module_to_save = self.model.module if self.local_rank == 0 else self.model
            torch.save(module_to_save, save_path)
            self._maintain_topk_checkpoint(valid_metric, save_path)
            print_log(f'Validation: {valid_metric}, save path: {save_path}')
            self.valid_metric_record.append(valid_metric)
            self._pred_valid_metric()
        if self._metric_better(valid_metric):
            self.patience = self.config.patience
        else:
            self.patience -= 1
        self.last_valid_metric = valid_metric
        # Reduce on Plateau
        if self.sched_freq == 'val_epoch':
            self.scheduler.step(valid_metric)
        # write valid_metric
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            self.log(name, value, self.epoch)
        self.writer_buffer = {}
    
    def _metric_better(self, new):
        old = self.last_valid_metric
        if old is None:
            return True
        if self.config.metric_min_better:
            return new < old
        else:
            return old < new

    def _maintain_topk_checkpoint(self, valid_metric, ckpt_path):
        topk = self.config.save_topk
        if self.config.metric_min_better:
            better = lambda a, b: a < b
        else:
            better = lambda a, b: a > b
        insert_pos = len(self.topk_ckpt_map)
        for i, (metric, _) in enumerate(self.topk_ckpt_map):
            if better(valid_metric, metric):
                insert_pos = i
                break
        self.topk_ckpt_map.insert(insert_pos, (valid_metric, ckpt_path))

        # maintain topk
        if topk > 0:
            while len(self.topk_ckpt_map) > topk:
                last_ckpt_path = self.topk_ckpt_map[-1][1]
                os.remove(last_ckpt_path)
                self.topk_ckpt_map.pop()

        # save map
        topk_map_path = os.path.join(self.model_dir, 'topk_map.txt')
        with open(topk_map_path, 'w') as fout:
            for metric, path in self.topk_ckpt_map:
                fout.write(f'{metric}: {path}\n')

    def train(self, device_ids, local_rank):
        # set local rank
        self.local_rank = local_rank
        # init writer
        if self._is_main_proc():
            self.writer = SummaryWriter(self.config.save_dir)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            with open(os.path.join(self.config.save_dir, 'train_config.yaml'), 'w') as fout:
                yaml.safe_dump(self.save_config, fout)
                # json.dump(self.config.__dict__, fout)
        # main device
        main_device_id = local_rank if local_rank != -1 else device_ids[0]
        device = torch.device('cpu' if main_device_id == -1 else f'cuda:{main_device_id}')
        self.model.to(device)
        if self.ema:
            self.ema.to(device)
        if local_rank != -1:
            print_log(f'Using data parallel, local rank {local_rank}, all {device_ids}')
            # self.model = DDPWrapper( # DistributedDataParallel(
            self.model = DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )
        else:
            print_log(f'training on {device_ids}')
        if self.load_last:
            self.recover_last_ckpt(device)
        start_epoch = self.epoch
        for _ in range(start_epoch, self.config.max_epoch):
            print_log(f'epoch{self.epoch} starts') if self._is_main_proc() else 1
            self._train_epoch(device)
            print_log(f'validating ...') if self._is_main_proc() else 1
            self._valid_epoch(device)
            self.save_last_ckpt()
            self.epoch += 1
            if self.patience <= 0:
                break
            if hasattr(self.config, "max_iter") and (self.config.max_iter > 0) and (self.global_step >= self.config.max_iter):
                break

    def log(self, name, value, step, val=False, batch_size=1):
        if self._is_main_proc():
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            if val:
                if name not in self.writer_buffer:
                    self.writer_buffer[name] = []
                self.writer_buffer[name].extend([value] * batch_size)
            else:
                self.writer.add_scalar(name, value, step)

    ########## Overload these functions below ##########
    # define model wrapper
    def get_ema(self):
        ema = None
        if hasattr(self.config, 'ema'):
            ema = EMA(self.model, **self.config.ema)
        return ema 

    # define optimizer
    def get_optimizer(self):
        opt_cfg = deepcopy(self.config.optimizer)
        cls = getattr(torch.optim, opt_cfg.pop('class'))
        optimizer = cls(self.model.parameters(), **opt_cfg)
        return optimizer

    # scheduler example: linear. Return None if no scheduler is needed.
    def get_scheduler(self, optimizer):
        if not hasattr(self.config, 'scheduler'):
            return None
        sched_cfg = deepcopy(self.config.scheduler)
        cls_name = sched_cfg.pop('class')
        cls = getattr(torch.optim.lr_scheduler, cls_name)
        freq = sched_cfg.pop('frequency')
        return {
            'scheduler': cls(optimizer, **sched_cfg),
            'frequency': freq # batch/epoch/val_epoch
        }

    # train step, note that batch should be dict/list/tuple/instance. Objects with .to(device) attribute will be automatically moved to the same device as the model
    def train_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/train', loss, batch_idx)
        return loss

    # validation step
    def valid_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/validation', loss, batch_idx, val=True)
        return loss
