#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.register as R
from utils.oom_decorator import oom_decorator

from .predictor import PredictorModel
from .predictor_noisy import PredictorNNModel


@R.register('Classifier')
class Classifier(PredictorModel):
    def __init__(self, n_class: int, graph_constructor: dict, encoder: dict) -> None:
        super().__init__(encoder, graph_constructor)
        self.n_class = n_class # how many classification categories?
        # disable energy head
        for param in self.energy_ffn.parameters():
            param.requires_grad = False

        # binary classification head
        self.class_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.n_class)
        )

    @oom_decorator
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label=None):
        return_value = super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, None, return_loss=False)
        pred_class = self.class_head(return_value.graph_repr)  # [bs, n_task]
        return pred_class
    

@R.register('PairClassifier')
class PairClassifier(PredictorModel):
    def __init__(self, n_class: int, graph_constructor: dict, encoder: dict) -> None:
        super().__init__(encoder, graph_constructor)
        self.n_class = n_class # how many classification categories?
        # disable energy head
        for param in self.energy_ffn.parameters():
            param.requires_grad = False

        # binary classification head
        self.class_head = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.n_class)
        )

    @oom_decorator
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label=None):
        return_value = super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, None, return_loss=False)
        pair_graph_repr = torch.cat([return_value.graph_repr[0::2], return_value.graph_repr[1::2]], dim=-1)
        pred_class = self.class_head(pair_graph_repr)  # [bs, n_task]
        return pred_class