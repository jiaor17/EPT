#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch_scatter import scatter_mean, scatter_sum

import utils.register as R
from data.format import VOCAB
from utils.nn_utils import std_conserve_scatter_sum


ReturnValue = namedtuple(
    'ReturnValue',
    ['energy', 
     'unit_repr', 'block_repr', 'graph_repr',
     'batch_id', 'block_id',
     'loss'],
    )

@R.register('PredictorModel')
class PredictorModel(nn.Module):

    def __init__(self, encoder: dict, graph_constructor: dict):
        super().__init__()

        self.encoder_config = encoder
        self.graph_config = graph_constructor
        self.hidden_size = self.encoder_config['hidden_size']

        self.global_block_id = VOCAB.symbol_to_idx(VOCAB.GLB)

        self.graph_constructor = R.construct(graph_constructor)
        self.encoder = R.construct(encoder, z_requires_grad=False)
        
        self.energy_ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def normalize(self, Z, B, block_id, batch_id):
        # centering
        center = Z[(B[block_id] == self.global_block_id)]  # [bs]
        Z = Z - center[batch_id][block_id]
        return Z

    
    def update_global_block(self, Z, B, block_id):
        is_global = B[block_id] == self.global_block_id  # [Nu]
        scatter_ids = torch.cumsum(is_global.long(), dim=0) - 1  # [Nu]
        not_global = ~is_global
        centers = scatter_mean(Z[not_global], scatter_ids[not_global], dim=0)  # [Nglobal, n_channel, 3], Nglobal = batch_size * 2
        Z = Z.clone()
        Z[is_global] = centers
        return Z, not_global
    


    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_loss=True) -> ReturnValue:
        

        graph = self.graph_constructor.forward(
            unit_type=A, unit_pos=Z, num_nodes=lengths, unit_position_ids=atom_positions,
            segment_ids=segment_ids, block_type=B, block_num_units=block_lengths
        )
        
        # normalize
        Z, B = graph.unit_pos, graph.block_type
        Z = self.normalize(Z, B, graph.unit2block, graph.batch_ids)

        Z, not_global = self.update_global_block(Z, B, graph.unit2block)

        # embedding
        H_0 = graph.unit_features
        block_id = graph.unit2block
        batch_id = graph.batch_ids
        edges = graph.edges
        edge_attr = graph.edge_attr

        not_global_edge = torch.logical_and(
            B[edges[0]] != self.global_block_id,
            B[edges[1]] != self.global_block_id
        )
        edges, edge_attr = (edges.T[not_global_edge]).T, edge_attr[not_global_edge]

        # encoding
        unit_repr, block_repr, graph_repr, pred_Z = self.encoder(H_0, Z, block_id, batch_id, edges, edge_attr)

        # predict energy
        # must be sum instead of mean! mean will make the gradient (predicted noise) pretty small, and the score net will easily converge to 0
        pred_energy = std_conserve_scatter_sum(self.energy_ffn(block_repr), batch_id, dim=0).squeeze(-1)

        if return_loss:
            
            loss = F.mse_loss(pred_energy, label)  # [Nperturb, n_channel, 3]

        else:
            loss = None

        return ReturnValue(

            # denoising variables
            energy=pred_energy,

            # representations
            unit_repr=unit_repr,
            block_repr=block_repr,
            graph_repr=graph_repr,

            # batch information
            batch_id=batch_id,
            block_id=block_id,

            # loss
            loss=loss,
        )