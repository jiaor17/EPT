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

from ..GET.tools import BlockEmbedding, KNNBatchEdgeConstructor
from ..graph_constructor import GraphConstructor

from utils.so3 import so3vec_to_rotation, random_normal_so3, ApproxAngularDistribution
from utils.nn_utils import stable_norm

ReturnValue = namedtuple(
    'ReturnValue',
    ['energy', 
     'unit_repr', 'block_repr', 'graph_repr',
     'batch_id', 'block_id',
     'loss'],
    )

@R.register('PredictorNNNModel')
class PredictorNNNModel(nn.Module):

    def __init__(self, encoder: dict, graph_constructor: dict, noise_type: str, agg_type: str,
                sigma_begin=0.01, sigma_end=10, n_noise_level=50,
                rot_sigma_begin=0.01, rot_sigma_end=10, rot_n_noise_level=50,
                denoise_loss_scale=1.0):
        super().__init__()

        self.noise_type = noise_type
        self.agg_type = agg_type

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

        sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), n_noise_level)), dtype=torch.float)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)  # [n_noise_level]
        rot_sigmas = torch.tensor(np.exp(np.linspace(np.log(rot_sigma_begin), np.log(rot_sigma_end), rot_n_noise_level)), dtype=torch.float)
        self.rot_sigmas = nn.Parameter(rot_sigmas, requires_grad=False)  # [n_noise_level]

        if self.noise_type == "block_rot":
            self.angular_distribution = ApproxAngularDistribution(self.rot_sigmas)

        self.denoise_loss_scale = denoise_loss_scale


    @torch.no_grad() # Enable the gradient of Z so that Z.grad can be obtained for certain tasks (e.g., MD17)
    def normalize(self, Z, B, block_id, batch_id):
        # centering
        center = Z[(B[block_id] == self.global_block_id)]  # [bs]
        Z = Z - center[batch_id][block_id]
        return Z

    
    @torch.no_grad() # Enable the gradient of Z so that Z.grad can be obtained for certain tasks (e.g., MD17)
    def update_global_block(self, Z, B, block_id):
        is_global = B[block_id] == self.global_block_id  # [Nu]
        scatter_ids = torch.cumsum(is_global.long(), dim=0) - 1  # [Nu]
        not_global = ~is_global
        centers = scatter_mean(Z[not_global], scatter_ids[not_global], dim=0)  # [Nglobal, n_channel, 3], Nglobal = batch_size * 2
        Z = Z.clone()
        Z[is_global] = centers
        return Z, not_global

    @torch.no_grad()
    def add_noise(self, Z, B, block_id, batch_id, noise_level):

        # noise_level = torch.randint(0, self.sigmas.shape[0], (batch_size,), device=Z.device)

        used_sigmas = self.sigmas[noise_level]
        used_rot_sigmas = self.rot_sigmas[noise_level]

        if self.noise_type == "atom":
            used_sigmas_atom = used_sigmas[batch_id][block_id]
            normal_noise_atom = torch.randn_like(Z)
            Z_perturbed = Z + normal_noise_atom * used_sigmas_atom.unsqueeze(-1).unsqueeze(-1)
            Z_perturbed = self.normalize(Z_perturbed, B, block_id, batch_id) # Zero CoM Subspace
            Z_perturbed, _ = self.update_global_block(Z_perturbed, B, block_id)
            return Z_perturbed, used_sigmas_atom
        
        elif self.noise_type == "block": # translation only
            used_sigmas_block = used_sigmas[batch_id]
            block_center = scatter_mean(Z, block_id, dim=0)
            normal_noise_block = torch.randn_like(block_center)
            normal_noise_atom = (normal_noise_block * used_sigmas_block.unsqueeze(-1).unsqueeze(-1))[block_id]
            Z_perturbed = Z + normal_noise_atom
            Z_perturbed = self.normalize(Z_perturbed, B, block_id, batch_id) # Zero CoM Subspace
            Z_perturbed, _ = self.update_global_block(Z_perturbed, B, block_id)
            return Z_perturbed, used_sigmas_block 

        elif self.noise_type == "block_rot": # translation & rotation
            used_sigmas_block = used_sigmas[batch_id]
            block_center = scatter_mean(Z, block_id, dim=0)
            normal_noise_block_trans = torch.randn_like(block_center)
            normal_noise_atom_trans = (normal_noise_block_trans * used_sigmas_block.unsqueeze(-1).unsqueeze(-1))[block_id]
            normal_noise_block_rot_vec, d_log_normal_noise_block_rot_vec = random_normal_so3(noise_level[batch_id], self.angular_distribution, device=self.sigmas.device)
            normal_noise_block_rot = so3vec_to_rotation(normal_noise_block_rot_vec)
            normal_noise_atom_rot = normal_noise_block_rot[block_id] # [Nu, 3, 3]
            block_center_atom = block_center[block_id]
            relative_position = Z - block_center_atom # [Nu, n_channel, 3]
            Z_perturbed = block_center_atom + relative_position @ normal_noise_atom_rot + normal_noise_atom_trans
            Z_perturbed = self.normalize(Z_perturbed, B, block_id, batch_id) # Zero CoM Subspace
            Z_perturbed, _ = self.update_global_block(Z_perturbed, B, block_id)
            # return Z_perturbed, (used_sigmas_block, d_log_normal_noise_block_rot_vec)  
            return Z_perturbed, (used_sigmas_block, - normal_noise_block_rot_vec) 

        elif self.noise_type == "none":

            return Z, None 

    @torch.no_grad()
    def get_score(self, Z, Z_perturbed, block_id, batch_id, block_lengths, ctx):

        if self.noise_type == "atom":
            used_sigmas_atom = ctx
            return (Z - Z_perturbed) / used_sigmas_atom.unsqueeze(-1).unsqueeze(-1)
        
        elif self.noise_type == "block": # translation only
            used_sigmas_block = ctx
            block_center = scatter_mean(Z, block_id, dim=0)
            block_center_perturbed = scatter_mean(Z_perturbed, block_id, dim=0)
            return (block_center - block_center_perturbed) / used_sigmas_block.unsqueeze(-1).unsqueeze(-1), block_id
        
        elif self.noise_type == "block_rot":
            used_sigmas_block, d_log_normal_noise_block_rot_vec = ctx
            block_center = scatter_mean(Z, block_id, dim=0)
            block_center_perturbed = scatter_mean(Z_perturbed, block_id, dim=0)
            score_trans = (block_center - block_center_perturbed) / used_sigmas_block.unsqueeze(-1).unsqueeze(-1)
            score_rot = d_log_normal_noise_block_rot_vec.view(-1, 1, 3)

            # Arms of Force
            relative_position = Z_perturbed - block_center_perturbed[block_id] # [Nu, n_channel, 3]

            # Inertia Matrix
            inertia_atom = stable_norm(relative_position) * torch.eye(3).unsqueeze(0).expand(relative_position.shape[0], -1, -1).to(relative_position.device) - relative_position.transpose(2,1) @ relative_position
            inertia = scatter_sum(inertia_atom, block_id, dim = 0) # [Nb, 3, 3]

            # Inverse of Inertia Matrix
            inertia_mask = block_lengths > 1
            inertia_inv = torch.eye(3).unsqueeze(0).expand(inertia.shape[0], -1, -1).to(relative_position.device)

            inertia_inv[inertia_mask] = torch.linalg.inv(inertia[inertia_mask])

            return score_trans, score_rot, relative_position, inertia_inv, inertia_mask, block_id

        elif self.noise_type == "none":

            return None
        
    def get_loss(self, pred, label_ctx):

        if self.noise_type == "atom":

            score = label_ctx
            return F.mse_loss(pred, score)
        
        elif self.noise_type == "block":

            score_trans, block_id = label_ctx
            pred_trans = scatter_mean(pred, block_id, dim=0)
            return F.mse_loss(pred_trans, score_trans)

        elif self.noise_type == "block_rot":

            score_trans, score_rot, relative_position, inertia_inv, inertia_mask, block_id = label_ctx
            pred_trans = scatter_mean(pred, block_id, dim=0)
            
            atom_moment = torch.cross(relative_position, pred, dim = -1)
            block_moment = scatter_sum(atom_moment, block_id, dim = 0) # [Nb, n_channels, 3]

            pred_rot = torch.einsum('bij, bnj -> bni', inertia_inv, block_moment) # [Nb, n_channels, 3]

            loss_trans = F.mse_loss(pred_trans, score_trans)

            loss_rot = F.mse_loss(pred_rot, score_rot, reduction='none')

            loss_rot = loss_rot[inertia_mask].mean()

            return loss_trans + loss_rot        
        
        elif self.noise_type == "none":

            return None
    


    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_loss=True) -> ReturnValue:
        

        graph = self.graph_constructor.forward(
            unit_type=A, unit_pos=Z, num_nodes=lengths, unit_position_ids=atom_positions,
            segment_ids=segment_ids, block_type=B, block_num_units=block_lengths
        )
        
        # normalize
        Z, B = graph.unit_pos, graph.block_type



        # batch_id = torch.zeros_like(segment_ids)  # [Nb]
        # batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        # batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

        # block_id = torch.zeros_like(A) # [Nu]
        # block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
        # block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

        batch_size = lengths.shape[0]
        # normalize
        Z = self.normalize(Z, B, graph.unit2block, graph.batch_ids)
        
        Z, not_global = self.update_global_block(Z, B, graph.unit2block)

        noise_level = torch.randint(0, self.sigmas.shape[0], (batch_size,), device=Z.device)


        Z_perturbed, ctx = self.add_noise(Z, B, graph.unit2block, graph.batch_ids, noise_level)

        target = self.get_score(Z, Z_perturbed, graph.unit2block, graph.batch_ids, graph.block_num_units, ctx)

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
        unit_repr, block_repr, graph_repr, _ = self.encoder(H_0, Z, block_id, batch_id, edges, edge_attr)
        # _, _, _, pred_Z = self.encoder(H_0, Z_perturbed, block_id, batch_id, edges, edge_attr)    
        # block_repr = scatter_sum(unit_repr, block_id, dim=0)
        # block_repr = F.normalize(block_repr, dim=-1)
        # pred_energy = scatter_mean(self.energy_ffn(block_repr), batch_id, dim=0).squeeze(-1)
        if self.agg_type == 'graph':
            pred_energy = self.energy_ffn(graph_repr).squeeze(-1)
        elif self.agg_type == 'block':
            pred_energy = std_conserve_scatter_sum(self.energy_ffn(block_repr), batch_id, dim=0).squeeze(-1)
        elif self.agg_type == 'atom':
            pred_energy = std_conserve_scatter_sum(self.energy_ffn(unit_repr), graph.batch_ids[graph.unit2block], dim=0).squeeze(-1)
        # pred_noise = (pred_Z - Z_perturbed).view(-1, 1, 3)
        # loss = self.get_loss(pred_noise, target)
        loss = None

        if loss is not None:
            loss = loss * self.denoise_loss_scale

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
