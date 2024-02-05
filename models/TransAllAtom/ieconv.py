#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from models.graph_constructor import scatter_topk, variadic_meshgrid

def get_ieconv_edge_feature(X, edges, block_ids):
    '''
        X: [N, 3]
        edges: [2, E]
        block_ids: [N]
    '''

    u = torch.ones_like(X)
    u[1:] = X[1:] - X[:-1]
    u = F.normalize(u, dim=-1)
    b = torch.ones_like(X)
    b[:-1] = u[:-1] - u[1:]
    b = F.normalize(b, dim=-1)
    n = torch.ones_like(X)
    n[:-1] = torch.cross(u[:-1], u[1:])
    n = F.normalize(n, dim=-1)

    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1)

    node_out, node_in = edges
    t = X[node_out] - X[node_in]
    t = torch.einsum('ijk, ij->ik', local_frame[node_in], t)
    r = torch.sum(local_frame[node_in] * local_frame[node_out], dim=1)
    delta = torch.abs(block_ids[node_in] - block_ids[node_out]).float() / 6
    delta = delta.unsqueeze(-1)

    return torch.cat([
        t, r, delta, 
        1 - 2 * t.abs(), 1 - 2 * r.abs(), 1 - 2 * delta.abs()
    ], dim=-1)


def get_equiv_ieconv_edge_feature(X, edges, block_ids, batch_ids):
    '''
        X: [N, 3]
        edges: [2, E]
        block_ids: [N]
    '''
    batch_ids = batch_ids[block_ids]

    u = torch.ones_like(X)
    u[1:] = X[1:] - X[:-1]
    u = F.normalize(u, dim=-1)
    b = torch.ones_like(X)
    b[:-1] = u[:-1] - u[1:]
    b = F.normalize(b, dim=-1)
    n = torch.ones_like(X)
    n[:-1] = torch.cross(u[:-1], u[1:])
    n = F.normalize(n, dim=-1)

    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1)
    last_elements = torch.cumsum(scatter_add(torch.ones_like(batch_ids), batch_ids), dim=0) - 1
    first_elements = F.pad((last_elements + 1)[:-1], (1, 0), value=0)
    local_frame[last_elements] = local_frame[last_elements - 1]
    local_frame[first_elements] = local_frame[first_elements + 1]

    src, dst = edges
    t = X[src] - X[dst]
    t = torch.einsum('ijk, ij->ik', local_frame[dst], t)
    r = torch.sum(local_frame[dst] * local_frame[src], dim=1)
    delta = torch.abs(block_ids[dst] - block_ids[src]).float() / 6
    delta = delta.unsqueeze(-1)
    return torch.cat([
        t, r, delta,
        1 - 2 * t.abs(), 1 - 2 * r.abs(), 1 - 2 * delta.abs()
    ], dim=-1)


class IEConvLayer(nn.Module):
    eps = 1e-6

    def __init__(self, input_dim, hidden_dim, output_dim, edge_input_dim=1, kernel_hidden_dim=32,
                dropout=0.05, dropout_before_conv=0.2, activation="relu", aggregate_func="sum"):
        super(IEConvLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.kernel_hidden_dim = kernel_hidden_dim
        self.aggregate_func = aggregate_func

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.kernel = nn.Sequential(
            nn.Linear(edge_input_dim, kernel_hidden_dim),
            nn.ReLU(),
            nn.Linear(kernel_hidden_dim, (hidden_dim + 1) * hidden_dim)
        )
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.input_batch_norm = nn.BatchNorm1d(input_dim)
        self.message_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.update_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.output_batch_norm = nn.BatchNorm1d(output_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout_before_conv = nn.Dropout(dropout_before_conv)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def message(self, H, edges, edge_attr):
        node_in = edges[1]
        message = self.linear1(H[node_in])
        message = self.message_batch_norm(message)
        message = self.dropout_before_conv(self.activation(message))
        kernel = self.kernel(edge_attr).view(-1, self.hidden_dim + 1, self.hidden_dim)
        message = torch.einsum('ijk, ik->ij', kernel[:, 1:, :], message) + kernel[:, 0, :]
        return message
    
    def aggregate(self, H, edges, message):
        node_out, node_in = edges
        num_node = H.shape[0]
        
        if self.aggregate_func == "sum":
            update = scatter_add(message, node_out, dim=0, dim_size=num_node) 
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        return update

    def combine(self, H, update):
        output = self.linear2(update)
        return output

    def forward(self, H, edges, edge_attr):
        H = self.input_batch_norm(H)
        layer_input = self.dropout(self.activation(H))
        
        message = self.message(layer_input, edges, edge_attr)
        update = self.aggregate(H, edges, message)
        update = self.dropout(self.activation(self.update_batch_norm(update)))
        
        output = self.combine(H, update)
        output = self.output_batch_norm(output)
        return output


class EfficientIEConvLayer(IEConvLayer):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_input_dim=1, kernel_hidden_dim=32, dropout=0.05, dropout_before_conv=0.2, activation="relu", aggregate_func="sum"):
        super().__init__(input_dim, hidden_dim, output_dim, edge_input_dim, kernel_hidden_dim, dropout, dropout_before_conv, activation, aggregate_func)

        self.kernel = nn.Sequential(
            nn.Linear(edge_input_dim, kernel_hidden_dim),
            nn.ReLU(),
            nn.Linear(kernel_hidden_dim, 2 * hidden_dim)
        )

    def message(self, H, edges, edge_attr):
        node_in = edges[1]
        message = self.linear1(H[node_in])
        message = self.message_batch_norm(message)
        message = self.dropout_before_conv(self.activation(message))
        kernel = self.kernel(edge_attr).view(-1, 2, self.hidden_dim)
        message = kernel[:, 0, :] + kernel[:, 1, :] * message
        return message
