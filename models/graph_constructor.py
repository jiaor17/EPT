#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Dict, Tuple, List
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_sum

import utils.register as R
from data.format import VOCAB

from .utils import _block_edge_dist


@dataclass
class Graph:

    # before processing
    unit_type: Tensor               # [Nu], torch.long
    unit_pos: Tensor                # [Nu, n_channel, 3], torch.float
    num_nodes: Tensor               # [batch_size], torch.long
    unit_position_ids: Tensor=None  # [Nu], torch.long
    segment_ids: Tensor=None        # [Nb], torch.long
    block_type: Tensor=None         # [Nb], torch.long
    block_position_ids: Tensor=None # [Nb], torch.long
    block_num_units: Tensor=None    # [Nb], torch.long (convertable with unit2block)

    # after processing
    unit_features: Tensor=None      # [Nu, num_features], torch.float
    block_features: Tensor=None     # [Nb, num_features], torch.float
    batch_ids: Tensor=None          # [Nb], torch.long
    unit2block: Tensor=None         # [Nu], torch.long (convertable with block_num_units)
    batch_size: int=None            # int
    edges: Tensor=None              # [2, Eb], torch.long
    edge_attr: Tensor=None          # [Eb], torch.long


def _instantiate(config: Dict):
    config = deepcopy(config)
    cls = globals()[config.pop('class')]
    return cls(**config)


@R.register('GraphConstructor')
class GraphConstructor(nn.Module):

    def __init__(self, node_layers: List[Dict], edge_layers: List[Dict], pre_layers:List[Dict]=[], edge_embed_size: int=0) -> None:
        super().__init__()

        self.pre_modules = nn.ModuleList([
            _instantiate(pre_layer) for pre_layer in pre_layers
        ])

        self.node_modules = nn.ModuleList([
            _instantiate(node_layer) for node_layer in node_layers
        ])
        self.edge_modules = nn.ModuleList([
            _instantiate(edge_layer) for edge_layer in edge_layers
        ])
        self.edge_embed_size = edge_embed_size
        if self.edge_embed_size > 0:
            self.edge_embed = nn.Embedding(self.num_edge_type, self.edge_embed_size)

    @property
    def num_edge_type(self):
        cnt = 0
        for layer in self.edge_modules:
            cnt += layer.num_edge_type
        return cnt

    @torch.no_grad()
    def prepare(self, graph: Graph) -> Graph:
        if graph.segment_ids is None:
            graph.segment_ids = torch.zeros_like(
                graph.unit_type if graph.block_type is None else graph.block_type)
        
        if graph.block_num_units is None:  # assume each unit is a block (i.e. each unit is a single node)
            graph.block_num_units = torch.ones_like(graph.unit_type)

        # construct batch id
        batch_ids = torch.zeros_like(graph.segment_ids)  # [Nb]
        batch_ids[torch.cumsum(graph.num_nodes, dim=0)[:-1]] = 1
        batch_ids.cumsum_(dim=0)  # [Nb], item idx in the batch
        graph.batch_ids = batch_ids

        unit2block = torch.zeros_like(graph.unit_type) # [Nu]
        unit2block[torch.cumsum(graph.block_num_units, dim=0)[:-1]] = 1
        unit2block.cumsum_(dim=0)  # [Nu], block (e.g. residue) ids of each unit (atom)
        graph.unit2block = unit2block

        graph.batch_size = graph.num_nodes.shape[0]

        return graph

    def forward(self, unit_type, unit_pos, num_nodes, unit_position_ids=None,
                segment_ids=None, block_type=None, block_position_ids=None, block_num_units=None) -> Graph:
        
        graph = Graph(
            unit_type=unit_type,
            unit_pos=unit_pos,
            num_nodes=num_nodes,
            unit_position_ids=unit_position_ids,
            segment_ids=segment_ids,
            block_type=block_type,
            block_position_ids=block_position_ids,
            block_num_units=block_num_units
        )

        graph = self.prepare(graph)

        # 0. preprocess
        with torch.no_grad():
            for pre_layer in self.pre_modules:
                graph = pre_layer(graph)

        # 1. node layers (embeddings, normalization etc.)
        for node_layer in self.node_modules:
            graph = node_layer(graph)
        if graph.unit_features is None:
            graph.unit_features = graph.block_features  # block-level nodes

        # 2. edge layers
        with torch.no_grad():
            all_edges = variadic_meshgrid(
                input1=torch.arange(graph.batch_ids.shape[0], device=graph.batch_ids.device),
                size1=graph.num_nodes,
                input2=torch.arange(graph.batch_ids.shape[0], device=graph.batch_ids.device),
                size2=graph.num_nodes
            ) # (row, col)

            edges, edge_attr, type_offset = [], [], 0
            for edge_layer in self.edge_modules:
                row, col, edge_type = edge_layer(graph, all_edges)
                edges.append(torch.stack([row, col], dim=0))
                edge_attr.append(edge_type + type_offset)
                type_offset += edge_layer.num_edge_type
            graph.edges = torch.cat(edges, dim=-1)
            graph.edge_attr = torch.cat(edge_attr, dim=0)

        # 3. edge embedding
        if self.edge_embed_size > 0:
            graph.edge_attr = self.edge_embed(graph.edge_attr)

        return graph

"""
Pre-layers:
changing unit_type, unit_pos, num_nodes, unit_position_ids,
         segment_ids, block_type, block_position_ids, block_num_units,
         batch_ids, unit2block
"""
class AlphaCarbonOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.carbon_id = VOCAB.atom_to_idx('C')
        self.alpha_id = VOCAB.atom_pos_to_idx('A')
        self.glb_id = VOCAB.get_atom_global_idx()

    def forward(self, graph: Graph) -> Graph:
        unit_mask = (graph.unit_type == self.carbon_id) & (graph.unit_position_ids == self.alpha_id)
        unit_mask = torch.logical_or(unit_mask, graph.unit_type == self.glb_id)

        # unit level
        graph.unit_type = graph.unit_type[unit_mask]
        graph.unit_pos = graph.unit_pos[unit_mask]
        graph.unit_position_ids = graph.unit_position_ids[unit_mask]

        # block level
        maintain_blocks = graph.unit2block[unit_mask]
        graph.segment_ids = graph.segment_ids[maintain_blocks]
        graph.block_type = graph.block_type[maintain_blocks]
        if graph.block_position_ids is not None:
            graph.block_position_ids = graph.block_position_ids[maintain_blocks]
        graph.batch_ids = graph.batch_ids[maintain_blocks]

        # others
        graph.unit2block = torch.arange(0, graph.block_type.shape[0], device=graph.block_type.device)
        graph.num_nodes = scatter_sum(torch.ones_like(graph.block_type), graph.batch_ids, dim=0)
        graph.block_num_units = torch.ones_like(graph.block_type)

        return graph


"""
Node layers
"""
def _check_level(level: str):
    assert level in ['unit', 'block'], f'Unknown level: {level}'


class OneHotEmbedding(nn.Module):
    def __init__(self, num_classes: int, level: str='unit') -> None:
        super().__init__()
        self.num_classes = num_classes
        self.level = level
        _check_level(level)

    def forward(self, graph: Graph) -> Graph:
        _type = getattr(graph, f'{self.level}_type')
        embed = F.one_hot(_type, self.num_classes).float()
        setattr(graph, f'{self.level}_features', embed)
        return graph


class ContinuousEmbedding(nn.Module):
    def __init__(self, num_classes: int, embed_size: int, level: str='unit') -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.level = level
        self.embedding = nn.Embedding(self.num_classes, self.embed_size)
        _check_level(level)

    def forward(self, graph: Graph) -> Graph:
        _type = getattr(graph, f'{self.level}_type')
        embed = self.embedding(_type)
        setattr(graph, f'{self.level}_features', embed)
        return graph

class DummyEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph: Graph) -> Graph:
        return graph
    

class PositionEncoding(nn.Module):
    def __init__(self, max_position: int, embed_size: int, merge: str='add', level: str='unit', omit_mol: bool=False) -> None:
        super().__init__()
        self.max_position = max_position
        self.embed_size = embed_size
        self.merge = merge
        self.level = level
        self.embedding = nn.Embedding(self.max_position, self.embed_size)
        self._omit_mol = omit_mol
        self.mol_pos_id = VOCAB.atom_pos_to_idx(VOCAB.atom_pos_sm)

    @property
    def should_omit_mol(self):
        if hasattr(self, 'omit_mol'):
            return self.omit_mol
        elif hasattr(self,'_omit_mol'):
            return self._omit_mol
        else:
            return False
    
    def forward(self, graph: Graph) -> Graph:
        features = getattr(graph, f'{self.level}_features')
        pos_ids = getattr(graph, f'{self.level}_position_ids')
        pos_embed = self.embedding(pos_ids)
        if self.should_omit_mol:
            pos_mask = pos_ids != self.mol_pos_id
            pos_embed_new = torch.zeros_like(pos_embed)
            pos_embed_new[pos_mask] = pos_embed[pos_mask]
            pos_embed = pos_embed_new
        if self.merge == 'add':
            features = features + pos_embed
        elif self.merge == 'concat':
            features = torch.cat([features, pos_embed], dim=-1)
        else:
            raise NotImplementedError(f'Merge action: {self.merge} not implemented')
        setattr(graph, f'{self.level}_features', features)
        return graph


class ScatterBlockFeatures(nn.Module):
    def __init__(self, merge: str='add'):
        super().__init__()
        self.merge = merge

    def forward(self, graph: Graph) -> Graph:
        if self.merge == 'add':
            graph.unit_features = graph.unit_features + graph.block_features[graph.unit2block]
        elif self.merge == 'concat':
            graph.unit_features = torch.cat([
                graph.unit_features, graph.block_features
            ], dim=-1)
        else:
            raise NotImplementedError(f'Merge action: {self.merge} not implemented')
        return graph


"""
Edge layers
"""

def variadic_arange(size):
    """
    from https://torchdrug.ai/docs/_modules/torchdrug/layers/functional/functional.html#variadic_arange

    Return a 1-D tensor that contains integer intervals of variadic sizes.
    This is a variadic variant of ``torch.arange(stop).expand(batch_size, -1)``.

    Suppose there are :math:`N` intervals.

    Parameters:
        size (LongTensor): size of intervals of shape :math:`(N,)`
    """
    starts = size.cumsum(0) - size

    range = torch.arange(size.sum(), device=size.device)
    range = range - starts.repeat_interleave(size)
    return range


def variadic_meshgrid(input1, size1, input2, size2):
    """
    from https://torchdrug.ai/docs/_modules/torchdrug/layers/functional/functional.html#variadic_meshgrid
    Compute the Cartesian product for two batches of sets with variadic sizes.

    Suppose there are :math:`N` sets in each input,
    and the sizes of all sets are summed to :math:`B_1` and :math:`B_2` respectively.

    Parameters:
        input1 (Tensor): input of shape :math:`(B_1, ...)`
        size1 (LongTensor): size of :attr:`input1` of shape :math:`(N,)`
        input2 (Tensor): input of shape :math:`(B_2, ...)`
        size2 (LongTensor): size of :attr:`input2` of shape :math:`(N,)`

    Returns
        (Tensor, Tensor): the first and the second elements in the Cartesian product
    """
    grid_size = size1 * size2
    local_index = variadic_arange(grid_size)
    local_inner_size = size2.repeat_interleave(grid_size)
    offset1 = (size1.cumsum(0) - size1).repeat_interleave(grid_size)
    offset2 = (size2.cumsum(0) - size2).repeat_interleave(grid_size)
    index1 = torch.div(local_index, local_inner_size, rounding_mode="floor") + offset1
    index2 = local_index % local_inner_size + offset2
    return input1[index1], input2[index2]


# TODO: Maybe 2D topology better depicts the "sequential edge" as this actually means bonding interactions
# think about two cases:
# 1. cyclic proteins
# 2. small molecules

def scope_mask(all_edges, segment_ids, scope):
    row, col = all_edges
    if scope == 'inner':
        select_mask = segment_ids[row] == segment_ids[col]
    elif scope == 'cross':
        select_mask = segment_ids[row] != segment_ids[col]
    elif scope == 'both':
        select_mask = torch.ones_like(row).bool()
    else:
        raise ValueError(f'Unknown scope: {scope}')
    return select_mask


class SequentialEdge(nn.Module):
    def __init__(self, max_dist: int) -> None:
        super().__init__()
        self.max_dist = max_dist

    @property
    def num_edge_type(self):
        return 2 * self.max_dist + 1
    
    def forward(self, graph: Graph, all_edges: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        row, col = all_edges
        if graph.block_position_ids is None:
            seq_dist = row - col
        else:
            seq_dist = graph.block_position_ids[row] - graph.block_position_ids[col]

        select_mask = torch.ones_like(row).bool()

        # 1. within distance threshold
        select_mask = torch.logical_and(select_mask, torch.abs(seq_dist) <= self.max_dist)

        # 2. in the same segment
        select_mask = torch.logical_and(select_mask,
            graph.segment_ids[row] == graph.segment_ids[col]
        )

        return row[select_mask], col[select_mask], seq_dist[select_mask] + self.max_dist


def scatter_sort(src: Tensor, index: Tensor, dim=0, descending=False, eps=1e-12):
    '''
    from https://github.com/rusty1s/pytorch_scatter/issues/48
    WARN: the range between src.max() and src.min() should not be too wide for numerical stability

    reproducible
    '''
    # f_src = src.float()
    # f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    # norm = (f_src - f_min)/(f_max - f_min + eps) + index.float()*(-1)**int(descending)
    # perm = norm.argsort(dim=dim, descending=descending)

    # return src[perm], perm
    src, src_perm = torch.sort(src, dim=dim, descending=descending)
    index = index.take_along_dim(src_perm, dim=dim)
    index, index_perm = torch.sort(index, dim=dim, stable=True)
    src = src.take_along_dim(index_perm, dim=dim)
    perm = src_perm.take_along_dim(index_perm, dim=0)
    return src, perm


def scatter_topk(src: Tensor, index: Tensor, k: int, dim=0, largest=True):
    indices = torch.arange(src.shape[dim], device=src.device)
    src, perm = scatter_sort(src, index, dim, descending=largest)
    index, indices = index[perm], indices[perm]
    mask = torch.ones_like(index).bool()
    mask[k:] = index[k:] != index[:-k]
    return src[mask], indices[mask]


class KNNEdge(nn.Module):
    def __init__(self, k: int, min_dist: int=0, scope: str='both') -> None:
        '''
        scope: inner, cross, or both
        '''
        super().__init__()
        self.k = k
        self.min_dist = min_dist
        self.scope = scope
    
    @property
    def num_edge_type(self):
        return 1
    
    def pre_filter(self, graph: Graph, all_edges: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        row, col = all_edges

        # 1. filter scope
        select_mask = scope_mask(all_edges, graph.segment_ids, self.scope)
        
        # 2. filter distance
        segment_ids = graph.segment_ids
        if self.min_dist > 0:
            if graph.block_position_ids is None:
                seq_dist = row - col
            else:
                seq_dist = graph.block_position_ids[row] - graph.block_position_ids[col]
            select_mask = torch.logical_and(select_mask,
                torch.logical_or(
                    torch.abs(seq_dist) >= self.min_dist,
                    segment_ids[row] != segment_ids[col]
                )
            )
        
        return row[select_mask], col[select_mask]

    
    def forward(self, graph: Graph, all_edges: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        row, col = self.pre_filter(graph, all_edges)

        # knn
        dist = _block_edge_dist(graph.unit_pos, graph.unit2block, torch.stack([row, col], dim=1))
        _, indices = scatter_topk(dist, row, k=self.k, largest=False)
        row, col = row[indices], col[indices]
        attr = torch.zeros_like(row)

        return row, col, attr
    

class RadialEdge(nn.Module):
    def __init__(self, cutoff: float, topo_cutoff: float=-1, scope: str='both', self_loop: bool=False) -> None:
        super().__init__()
        '''
        topo_cutoff: distinguish bonding and non-bonding edge
        scope: inner, cross, or both
        '''
        self.cutoff = cutoff
        self.topo_cutoff = topo_cutoff
        self.scope = scope
        self.self_loop = self_loop
        self.eps = 1e-10
        assert not (self.scope == 'cross' and self_loop), f'Self-loop is not compatible with scope {self.scope}'

    @property
    def num_edge_type(self):
        cnt = 2 if self.topo_cutoff > 0 else 1
        if self.scope == 'both':  # distinguish cross links
            cnt += 1
        if self.self_loop:
            cnt += 1
        return cnt
    
    def pre_filter(self, graph: Graph, all_edges: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        row, col = all_edges
        select_mask = scope_mask(all_edges, graph.segment_ids, self.scope)
        return row[select_mask], col[select_mask]
    
    def forward(self, graph: Graph, all_edges: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        row, col = self.pre_filter(graph, all_edges)
        dist = _block_edge_dist(graph.unit_pos, graph.unit2block, torch.stack([row, col], dim=1))
        select_mask = dist <= self.cutoff
        row, col, dist = row[select_mask], col[select_mask], dist[select_mask]
        attr = (dist < self.topo_cutoff).long() # if topo_cutoff <= 0, all will be 0

        is_self_loop = dist < self.eps
        if self.self_loop:
            attr[is_self_loop] = self.num_edge_type - 1
        else:
            not_self_loop = torch.logical_not(is_self_loop)
            row, col, attr = row[not_self_loop], col[not_self_loop], attr[not_self_loop]
        
        if self.scope == 'both':
            cross = graph.segment_ids[row] != graph.segment_ids[col]
            attr[cross] = self.num_edge_type - 2

        return row, col, attr
    

class FullyConnectEdge(RadialEdge):
    def __init__(self, topo_cutoff: float = -1, scope: str = 'both', self_loop: bool = False) -> None:
        super().__init__(float('inf'), topo_cutoff, scope, self_loop)


if __name__ == '__main__':

    num_unit_type = 10
    num_block_type = 20
    max_unit_position = 5
    max_block_position = 8

    embed_size = 16

    config = {
        'node_layers': [
            {
                'class': 'ContinuousEmbedding',
                'num_classes': num_unit_type,
                'embed_size': embed_size,
                'level': 'unit'
            },
            {
                'class': 'ContinuousEmbedding',
                'num_classes': num_block_type,
                'embed_size': embed_size,
                'level': 'block'
            },
            {
                'class': 'PositionEncoding',
                'max_position': max_unit_position,
                'embed_size': embed_size,
                'merge': 'add',
                'level': 'unit'
            },
            {
                'class': 'ScatterBlockFeatures',
                'merge': 'add'
            }

        ],
        'edge_layers': [
            {
                'class': 'SequentialEdge',
                'max_dist': 2
            },
            {
                'class': 'KNNEdge',
                'k': 2,
                'min_dist': 0
            }

        ]
    }

    constructor = GraphConstructor(config)

    n_unit = 10
    n_channel = 1
    block_num_units = torch.tensor([3, 4, 1, 2], dtype=torch.long)
    n_block = block_num_units.shape[0]
    assert block_num_units.sum() == n_unit
    num_nodes = torch.tensor([1, 3], dtype=torch.long)
    assert num_nodes.sum() == n_block

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    graph = constructor(
        unit_type=torch.randint(0, num_unit_type, (n_unit,)),
        unit_pos=torch.randn((n_unit, n_channel, 3)),
        num_nodes=num_nodes,
        unit_position_ids=torch.randint(0, max_unit_position, (n_unit,)),
        block_type=torch.randint(0, num_block_type, (n_block,)),
        block_num_units=block_num_units
    )

    print(graph)

    # print(variadic_meshgrid(
    #     torch.tensor([0, 1, 2, 3], dtype=torch.long),
    #     torch.tensor([3, 1], dtype=torch.long),
    #     torch.tensor([0, 1, 2], dtype=torch.long),
    #     torch.tensor([1, 2], dtype=torch.long),
    # ))