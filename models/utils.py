"""
Utils file for model construction.
"""
from copy import deepcopy as c
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data


def clones(module: nn.Module,
           N: int) -> [nn.ModuleList, None]:
    r"""Layer clone function, used for concise code writing. If input is None, simply return None.
    Args:
        module (nn.Module): Module want to clone.
        N (int): Clone times.
    """
    if module is None:
        return module
    else:
        return nn.ModuleList(c(module) for _ in range(N))


def get_pyg_attr(data: Data,
                 attr: str) -> Any:
    r"""Get attribute from PyG data. If not exist, return None instead.
    Args:
        data (torch_geometric.Data): PyG data object.
        attr (str): Attribute you want to get.
    """
    return getattr(data, attr, None)


def get_subgraph_idx(batched_data: Data) -> Tensor:
    r"""Given a batch subgraph data, generate subgraph id the node belong to for each node.
    Args:
        batched_data (Data): A batched PyG data.
    """

    num_subgraphs = batched_data.num_subgraphs
    tmp = torch.cat([torch.zeros(1, device=num_subgraphs.device, dtype=num_subgraphs.dtype),
                     torch.cumsum(num_subgraphs, dim=0)])
    graph_offset = tmp[batched_data.batch]
    subgraph_idx = graph_offset + batched_data.node2subgraph
    return subgraph_idx


def get_root_idx(batched_data: Data) -> Tensor:
    r"""Given a batch subgraph data, generate the root id for each subgraph.
    Args:
        batched_data (Data): A batched PyG data.
    """
    num_nodes_per_subgraph = batched_data.num_nodes_per_subgraph
    # repeat for each subgraph in the graph
    num_nodes_per_subgraph = num_nodes_per_subgraph[batched_data.subgraph_idx_batch]
    subgraph_offset = torch.cat(
        [torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype),
         torch.cumsum(num_nodes_per_subgraph, dim=0)])[:-1]

    root_idx = subgraph_offset + batched_data.subgraph_idx
    return root_idx


def get_node_idx(batched_data: Data) -> Tensor:
    r"""Given a batch subgraph data, generate node idx across different subgraphs.
    Args:
        batched_data (Data): A batched PyG data.
    """
    num_nodes_per_subgraph = batched_data.num_nodes_per_subgraph
    tmp = torch.cat([torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype),
                     torch.cumsum(num_nodes_per_subgraph, dim=0)])
    graph_offset = tmp[batched_data.batch]
    # Same idx for a node appearing in different subgraphs of the same graph
    node_idx = graph_offset + batched_data.subgraph_node_idx
    return node_idx


def get_transpose_idx(batched_data: Data) -> Tensor:
    r"""Given index of (u, v), find the index for (v, u).
    Args:
        batched_data (Data): A batched PyG data.
    """
    num_nodes_per_subgraph = batched_data.num_nodes_per_subgraph
    # repeat for each node in each subgraph of the graph
    num_nodes_nod = num_nodes_per_subgraph[batched_data.batch]
    # repeat for each subgraph in the graph
    num_nodes_sub = num_nodes_per_subgraph[batched_data.subgraph_idx_batch]

    subgraph_node_idx = batched_data.subgraph_node_idx
    node2subgraph = batched_data.node2subgraph

    index = torch.cat([torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype),
                       torch.cumsum(num_nodes_per_subgraph, dim=0)])[:-1]
    subgraph_offset = torch.cat(
        [torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype),
         torch.cumsum(num_nodes_sub, dim=0)])[:-1]

    graph_offset = subgraph_offset[index][batched_data.batch]

    result = subgraph_node_idx * num_nodes_nod + node2subgraph + graph_offset

    return result
