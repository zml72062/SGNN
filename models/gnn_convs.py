"""
GNN conv layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import uniform
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_add

from .mlp import MLP


class GINEConv(MessagePassing):
    r"""Graph isomorphism network layer, adapted from PyG.
    Args:
        in_channels (int): Input feature size.
        out_channels (int): Output feature size.
        eps (float): Epsilon for center node information in GIN.
        train_eps (bool): If true, the epsilon is trainable.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 eps: float = 0.,
                 train_eps: bool = False,
                 norm_type: str = "Batch"):
        super(GINEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_eps = eps
        self.norm_type = norm_type
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.mlp = MLP(self.in_channels, self.out_channels, self.norm_type)
        self.reset_parameters()

    def weights_init(self,
                     m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_attr: Tensor = None,
                ) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = out + (1 + self.eps) * x
        return self.mlp(out)

    def message(self,
                x_j: Tensor,
                edge_attr: OptTensor = None) -> Tensor:
        if edge_attr is not None:
            x_j = x_j + edge_attr
        return F.relu(x_j)


class GINETupleConv(MessagePassing):
    r"""Graph isomorphism network layer, adapted from PyG.
    Args:
        in_channels (int): Input feature size.
        out_channels (int): Output feature size.
        eps (float): Epsilon for center node information in GIN.
        train_eps (bool): If true, the epsilon is trainable.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 eps: float = 0.,
                 train_eps: bool = False,
                 norm_type: str = "Batch"):
        super(GINETupleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_eps = eps
        self.norm_type = norm_type
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.tuple_proj = Linear(self.in_channels * 2, self.out_channels)
        self.mlp = MLP(self.in_channels, self.out_channels, self.norm_type)
        self.reset_parameters()

    def weights_init(self,
                     m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                tuple_edge_index: Adj,
                edge_attr: Tensor = None,
                ) -> Tensor:
        x_j = x[edge_index[1]]
        x_tuple_j = x[tuple_edge_index[1]]
        x_tuple_j = self.tuple_proj(torch.cat([x_j, x_tuple_j], dim=-1))
        if edge_attr is not None:
            x_tuple_j = x_tuple_j + edge_attr

        x_tuple_j = F.relu(x_tuple_j)
        out = scatter_add(x_tuple_j, edge_index[0], dim=self.node_dim, out=torch.zeros_like(x))
        out = out + (1 + self.eps) * x
        return self.mlp(out)


class ResGatedGraphConv(MessagePassing):
    r"""Residual gated graph convolution layer, adapted from PyG.
    Args:
        in_channels (int): Input feature size.
        out_channels (int): Output feature size.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResGatedGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_key = Linear(in_channels, out_channels)
        self.lin_query = Linear(in_channels, out_channels)
        self.lin_value = Linear(in_channels, out_channels)
        self.lin_skip = Linear(in_channels, out_channels, bias=False)
        self.act = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_attr: Tensor = None) -> Tensor:
        k = self.lin_key(x)
        q = self.lin_query(x)
        v = self.lin_value(x)
        out = self.propagate(edge_index,
                             k=k,
                             q=q,
                             v=v,
                             edge_attr=edge_attr)

        return F.relu(out) + self.lin_skip(x)

    def message(self,
                k_i: Tensor,
                q_j: Tensor,
                v_j: Tensor,
                edge_attr: OptTensor = None) -> Tensor:
        if edge_attr is not None:
            x_j = self.act(k_i + q_j) * (v_j + edge_attr)
        else:
            x_j = self.act(k_i + q_j) * v_j
        return x_j


class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution layer, adapted from PyG.
    Args:
        in_channels (int): Input feature size.
        out_channels (int): Output feature size.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 aggr: str = 'add'):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = 1
        self.proj = Linear(self.in_channels, self.out_channels)
        self.weight = nn.Parameter(Tensor(1, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.proj.reset_parameters()
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:

        x = self.proj(x)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            m = self.propagate(edge_index, x=m, edge_attr=edge_attr,
                               size=None)
            x = self.rnn(m, x)

        return x

    def message(self,
                x_j: Tensor,
                edge_attr: OptTensor = None) -> Tensor:
        if edge_attr is not None:
            x_j = x_j + edge_attr
        return x_j
