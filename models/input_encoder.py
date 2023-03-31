"""
Different encoder for different dataset.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.dense import Linear

from .utils import get_pyg_attr


class EmbeddingEncoder(nn.Module):
    r"""Input encoder with embedding layer.
    Args:
        in_channels (int): Input feature size.
        hidden_channels (int): Hidden size.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int):
        super(EmbeddingEncoder, self).__init__()
        self.init_proj = nn.Embedding(in_channels, hidden_channels)

    def reset_parameters(self):
        self.init_proj.reset_parameters()

    def forward(self,
                x: Tensor) -> Tensor:
        return self.init_proj(x)


class LinearEncoder(nn.Module):
    r"""Input encoder with linear projection layer.
    Args:
        in_channels (int): Input feature size.
        hidden_channels (int): Hidden size.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int):
        super(LinearEncoder, self).__init__()
        self.init_proj = Linear(in_channels, hidden_channels)

    def reset_parameters(self):
        self.init_proj.reset_parameters()

    def forward(self,
                x: Tensor) -> Tensor:
        return self.init_proj(x)


class SubgraphFeatureEncoder(nn.Module):
    r"""Encoder for additional subgraph feature.
    Args
        encoder (nn.Module): Feature encoder.
        to_dense (bool): If true, convert feature to dense batch.
    """

    def __init__(self,
                 encoder: nn.Module,
                 to_dense: bool = False):
        super(SubgraphFeatureEncoder, self).__init__()
        self.encoder = encoder
        self.to_dense = to_dense

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self,
                x: Tensor,
                batch: Tensor) -> Tensor:
        if self.to_dense:
            return self.encoder(x, batch)
        else:
            return self.encoder(x)


class QM9InputEncoder(nn.Module):
    r"""Input encoder for QM9 dataset, which add z and pos embedding.
    Args:
        hidden_channels (int): Hidden size.
        use_pos (bool): If True, add position feature to embedding.
    """

    def __init__(self,
                 hidden_channels: int,
                 use_pos: bool = False):
        super(QM9InputEncoder, self).__init__()
        self.use_pos = use_pos
        if use_pos:
            in_channels = 22
        else:
            in_channels = 19
        self.init_proj = Linear(in_channels, hidden_channels)
        self.z_embedding = nn.Embedding(1000, 8)

    def reset_parameters(self):
        self.init_proj.reset_parameters()
        self.z_embedding.reset_parameters()

    def forward(self,
                data: Data) -> Tensor:
        x = data.x
        z = data.z
        pos = get_pyg_attr(data, "pos")

        z_emb = 0
        if z is not None:
            # computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)

        # concatenate with continuous node features
        x = torch.cat([z_emb, x], -1)

        if self.use_pos:
            x = torch.cat([x, pos], 1)

        x = self.init_proj(x)

        return x
