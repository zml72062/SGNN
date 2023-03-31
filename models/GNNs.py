"""
General GNN framework.
"""
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_add_pool

from models.mlp import MLP
from .jumping_knowledge import JumpingKnowledge
from .norms import Normalization
from .utils import *


class GNN(nn.Module):
    r"""A generalized GNN framework.
    Args:
        num_layers (int): The number of GNN layer.
        gnn_layer (nn.Module): The gnn layer used in GNN model.
        init_encoder (nn.Module): Initial node feature encoder.
        edge_encoder (nn.Module): Edge feature encoder.
        node_aug_encoder (nn.Module): Node augmented feature encoder.
        edge_aug_encoder (nn.Module): Edge augmented feature encoder.
        JK (str): Method of jumping knowledge, choose from (last, concat, max, sum, attention).
        norm_type (str): Method of normalization, choose from (Batch, Layer, Instance, GraphSize, Pair).
        residual (bool): If true, add residual connection between each layer.
        drop_prob (float): Dropout rate.
    """

    def __init__(self,
                 num_layers: int,
                 gnn_layer: nn.Module,
                 init_encoder: nn.Module,
                 edge_encoder: nn.Module = None,
                 node_aug_encoder: nn.Module = None,
                 edge_aug_encoder: nn.Module = None,
                 JK: str = "last",
                 norm_type: str = "Batch",
                 residual: bool = False,
                 drop_prob: float = 0.1):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = gnn_layer.out_channels
        self.JK = JK
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob
        self.subgraph_pooling = None

        self.dropout = nn.Dropout(drop_prob)
        self.init_encoder = init_encoder
        self.edge_encoders = clones(edge_encoder, self.num_layers)
        self.node_aug_encoders = clones(node_aug_encoder, self.num_layers)
        self.edge_aug_encoders = clones(edge_aug_encoder, self.num_layers)
        self.jk_decoder = JumpingKnowledge(self.hidden_channels, self.JK, self.num_layers, self.drop_prob)

        # gnn layer list
        self.gnns = clones(gnn_layer, self.num_layers)
        # norm list
        norm = Normalization(self.hidden_channels, norm_type=self.norm_type)
        self.norms = clones(norm, self.num_layers)

        self.reset_parameters()

    def weights_init(self,
                     m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.init_encoder.reset_parameters()

        for g in self.gnns:
            g.reset_parameters()

        for n in self.norms:
            n.reset_parameters()

        if self.edge_encoders is not None:
            for e in self.edge_encoders:
                e.reset_parameters()

        if self.node_aug_encoders is not None:
            for n in self.node_aug_encoders:
                n.reset_parameters()

        if self.edge_aug_encoders is not None:
            for e in self.edge_aug_encoders:
                e.reset_parameters()

        self.jk_decoder.reset_parameters()

    def forward(self,
                data: Data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        num_edges = edge_index.size(-1)
        edge_attr = get_pyg_attr(data, "edge_attr")
        node_aug_feature = get_pyg_attr(data, "node_aug_feature")
        edge_aug_feature = get_pyg_attr(data, "edge_aug_feature")

        # initial projection
        x = self.init_encoder(x).squeeze()

        # forward in gnn layer
        h_list = [x]
        for l in range(self.num_layers):
            h = h_list[l]
            if node_aug_feature is not None and self.node_aug_encoders is not None:
                h = h + self.node_aug_encoders[l](node_aug_feature).squeeze()

            edge_emb = torch.zeros([num_edges, self.hidden_channels], device=edge_index.device)
            if edge_attr is not None and self.edge_encoders is not None:
                edge_feature_emb = self.edge_encoders[l](edge_attr).squeeze()
                num_edge_feature = edge_feature_emb.size(0)
                # K hop, edge greater than 1 hop don't have edge feature
                if num_edge_feature != num_edges:
                    padding_dim = num_edges - num_edge_feature
                    padding_emb = torch.zeros([padding_dim, self.hidden_channels], device=edge_feature_emb.device)
                    edge_feature_emb = torch.cat([edge_feature_emb, padding_emb], dim=0)
                edge_emb += edge_feature_emb

            if edge_aug_feature is not None and self.edge_aug_encoders is not None:
                edge_emb += self.edge_aug_encoders[l](edge_aug_feature).squeeze()

            h = self.gnns[l](h, edge_index, edge_emb)
            h = self.norms[l](h)
            # if not the last gnn layer, add dropout layer
            if l != self.num_layers - 1:
                h = self.dropout(h)

            if self.residual:
                h = h + h_list[l]
            h_list.append(h)

        return self.jk_decoder(h_list)


class SGNN(nn.Module):
    r"""A generalized subgraph GNN framework with additional augmented feature.
    Args:
        num_layers (int): the total number of GNN layer.
        gnn_layer (nn.Module): gnn layer used in GNN model.
        init_encoder (nn.Module): initial node feature encoding.
        subgraph_encoder (nn.Module): Additional subgraph feature encoder.
        edge_encoder (nn.Module): Edge feature encoder.
        add_lu (Bool): If true, ad local message passing (neighbor of node v in subgraoh u).
        add_vv (Bool): If true, add vv aggregation (node v in subgraph v) in message passing.
        add_vu (Bool): If true, add vu aggregation (node u in subgraph v) in message passing.
        add_global (Bool): If true, add gloabl aggregation (node v in all subgraphs) in message passing.
        add_lv (Bool): If true, add symmetric local message passing (Neighbors of node u in subgraph v).
        add_lu_tuple (Bool): If true, add local tuple aggregation (x(u, w), x(w, v)) for neighbors of node v in subgraph u.
        add_lv_tuple (Bool): If true, add local tuple aggregation (x(u, w), x(w, v)) for neighbors of node u in subgraph v.
        JK (str):method of jumping knowledge, last,concat,max or sum.
        norm_type (str): method of normalization, batch or layer.
        residual (bool): whether to add residual connection.
        drop_prob (float): dropout rate.
    """

    def __init__(self,
                 num_layers: int,
                 gnn_layer: nn.Module,
                 init_encoder: nn.Module,
                 subgraph_encoder: nn.Module = None,
                 edge_encoder: nn.Module = None,
                 node_aug_encoder: nn.Module = None,
                 edge_aug_encoder: nn.Module = None,
                 add_lu: bool = True,
                 add_vv: bool = False,
                 add_vu: bool = False,
                 add_global: bool = False,
                 add_lv: bool = False,
                 add_lu_tuple: bool = False,
                 add_lv_tuple: bool = False,
                 JK: str = "last",
                 subgraph_pooling: bool = "SV",
                 norm_type: bool = "Batch",
                 residual: bool = False,
                 drop_prob: float = 0.1):
        super(SGNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = gnn_layer.out_channels
        self.JK = JK
        self.add_lu = add_lu
        self.add_vv = add_vv
        self.add_vu = add_vu
        self.add_global = add_global
        self.add_lv = add_lv
        self.add_lv_tuple = add_lv_tuple
        self.add_lu_tuple = add_lu_tuple
        self.subgraph_pooling = subgraph_pooling
        self.norm_type = norm_type
        self.residual = residual
        self.drop_prob = drop_prob

        self.dropout = nn.Dropout(drop_prob)
        self.init_encoder = init_encoder
        self.subgraph_encoder = subgraph_encoder
        # if self.subgraph_encoder is not None:
        #    self.concat_projs = clones(Linear(2 * self.hidden_channels, self.hidden_channels), self.num_layers)

        # augmented feature encoder
        self.node_aug_encoders = clones(node_aug_encoder, self.num_layers)
        self.edge_aug_encoders = clones(edge_aug_encoder, self.num_layers)
        self.jk_decoder = JumpingKnowledge(self.hidden_channels,
                                           self.JK,
                                           self.num_layers,
                                           self.drop_prob)

        # gnn layer list
        gnns = clones(gnn_layer, self.num_layers)
        # mlp layer list
        mlp = MLP(self.hidden_channels, self.hidden_channels, self.norm_type)
        mlps = clones(mlp, self.num_layers)

        # learnable root weight
        eps = torch.nn.Parameter(torch.Tensor([0. for _ in range(self.num_layers)]))

        # norm list
        norm = Normalization(self.hidden_channels, self.norm_type)
        norms = clones(norm, self.num_layers)

        # edge feature encoder
        edge_encoders = clones(edge_encoder, self.num_layers)

        # intra subgraph message passing
        if self.add_lu or self.add_lu_tuple:
            self.lu_gnns = c(gnns)
            self.lu_norms = c(norms)
            self.lu_edge_encoders = c(edge_encoders)

        if self.add_vv:
            self.vv_mlps = c(mlps)
            self.vv_norms = c(norms)
            self.vv_eps = c(eps)

        if self.add_vu:
            self.vu_mlps = c(mlps)
            self.vu_norms = c(norms)
            self.vu_eps = c(eps)

        if self.add_global:
            self.global_mlps = c(mlps)
            self.global_norms = c(norms)
            self.global_eps = c(eps)

        if self.add_lv or self.add_lv_tuple:
            self.lv_gnns = c(gnns)
            self.lv_norms = c(norms)
            self.lv_edge_encoders = c(edge_encoders)

        self.reset_parameters()

    def weights_init(self,
                     m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def reset_parameters(self):
        self.init_encoder.reset_parameters()
        self.jk_decoder.reset_parameters()
        if self.add_lu or self.add_lu_tuple:
            for g in self.lu_gnns:
                g.reset_parameters()
            for n in self.lu_norms:
                n.reset_parameters()

        if self.lu_edge_encoders is not None:
            for e in self.lu_edge_encoders:
                e.reset_parameters()

        if self.subgraph_encoder is not None:
            self.subgraph_encoder.reset_parameters()
            # for c in self.concat_projs:
            #    c.reset_parameters()

        if self.node_aug_encoders is not None:
            for n in self.node_aug_encoders:
                n.reset_parameters()

        if self.edge_aug_encoders is not None:
            for e in self.edge_aug_encoders:
                e.reset_parameters()

        if self.add_lv or self.add_lv_tuple:
            for g in self.lv_gnns:
                g.reset_parameters()
            for n in self.lv_norms:
                n.reset_parameters()
            if self.lv_edge_encoders is not None:
                for e in self.lv_edge_encoders:
                    e.reset_parameters()

        if self.add_vv:
            for m in self.vv_mlps:
                m.reset_parameters()
            for n in self.vv_norms:
                n.reset_parameters()
            self.vv_eps.data.fill_(0.)

        if self.add_vu:
            for m in self.vu_mlps:
                m.reset_parameters()
            for n in self.vu_norms:
                n.reset_parameters()
            self.vu_eps.data.fill_(0.)

        if self.add_global:
            for m in self.global_mlps:
                m.reset_parameters()
            for n in self.global_norms:
                n.reset_parameters()
            self.global_eps.data.fill_(0.)

    def forward(self,
                data: Data) -> Tensor:
        x, edge_index, tuple_edge_index, trans_edge_index, tuple_trans_edge_index, original_edge_index = \
            data.x, data.edge_index, data.tuple_edge_index, data.trans_edge_index, data.tuple_trans_edge_index, data.original_edge_index

        num_edges = edge_index.size(-1)
        num_trans_edge = trans_edge_index.size(-1)
        edge_attr = get_pyg_attr(data, "edge_attr")
        trans_edge_attr = get_pyg_attr(data, "trans_edge_attr")
        node_aug_feature = get_pyg_attr(data, "node_aug_feature")
        edge_aug_feature = get_pyg_attr(data, "edge_aug_feature")
        z = get_pyg_attr(data, "z")
        node_idx = get_node_idx(data)
        subgraph_idx = get_subgraph_idx(data)
        root_idx = get_root_idx(data)
        transpose_idx = get_transpose_idx(data)
        # transpose_reverse_idx = torch.argsort(transpose_idx)

        # initial projection
        x = self.init_encoder(x).squeeze()
        if self.subgraph_encoder is not None and z is not None:
            x += self.subgraph_encoder(z).squeeze()

        # forward in gnn layer
        h_list = [x]
        for l in range(self.num_layers):
            h = h_list[l]
            # augmented node feature encoding
            if self.node_aug_encoders is not None and node_aug_feature is not None:
                h += self.node_aug_encoders[l](node_aug_feature).squeeze()
            # augmented edge feature encoding, only add to lu aggregation
            edge_aug_emb = torch.zeros([num_edges, self.hidden_channels], device=edge_index.device)
            if self.edge_aug_encoders is not None and edge_aug_feature is not None:
                edge_aug_emb += self.edge_aug_encoders[l](edge_aug_feature).squeeze()

            out = torch.zeros_like(h)
            if self.add_lu:
                # lu edge embedding
                lu_edge_emb = edge_aug_emb
                if edge_attr is not None:
                    lu_edge_emb += self.lu_edge_encoders[l](edge_attr).squeeze()
                h_lu = self.lu_norms[l](self.lu_gnns[l](h, edge_index, lu_edge_emb))
                out += h_lu

            # lu tuple aggregation
            if self.add_lu_tuple:
                # lu edge embedding
                lu_edge_emb = edge_aug_emb
                if edge_attr is not None:
                    lu_edge_emb += self.lu_edge_encoders[l](edge_attr).squeeze()
                h_lu_tuple = self.lu_norms[l](self.lu_gnns[l](h, edge_index, tuple_edge_index, lu_edge_emb))
                out += h_lu_tuple

            # vv aggregation
            if self.add_vv:
                h_root = h[root_idx]
                h_vv = self.vv_norms[l](self.vv_mlps[l]((1 + self.vv_eps[l]) * h + h_root[node_idx]))
                out += h_vv

            # vu aggregation
            if self.add_vu:
                h_transpose = h[transpose_idx]
                h_vu = self.vu_norms[l](self.vu_mlps[l]((1 + self.vu_eps[l]) * h + h_transpose))
                out += h_vu

            # global v aggregation
            if self.add_global:
                h_global = global_mean_pool(h, node_idx)
                h_global = self.global_norms[l](self.global_mlps[l]((1 + self.global_eps[l]) * h + h_global[node_idx]))
                out += h_global

            # lv aggregation
            if self.add_lv:
                lv_edge_emb = torch.zeros([num_trans_edge, self.hidden_channels], device=trans_edge_index.device)
                if edge_attr is not None:
                    lv_edge_emb += self.lv_edge_encoders[l](trans_edge_attr).squeeze()
                h_lv = self.lv_norms[l](self.lv_gnns[l](h, trans_edge_index, lv_edge_emb))
                out += h_lv

            if self.add_lv_tuple:
                lv_edge_emb = torch.zeros([num_trans_edge, self.hidden_channels], device=trans_edge_index.device)
                # for SLFWL, we share the weight for lu and lv
                if edge_attr is not None:
                    lv_edge_emb += self.lv_edge_encoders[l](trans_edge_attr).squeeze()
                h_lv_tuple = self.lv_norms[l](self.lv_gnns[l](h, trans_edge_index, tuple_trans_edge_index, lv_edge_emb))
                out += h_lv_tuple

            out = self.dropout(out)
            if self.residual:
                out = out + h_list[l]

            h_list.append(out)

        # subgraph_pooling
        if self.subgraph_pooling == "VS":
            h_list = [global_add_pool(h, subgraph_idx) for h in h_list]
        else:
            h_list = [global_add_pool(h, node_idx) for h in h_list]

        return self.jk_decoder(h_list)
