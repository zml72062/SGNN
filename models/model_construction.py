"""
Model construction and Pytorch lighting integration.
"""
from argparse import ArgumentParser

from models.GNNs import *
from models.gnn_convs import *
from models.input_encoder import *
from models.output_decoder import *


def make_gnn_layer(args: ArgumentParser) -> nn.Module:
    r"""Function to construct gnn layer.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """
    if args.gnn_name == "GINE":
        gnn_layer = GINEConv(args.hidden_channels,
                             args.hidden_channels,
                             eps=args.eps,
                             train_eps=args.train_eps,
                             norm_type=args.norm_type
                             )
    elif args.gnn_name == "GINETuple":
        gnn_layer = GINETupleConv(args.hidden_channels,
                                  args.hidden_channels,
                                  eps=args.eps,
                                  train_eps=args.train_eps,
                                  norm_type=args.norm_type)

    elif args.gnn_name == "resGatedGCN":
        gnn_layer = ResGatedGraphConv(args.hidden_channels, args.hidden_channels)
    elif args.gnn_name == "gatedGraph":
        gnn_layer = GatedGraphConv(args.hidden_channels,
                                   args.hidden_channels)

    else:
        raise ValueError("Not supported GNN type")
    return gnn_layer


def make_GNN(args: ArgumentParser,
             gnn_layer: nn.Module,
             edge_encoder: nn.Module,
             init_encoder: nn.Module) -> nn.Module:
    r"""Make GNN model given input parameters.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
        gnn_layer (nn.Module): GNN layer.
        edge_encoder (nn.Module): Edge feature input encoder.
        init_encoder (nn.Module): Node feature initial encoder.
    """

    node_aug_encoder, edge_aug_encoder = None, None

    # construct gnn modle
    if args.model_name == "WL":

        gnn = GNN(num_layers=args.num_layers,
                  gnn_layer=gnn_layer,
                  init_encoder=init_encoder,
                  edge_encoder=edge_encoder,
                  node_aug_encoder=node_aug_encoder,
                  edge_aug_encoder=edge_aug_encoder,
                  JK=args.JK,
                  norm_type=args.norm_type,
                  residual=args.residual,
                  drop_prob=args.drop_prob)


    elif args.model_name in ["SWL", "PSWL", "GSWL", "SSWL", "SSWL+", "LFWL", "SLFWL"]:
        if args.model_name in ["LFWL", "SLFWL"]:
            assert args.gnn_name == "GINETuple"

        # subgraph feature encoder
        if args.policy == "ego_nets_de":
            subgraph_encoder = EmbeddingEncoder(args.num_hops + 2, args.hidden_channels)
        elif args.policy in ["ego_net_plus", "node_marked"]:
            subgraph_encoder = LinearEncoder(2, args.hidden_channels)
        else:
            subgraph_encoder = None

        add_lu = True
        add_vv = False
        add_vu = False
        add_global = False
        add_lv = False
        add_lv_tuple = False
        add_lu_tuple = False

        if args.model_name == "PSWL":
            add_vv = True
        elif args.model_name == "GSWL":
            add_global = True
        elif args.model_name == "SSWL":
            add_vv = True
            add_vu = True
        elif args.model_name == "SSWL+":
            add_vv = True
            add_lv = True
        elif args.model_name == "LFWL":
            add_lu = False
            add_vv = True
            add_lu_tuple = True
        elif args.model_name == "SLFWL":
            add_lu = False
            add_vv = True
            add_lu_tuple = True
            add_lv_tuple = True

        gnn = SGNN(num_layers=args.num_layers,
                   gnn_layer=gnn_layer,
                   init_encoder=init_encoder,
                   edge_encoder=edge_encoder,
                   subgraph_encoder=subgraph_encoder,
                   node_aug_encoder=node_aug_encoder,
                   edge_aug_encoder=edge_aug_encoder,
                   add_lu=add_lu,
                   add_vv=add_vv,
                   add_global=add_global,
                   add_lv=add_lv,
                   add_vu=add_vu,
                   add_lu_tuple=add_lu_tuple,
                   add_lv_tuple=add_lv_tuple,
                   subgraph_pooling=args.subgraph_pooling,
                   norm_type=args.norm_type,
                   residual=args.residual,
                   drop_prob=args.drop_prob
                   )

    else:
        raise ValueError("Not implemented model class.")
    return gnn


def make_decoder(args, embedding_model):
    r"""Make decoder layer for different dataset.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
        embedding_model (nn.Module): Graph representation model, typically a gnn output node representation.
    """
    if args.dataset_name in ["ZINC", "StructureCounting", "GraphProperty"]:
        model = GraphRegression(embedding_model, pooling_method=args.pooling_method)
    elif args.dataset_name in ["count_cycle", "count_graphlet"]:
        model = NodeRegression(embedding_model)
    else:
        model = GraphClassification(embedding_model, out_channels=args.out_channels, pooling_method=args.pooling_method)
    return model


def make_model(args: ArgumentParser,
               init_encoder: nn.Module = None,
               edge_encoder: nn.Module = None) -> nn.Module:
    r"""Make learning model given input arguments.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
        init_encoder (nn.Module): Node feature initial encoder.
        edge_encoder (nn.Module): Edge feature encoder.
    """

    gnn_layer = make_gnn_layer(args)
    gnn = make_GNN(args, gnn_layer, edge_encoder, init_encoder)
    model = make_decoder(args, gnn)
    return model
