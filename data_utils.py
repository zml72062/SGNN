"""
Utils file for processing data.
"""
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Callable, Tuple
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_undirected, k_hop_subgraph, subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import coalesce
import scipy.sparse as sparse
from operator import itemgetter
from models.utils import get_pyg_attr
ORIG_EDGE_INDEX_KEY = 'original_edge_index'


# TODO: update Pytorch Geometric since this function is on the newest version
def to_undirected(edge_index,
                  edge_attr=None,
                  num_nodes=None,
                  reduce: str = "add") -> (Tensor, Optional[Tensor]):
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (string, optional): The reduce operation to use for merging edge
            features. (default: :obj:`"add"`)
    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :class:`Tensor`)
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        edge_attr = None
        num_nodes = edge_attr

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    if edge_attr is not None:
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes, reduce)

    if edge_attr is None:
        return edge_index
    else:
        return edge_index, edge_attr


def edge_list_to_sparse_adj(edge_list: np.ndarray,
                            num_nodes: int) -> sparse.coo_matrix:
    r"""Convert graph edge list to a sparse adjacency matrix.
    Args:
        edge_list (np.ndarray): Edge list of the graph.
        num_nodes (int): Number of nodes in the graph
    """
    coo = sparse.coo_matrix(([1 for _ in range(edge_list.shape[-1])], (edge_list[0, :], edge_list[1, :])),
                            shape=(num_nodes, num_nodes))
    return coo


def shortest_dist_sparse_mult(adj_mat: sparse.coo_matrix,
                              hop: int = 6,
                              source: int = None) -> np.ndarray:
    r"""Compute the shortest path distance given a graph adjacency matrix.
    Args:
        adj_mat (sparse.coo_matrix): Sparse graph adjacency matrix.
        hop (int): The maximum number of hop to consider when computing shortest path distance.
        source (int): Source node for compute the shortest path distance. If not specified, return the shortest path distance matrix.
    """
    if source is not None:
        neighbor_adj = adj_mat[source]
        ind = source
    else:
        neighbor_adj = adj_mat
        ind = np.arange(adj_mat.shape[0])
    neighbor_adj_set = [neighbor_adj]
    neighbor_dist = neighbor_adj.todense()
    for i in range(hop - 1):
        new_adj = neighbor_adj_set[i].dot(adj_mat)
        neighbor_adj_set.append(new_adj)
        update_ind = (new_adj.sign() - np.sign(neighbor_dist)) == 1
        r, c = update_ind.nonzero()
        neighbor_dist[r, c] = i + 2
    neighbor_dist[neighbor_dist < 1] = -9999
    neighbor_dist[np.arange(len(neighbor_dist)), ind] = 0
    return np.asarray(neighbor_dist)


def vec_translate(a: np.ndarray,
                  my_dict: dict) -> np.ndarray:
    r"""Given a numpy array and a dict, convert all value in the array corresponding to
        the key-value relationship specified by the dict.
    Args:
        a (np.ndarray): A numpy array.
        my_dict (dict): A dict.
    """
    return np.vectorize(my_dict.__getitem__)(a)


def get_edge_original_index(edge_list: np.ndarray,
                            edge_dict: dict) -> np.ndarray:
    r"""Return edge index for input edge list given a dict specify the index of all edge.

    Args:
        edge_list (np.ndarray): Edge list of graph.
        edge_dict (dict): A dict to save index for all edges in the graph.
    """
    return np.array(itemgetter(*map(tuple, edge_list.T))(edge_dict))


def create_edge_dict(edge_list: np.ndarray) -> dict:
    r"""Create an edge dict to record index for all edges.
    Args:
        edge_list (np.ndarray): Edge list of graph.
    """
    return dict(zip(map(tuple, edge_list.T), range(edge_list.shape[-1])))


def reindex_edge_list(edge_list: np.ndarray) -> (np.ndarray, dict, int):
    """Reindex edge list to have node index from 0 to the num_nodes - 1.
    Args:
        edge_list (np.ndarray): Edge list of graph.
    """
    unique_nodes = np.unique(edge_list)
    num_nodes = len(unique_nodes)
    index_dict = dict(zip(unique_nodes, range(num_nodes)))
    index_dict_back = dict(zip(range(num_nodes), unique_nodes))
    new_edge_list = vec_translate(edge_list, index_dict)
    return new_edge_list, index_dict_back, num_nodes


def extract_de_feature(data: Data,
                       num_hops: int) -> Data:
    r"""Extract distance encoding features.
    Args:
        data (torch_geometric.Data): A PyG graph data.
        num_hops (int): Number of components to be kept.
    """
    num_nodes = data.num_nodes
    edge_list = data.edge_index.numpy()
    dist = shortest_dist_sparse_mult(edge_list_to_sparse_adj(edge_list, num_nodes), hop=num_hops)
    dist[dist < 0] = num_hops + 1
    data.de = torch.from_numpy(dist).long()
    return data


class SubgraphData(Data):
    r"""Data abstract class for subgraph. rewrite __inc__ function to adapt different increment
        value for num_nodes_per_subgraph key.
    """
    def __inc__(self,
                key,
                value,
                *args,
                **kwargs):
        if key == ORIG_EDGE_INDEX_KEY:
            return self.num_nodes_per_subgraph
        else:
            return super().__inc__(key, value, *args, **kwargs)

def get_cross_subgraph_edge_index(data: Data,
                                  batch: Data) -> Tuple[Tensor, Tensor]:
    r"""Compute symmetric edge index (x(w,v) for w in N(u)).
    Args:
        data (torch_geometric.Data): A PyG graph data.
        batch (torch_geometric.Data): The corresponding subgraph data.
    """

    mask = get_pyg_attr(batch, "mask")
    original_edge_index = data.edge_index
    original_edge_attr = data.edge_attr
    num_nodes_per_subgraph = data.num_nodes

    new_edge_index = torch.cat([torch.where(original_edge_index[0] == i)[0].repeat(data.num_nodes)
                                for i in range(data.num_nodes)], dim=-1)
    ww = original_edge_index[1, new_edge_index] * num_nodes_per_subgraph

    vv = torch.cat([torch.repeat_interleave(torch.arange(num_nodes_per_subgraph).unsqueeze(-1),
                                torch.sum(original_edge_index[0] == i)) for i in range(data.num_nodes)])

    xv = original_edge_index[0, new_edge_index] * num_nodes_per_subgraph + vv
    wv = ww + vv
    trans_edge_index = torch.cat([xv.unsqueeze(0), wv.unsqueeze(0)], dim=0)
    keep_index = torch.logical_and(mask[trans_edge_index[0]], mask[trans_edge_index[1]]).squeeze()
    trans_edge_index = trans_edge_index[:, keep_index]

    if original_edge_attr is not None:
        trans_edge_attr = original_edge_attr[new_edge_index]
        trans_edge_attr = trans_edge_attr[keep_index]
    else:
        trans_edge_attr = None

    return trans_edge_index, trans_edge_attr

def get_tuple_edge_index(data: Data,
                         batch: Data,
                         edge_index: Tensor) -> Tensor:
    r"""Given edge_index x(u, w) for w in N(v), find corresponding x(w, v).
    Args:
        data (torch_geometric.Data): A PyG graph data.
        batch (torch_geometric.Data): The corresponding subgraph data.
        edge_index (Tensor): The edge index you want to compute tuple index.
    """

    subgraph_node_idx = batch.subgraph_node_idx
    num_nodes_per_subgraph = data.num_nodes

    vv = subgraph_node_idx[edge_index[0]]
    ww = subgraph_node_idx[edge_index[1]]
    wv = ww * num_nodes_per_subgraph + vv
    tuple_edge_index = torch.cat([edge_index[0].unsqueeze(0), wv.unsqueeze(0)], dim=0)
    return tuple_edge_index

def get_tuple_trans_edge_index(data: Data,
                               batch: Data,
                               trans_edge_index: Tensor) -> Tensor:
    r"""Given edge_index x(w, v) for w in N(u), find corresponding x(u, w).
    Args:
        data (torch_geometric.Data): A PyG graph data.
        batch (torch_geometric.Data): The corresponding subgraph data.
        trans_edge_index (Tensor): The edge index you want to compute tuple index.
    """

    num_nodes_per_subgraph = data.num_nodes
    node_to_subgraphs = batch.batch

    xx = node_to_subgraphs[trans_edge_index[0]]
    ww = node_to_subgraphs[trans_edge_index[1]]
    xw = xx * num_nodes_per_subgraph + ww
    tuple_trans_edge_index = torch.cat([trans_edge_index[0].unsqueeze(0), xw.unsqueeze(0)], dim=0)
    return tuple_trans_edge_index


class Graph2Subgraph:
    r"""Abstract class for process graph into batch of subgraphs.
    Args:
        encoding_function (function): A function to compute additional
                                      encoding for subgraph generation.
        process_subgraphs (function): A fuction to further process each generated subgraph.
    """

    def __init__(self,
                 encoding_function=lambda x: x,
                 process_subgraphs=lambda x: x):
        self.encoding_function = encoding_function
        self.process_subgraphs = process_subgraphs

    def __call__(self,
                 data: Data) -> Data:
        assert data.is_undirected()
        data = self.encoding_function(data)
        subgraphs = self.to_subgraphs(data)
        subgraphs = [self.process_subgraphs(s) for s in subgraphs]

        batch = Batch.from_data_list(subgraphs)
        node_aug_feature = get_pyg_attr(batch, "node_aug_feature")
        edge_aug_feature = get_pyg_attr(batch, "edge_aug_feature")
        z = get_pyg_attr(batch, "z")
        edge_index = batch.edge_index
        trans_edge_index, trans_edge_attr = get_cross_subgraph_edge_index(data, batch)
        tuple_edge_index = get_tuple_edge_index(data, batch, edge_index)
        tuple_trans_edge_index = get_tuple_trans_edge_index(data, batch, trans_edge_index)


        return SubgraphData(x=batch.x,
                            z=z,
                            # edge index for all subgraphs
                            edge_index=edge_index,
                            tuple_edge_index=tuple_edge_index,
                            edge_attr=batch.edge_attr,
                            trans_edge_index=trans_edge_index,
                            tuple_trans_edge_index=tuple_trans_edge_index,
                            trans_edge_attr=trans_edge_attr,
                            node_aug_feature=node_aug_feature,
                            edge_aug_feature=edge_aug_feature,
                            # each node belong to which subgraph, [repeat(i, num_nodes) for i in num_subgraphs].
                            node2subgraph=batch.batch,
                            y=data.y,
                            # index for all subgraphs. range(num_subgraphs)
                            subgraph_idx=batch.subgraph_idx,
                            node_idx=torch.arange(data.num_nodes),
                            # index for all nodes in all subgraphs repeat(range(num_nodes), num_subgraphs)
                            subgraph_node_idx=batch.subgraph_node_idx,
                            num_subgraphs=len(subgraphs),
                            num_nodes_per_subgraph=data.num_nodes,
                            original_edge_index=data.edge_index,
                            original_edge_attr=data.edge_attr)

    def to_subgraphs(self, data):
        raise NotImplementedError



class NodeDeleted(Graph2Subgraph):
    r"""Subgraph generation with node deletion policy.
    """
    def to_subgraphs(self,
                     data: Data) -> Data:
        r"""Generate subgraph by deleting one node for each subgraph (Keep the deleted node as isolated).
        Args:
            data (torch_geometric.Data): A PyG graph data.
        """
        subgraphs = []
        all_nodes = torch.arange(data.num_nodes)

        for i in range(data.num_nodes):
            subset = torch.cat([all_nodes[:i], all_nodes[i + 1:]])
            subgraph_edge_index, subgraph_edge_attr = subgraph(subset, data.edge_index, data.edge_attr,
                                                               relabel_nodes=False, num_nodes=data.num_nodes)
            node_mask = torch.ones([data.num_nodes, 1])
            node_mask[i] = 0
            subgraphs.append(
                Data(
                    x=data.x,
                    mask=node_mask,
                    edge_index=subgraph_edge_index,
                    edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i),
                    subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        return subgraphs


class NodeMarked(Graph2Subgraph):
    r"""Subgraph generation with node marking policy.
    """
    def to_subgraphs(self,
                     data: Data) -> Data:
        r"""Generate subgraph by marking the center node for each subgraph.
        Args:
            data (torch_geometric.Data): A PyG graph data.
        """
        subgraphs = []
        x = data.x
        for i in range(data.num_nodes):
            z = torch.arange(2).repeat(data.num_nodes, 1)
            z[i] = torch.tensor([z[i, 1], z[i, 0]])

            subgraphs.append(
                Data(
                    x=x,
                    z=z,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    subgraph_idx=torch.tensor(i),
                    subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        return subgraphs


class EgoNets(Graph2Subgraph):
    r"""Subgraph generation with ego network policy.
    Args:
        num_hops (int): Number of hop in ego network.
        add_additional_feature (str): The type of additional feature to add in ego network, choose from (None, DE, ID).
        encoding_function (Callable): A function to compute additional encoding for input graph.
        process_subgraphs (Callable): Subgraph process function.
    """
    def __init__(self,
                 num_hops: int,
                 add_additional_feature: str = "None",
                 encoding_function: Callable = lambda x: x,
                 process_subgraphs: Callable = lambda x: x):
        self.num_hops = num_hops
        self.add_additional_feature = add_additional_feature

        if self.add_additional_feature == "DE":
            encoding_function = lambda x: extract_de_feature(x, num_hops)

        super().__init__(encoding_function, process_subgraphs)

    def to_subgraphs(self,
                     data: Data) -> Data:
        r"""Generate subgraph by extracting K-hop egonet of center node for each subgraph
            (keep nodes not in K-hop egonet as isolated nodes).
        Args:
            data (torch_geometric.Data): A PyG graph data.
        """
        subgraphs = []
        x = data.x
        for i in range(data.num_nodes):

            subset, _, _, edge_mask = k_hop_subgraph(i, self.num_hops, data.edge_index, relabel_nodes=False,
                                                num_nodes=data.num_nodes)
            subgraph_edge_index = data.edge_index[:, edge_mask]
            subgraph_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else data.edge_attr
            node_mask = torch.zeros([data.num_nodes, 1])
            node_mask[subset] = 1

            if self.add_additional_feature == "ID":

                z = torch.arange(2).repeat(data.num_nodes, 1)
                z[i] = torch.tensor([z[i, 1], z[i, 0]])

            elif self.add_additional_feature == "DE":
                z = data.de[:, i].unsqueeze(-1)
            else:
                z = None

            subgraphs.append(
                Data(
                    x=x,
                    z=z,
                    mask=node_mask,
                    edge_index=subgraph_edge_index,
                    edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i),
                    subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        return subgraphs



def policy2transform(policy: str, num_hops, process_subgraphs=lambda x: x):
    r"""Subgraph generation policy selection.
    Args:
        policy (str): Subgraph generation polict name.
        num_hops (int): number of hop in ego network related policies.
        process_subgraphs (Callable): Subgraph process function.
    """

    if policy == "node_deleted":
        return NodeDeleted(process_subgraphs=process_subgraphs)
    elif policy == "node_marked":
        return NodeMarked(process_subgraphs=process_subgraphs)
    elif policy == "ego_nets":
        return EgoNets(num_hops,
                       process_subgraphs=process_subgraphs)
    elif policy == "ego_nets_plus":
        return EgoNets(num_hops,
                       add_additional_feature="ID",
                       process_subgraphs=process_subgraphs)
    elif policy == "ego_nets_de":
        return EgoNets(num_hops,
                       add_additional_feature="DE",
                       process_subgraphs=process_subgraphs)
    elif policy == "original":
        return process_subgraphs

    raise ValueError("Invalid subgraph policy type")

