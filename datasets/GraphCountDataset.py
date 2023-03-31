"""
Graph substructure counting dataset.
"""

import os

import numpy as np
import scipy.io as sio
import torch
# from math import comb
from scipy.special import comb
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from typing import Callable


class GraphCountDatasetLRP(InMemoryDataset):
    r"""Graph substructure counting dataset from LRP paper : https://arxiv.org/abs/2002.04025.
    Args:
        root (str): Root path for saving dataset.
        transform (Callable): Data transformation function after saving.
        pre_transform (Callable): Data transformation function before saving.
    """
    def __init__(self,
                 root: str,
                 transform: Callable = None,
                 pre_transform: Callable = None):
        super(GraphCountDatasetLRP, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # Normalize as in GNN-AK
        self.data.y = self.data.y / self.data.y.std(0)
        a = sio.loadmat("data/subgraphcount/raw/randomgraph.mat")
        self.train_idx = torch.from_numpy(a['train_idx'][0])
        self.val_idx = torch.from_numpy(a['val_idx'][0])
        self.test_idx = torch.from_numpy(a['test_idx'][0])

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        b = self.processed_paths[0]
        a = sio.loadmat("data/subgraphcount/raw/randomgraph.mat")  # 'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A = a['A'][0]
        # list of output
        Y = a['F']

        data_list = []
        for i in range(len(A)):
            a = A[i]
            A2 = a.dot(a)
            A3 = A2.dot(a)
            tri = np.trace(A3) / 6
            tailed = ((np.diag(A3) / 2) * (a.sum(0) - 2)).sum()
            cyc4 = 1 / 8 * (np.trace(A3.dot(a)) + np.trace(A2) - 2 * A2.sum())
            cus = a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()

            deg = a.sum(0)
            star = 0
            for j in range(a.shape[0]):
                star += comb(int(deg[j]), 3)

            expy = torch.tensor([[tri, tailed, star, cyc4, cus]])

            E = np.where(A[i] > 0)
            edge_index = torch.Tensor(np.vstack((E[0], E[1]))).type(torch.int64)
            x = torch.ones(A[i].shape[0], 1)
            # y=torch.tensor(Y[i:i+1,:])
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GraphCountDatasetI2(InMemoryDataset):
    r"""Graph substructure counting dataset from I2GNN paper : https://openreview.net/pdf?id=kDSmxOspsXQ.
    Args:
        dataname (str): Dataset name for loading, choose from (count_cycle, count_graphlet).
        root (str): Root path for saving dataset.
        split (str): Dataset split, choose from (train, val, test).
        transform (Callable): Data transformation function after saving.
        pre_transform (Callable): Data transformation function before saving.
    """
    def __init__(self,
                 dataname: str = 'count_cycle',
                 root: str = 'data',
                 split: str = 'train',
                 transform: Callable = None,
                 pre_transform: Callable = None):
        self.root = root
        self.dataname = dataname
        self.transform = transform
        self.pre_transform = pre_transform
        self.raw = os.path.join(root, dataname)
        self.processed = os.path.join(root, dataname, "processed")
        super(GraphCountDatasetI2, self).__init__(root=root, transform=transform, pre_transform=pre_transform)
        split_id = 0 if split == 'train' else 1 if split == 'val' else 2
        self.data, self.slices = torch.load(self.processed_paths[split_id])
        self.y_dim = self.data.y.size(-1)
        # self.e_dim = torch.max(self.data.edge_attr).item() + 1

    @property
    def raw_dir(self):
        name = 'raw'
        return os.path.join("data", self.dataname, name)

    @property
    def processed_dir(self):
        return self.processed

    @property
    def raw_file_names(self):
        names = ["data"]
        return ['{}.mat'.format(name) for name in names]

    @property
    def processed_file_names(self):
        return ['data_tr.pt', 'data_val.pt', 'data_te.pt']

    def adj2data(self, A, y):
        # x: (n, d), A: (e, n, n)
        # begin, end = np.where(np.sum(A, axis=0) == 1.)
        begin, end = np.where(A == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        # edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        # y = torch.tensor(np.concatenate((y[1], y[-1])))
        # y = torch.tensor(y[-1])
        # y = y.view([1, len(y)])

        # sanity check
        # assert np.min(begin) == 0
        num_nodes = A.shape[0]
        if y.ndim == 1:
            y = y.reshape([1, -1])
        return Data(edge_index=edge_index, y=torch.tensor(y), num_nodes=torch.tensor([num_nodes]))

    @staticmethod
    def wrap2data(d):
        # x: (n, d), A: (e, n, n)
        x, A, y = d['x'], d['A'], d['y']
        x = torch.tensor(x)
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        y = torch.tensor(y[-1:])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def process(self):
        # process npy data into pyg.Data
        print('Processing data from ' + self.raw_dir + '...')
        raw_data = sio.loadmat(self.raw_paths[0])
        if raw_data['F'].shape[0] == 1:
            data_list_all = [[self.adj2data(raw_data['A'][0][i], raw_data['F'][0][i]) for i in idx]
                             for idx in [raw_data['train_idx'][0], raw_data['val_idx'][0], raw_data['test_idx'][0]]]
        else:
            data_list_all = [[self.adj2data(A, y) for A, y in zip(raw_data['A'][0][idx][0], raw_data['F'][idx][0])]
                             for idx in [raw_data['train_idx'], raw_data['val_idx'], raw_data['test_idx']]]
        for save_path, data_list in zip(self.processed_paths, data_list_all):
            print('pre-transforming for data at' + save_path)
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                temp = []
                for i, data in enumerate(data_list):
                    if i % 100 == 0:
                        print('Pre-processing %d/%d' % (i, len(data_list)))
                    data.num_nodes = data.num_nodes.item()
                    temp.append(self.pre_transform(data))
                data_list = temp
                # data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            torch.save((data, slices), save_path)
