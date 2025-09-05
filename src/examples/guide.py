# -*- coding: utf-8 -*-
"""Higher-order Structure based Anomaly Detection on Attributed
    Networks (GUIDE)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import os
import torch
import hashlib
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import NeighborLoader
from networkx.generators.atlas import graph_atlas_g
from torch_geometric.utils import to_dense_adj


from base import BaseDetector
from utility import validate_device
from metrics import eval_roc_auc


from typing import Any, Dict, List, Optional, Union, Callable
from torch.nn import BatchNorm1d, Identity, Linear, ModuleList
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
import os


class GCN(torch.nn.Module):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.
    Adapted from PyG for upward compatibility
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`"last"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[Callable, None] = F.gelu,
        norm: Optional[torch.nn.Module] = None,
        jk: Optional[str] = None,
        act_first: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = dropout
        self.act = act
        self.jk_mode = jk
        self.act_first = act_first

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        self.convs.append(
            self.init_conv(in_channels, hidden_channels, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))
        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(hidden_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))

        self.norms = None
        if norm is not None:
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(copy.deepcopy(norm))
            if jk is not None:
                self.norms.append(copy.deepcopy(norm))

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == 'cat':
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels)

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, **kwargs)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        """"""
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class GUIDE(BaseDetector):
    """
    GUIDE (Higher-order Structure based Anomaly Detection on Attributed
    Networks) is an anomaly detector consisting of an attribute graph
    convolutional autoencoder, and a structure graph attentive
    autoencoder (not the same as the graph attention networks). Instead
    of the adjacency matrix, node motif degree is used as input of
    structure autoencoder. The reconstruction mean square error of the
    autoencoders are defined as structure anomaly score and attribute
    anomaly score, respectively.

    Note: The calculation of node motif degree in preprocessing has
    high time complexity. It may take longer than you expect.

    See :cite:`yuan2021higher` for details.

    Parameters
    ----------
    a_hid :  int, optional
        Hidden dimension for attribute autoencoder. Default: ``32``.
    s_hid :  int, optional
        Hidden dimension for structure autoencoder. Default: ``4``.
    num_layers : int, optional
        Total number of layers in autoencoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    alpha : float, optional
        Loss balance weight for attribute and structure. ``None`` for
        balancing by standard deviation. Default: ``None``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``10``.
    gpu : int, optional
        GPU Index, -1 for using CPU. Default: ``0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    graphlet_size : int, optional
        The maximum graphlet size used to compute structure input.
        Default: ``4``.
    selected_motif : bool, optional
        Use selected motifs which are defined in the original paper.
        Default: ``True``.
    cache_dir : str, option
        The directory for the node motif degree caching.
        Default: ``None``.
    verbose : bool, optional
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import GUIDE
    >>> model = GUIDE()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 a_hid=64,
                 s_hid=32,
                 num_layers=8,
                 dropout=0.,
                 weight_decay=0.,
                 act=F.gelu,
                 alpha=0.2,
                 contamination=0.1,
                 lr=0.0005,
                 epoch=200,
                 gpu=0,
                 graphlet_size=4,
                 selected_motif=True,
                 cache_dir=None,
                 verbose=True):
        super(GUIDE, self).__init__(contamination=contamination)

        # model param
        self.a_hid = a_hid
        self.s_hid = s_hid
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)


        # other param
        self.graphlet_size = graphlet_size
        if selected_motif:
            assert self.graphlet_size == 4, \
                "Graphlet size is fixed when using selected motif"
        self.selected_motif = selected_motif
        self.verbose = verbose
        self.model = None

        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.pygod')
        self.cache_dir = cache_dir

    def fit(self, loader, in_feats, adj_feats, y_true=None):
        """
        Fit detector with input data.

        Parameters
        ----------
        G : torch_geometric.data.Data
            The input data.
        y_true : numpy.ndarray, optional
            The optional outlier ground truth labels used to monitor
            the training progress. They are not used to optimize the
            unsupervised model. Default: ``None``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.in_feats = in_feats
        self.adj_feats = adj_feats


        self.model = GUIDE_Base(a_dim=self.in_feats,
                                s_dim=self.adj_feats,
                                a_hid=self.a_hid,
                                s_hid=self.s_hid,
                                num_layers=self.num_layers,
                                dropout=self.dropout,
                                act=self.act).to(self.device) # s_dim 要求节点数量是固定的

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        self.model.train()
        # decision_scores = np.zeros(G.x.shape[0])
        for epoch in range(self.epoch):
            epoch_loss = 0
            t = 0
            for data in loader:
                # data = data.to(self.device)
                # s = to_dense_adj(data.edge_index).squeeze(0)
                # print(s.shape)

                x_, s_ = self.model(data.x, data.adj, data.edge_index)
                score = self.loss_func(data.x, x_, data.adj, s_)

                loss = torch.mean(score)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t = t + 1

            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, epoch_loss / t), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_scores)
                    print(" | AUC {:.4f}".format(auc), end='')
                print()

        save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }, os.path.join('./train_model/guide/'))

        return self

    def decision_function(self, data):
        """
        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        outlier_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        self.model2 = GUIDE_eval(a_dim=self.in_feats,
                                s_dim=self.adj_feats,
                                a_hid=self.a_hid,
                                s_hid=self.s_hid,
                                num_layers=self.num_layers,
                                dropout=self.dropout,
                                act=self.act).to(self.device)


        checkpoint = torch.load('./train_model/guide/model.pth')
        self.model2.load_state_dict(checkpoint['state_dict'])

        self.model2.eval()

        # data = data.to(self.device)
        # s = to_dense_adj(data.edge_index).squeeze(0)
        x_, s_ = self.model2(data.x, data.adj, data.edge_index)
        score = self.loss_func(data.x, x_, data.adj, s_)

        outlier_scores = score.detach().squeeze(0).cpu().numpy()

        return outlier_scores


    def loss_func(self, x, x_, s, s_):
        # attribute reconstruction loss
        diff_attribute = torch.pow(x - x_, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        # structure reconstruction loss
        diff_structure = torch.pow(s - s_, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = self.alpha * attribute_errors + (1 - self.alpha) * \
                structure_errors
        return score

    def _get_nmf(self, G, cache_dir):
        """
        Calculation of Node Motif Degree / Graphlet Degree
        Distribution. Part of this function is adapted
        from https://github.com/benedekrozemberczki/OrbitalFeatures.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        cache_dir : str
            The directory for the node motif degree caching

        Returns
        -------
        s : torch.Tensor
            Structure matrix (node motif degree/graphlet degree)
        """
        hash_func = hashlib.sha1()
        hash_func.update(str(G).encode('utf-8'))
        file_name = 'nmd_' + str(hash_func.hexdigest()[:8]) + \
                    str(self.graphlet_size) + \
                    str(self.selected_motif)[0] + '.pt'
        file_path = os.path.join(cache_dir, file_name)
        if os.path.exists(file_path):
            s = torch.load(file_path)
        else:
            edge_index = G.edge_index
            g = nx.from_edgelist(edge_index.T.tolist())

            # create edge subsets
            edge_subsets = dict()
            subsets = [[edge[0], edge[1]] for edge in g.edges()]
            edge_subsets[2] = subsets
            unique_subsets = dict()
            for i in range(3, self.graphlet_size + 1):
                for subset in subsets:
                    for node in subset:
                        for neb in g.neighbors(node):
                            new_subset = subset + [neb]
                            if len(set(new_subset)) == i:
                                new_subset.sort()
                                unique_subsets[tuple(new_subset)] = 1
                subsets = [list(k) for k, v in unique_subsets.items()]
                edge_subsets[i] = subsets
                unique_subsets = dict()

            # enumerate graphs
            graphs = graph_atlas_g()
            interesting_graphs = {i: [] for i in
                                  range(2, self.graphlet_size + 1)}
            for graph in graphs:
                if 1 < graph.number_of_nodes() < self.graphlet_size + 1:
                    if nx.is_connected(graph):
                        interesting_graphs[graph.number_of_nodes()].append(
                            graph)

            # enumerate categories
            main_index = 0
            categories = dict()
            for size, graphs in interesting_graphs.items():
                categories[size] = dict()
                for index, graph in enumerate(graphs):
                    categories[size][index] = dict()
                    degrees = list(
                        set([graph.degree(node) for node in graph.nodes()]))
                    for degree in degrees:
                        categories[size][index][degree] = main_index
                        main_index += 1
            unique_motif_count = main_index

            # setup feature
            features = {node: {i: 0 for i in range(unique_motif_count)}
                        for node in g.nodes()}
            for size, node_lists in edge_subsets.items():
                graphs = interesting_graphs[size]
                for nodes in node_lists:
                    sub_gr = g.subgraph(nodes)
                    for index, graph in enumerate(graphs):
                        if nx.is_isomorphic(sub_gr, graph):
                            for node in sub_gr.nodes():
                                features[node][categories[size][index][
                                    sub_gr.degree(node)]] += 1
                            break

            motifs = [[n] + [features[n][i] for i in range(
                unique_motif_count)] for n in g.nodes()]
            motifs = torch.Tensor(motifs)
            motifs = motifs[torch.sort(motifs[:, 0]).indices, 1:]

            if self.selected_motif:
                # use motif selected in the original paper only
                s = torch.zeros((G.x.shape[0], 6))
                # m31
                s[:, 0] = motifs[:, 3]
                # m32
                s[:, 1] = motifs[:, 1] + motifs[:, 2]
                # m41
                s[:, 2] = motifs[:, 14]
                # m42
                s[:, 3] = motifs[:, 12] + motifs[:, 13]
                # m43
                s[:, 4] = motifs[:, 11]
                # node degree
                s[:, 5] = motifs[:, 0]
            else:
                # use graphlet degree
                s = motifs

            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save(s, file_path)

        return s


class GUIDE_Base(nn.Module):
    def __init__(self,
                 a_dim,
                 s_dim,
                 a_hid,
                 s_hid,
                 num_layers,
                 dropout,
                 act):
        super(GUIDE_Base, self).__init__()

        self.attr_ae = GCN(in_channels=a_dim,
                           hidden_channels=a_hid,
                           num_layers=num_layers,
                           out_channels=a_dim,
                           dropout=dropout,
                           act=act)

        self.struct_ae = GNA(in_channels=s_dim,
                             hidden_channels=s_hid,
                             num_layers=num_layers,
                             out_channels=s_dim,
                             dropout=dropout,
                             act=act)

    def forward(self, x, s, edge_index):
        x_ = self.attr_ae(x, edge_index)
        s_ = self.struct_ae(s, edge_index)
        return x_, s_


class GNA(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_layers,
                 out_channels,
                 dropout,
                 act):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GNAConv(in_channels, hidden_channels))
        for layer in range(num_layers - 2):
            self.layers.append(GNAConv(hidden_channels,
                                       hidden_channels))
        self.layers.append(GNAConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.act = act

    def forward(self, s, edge_index):
        for layer in self.layers:
            s = layer(s, edge_index)
            s = F.dropout(s, self.dropout, training=self.training)
            s = self.act(s)
        return s


class GNAConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__(aggr='add')
        self.w1 = torch.nn.Linear(in_channels, out_channels)
        self.w2 = torch.nn.Linear(in_channels, out_channels)
        self.a = nn.Parameter(torch.randn(out_channels, 1))

    def forward(self, s, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=s.size(0))
        out = self.propagate(edge_index, s=self.w2(s))
        return self.w1(s) + out

    def message(self, s_i, s_j, edge_index):
        alpha = (s_i - s_j) @ self.a
        alpha = softmax(alpha, edge_index[1], num_nodes=s_i.shape[0])
        return alpha * s_j



class GUIDE_eval(nn.Module):
    def __init__(self,
                 a_dim,
                 s_dim,
                 a_hid,
                 s_hid,
                 num_layers,
                 dropout,
                 act):
        super(GUIDE_eval, self).__init__()

        self.attr_ae = GCN(in_channels=a_dim,
                           hidden_channels=a_hid,
                           num_layers=num_layers,
                           out_channels=a_dim,
                           dropout=dropout,
                           act=act)

        self.struct_ae = GNA(in_channels=s_dim,
                             hidden_channels=s_hid,
                             num_layers=num_layers,
                             out_channels=s_dim,
                             dropout=dropout,
                             act=act)

    def forward(self, x, s, edge_index):
        x_ = self.attr_ae(x, edge_index)
        s_ = self.struct_ae(s, edge_index)
        return x_, s_


def save_checkpoint(state, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    best_file = os.path.join(outdir, 'model.pth')
    torch.save(state, best_file)