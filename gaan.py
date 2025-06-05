# -*- coding: utf-8 -*-
"""Generative Adversarial Attributed Network Anomaly Detection (GAAN)"""
# Author: Ruitong Zhang <rtzhang@buaa.edu.cn>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import os

from base import BaseDetector
from utility import validate_device
from metrics import eval_roc_auc


from typing import Any, Dict, List, Optional, Union, Callable
from torch.nn import BatchNorm1d, Identity, Linear, ModuleList
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(torch.nn.Module):
    r"""Multilayer Perceptron (MLP) model.
    Adapted from PyG for upward compatibility
    There exists two ways to instantiate an :class:`MLP`:
    1. By specifying explicit channel sizes, *e.g.*,
       .. code-block:: python
          mlp = MLP([16, 32, 64, 128])
       creates a three-layer MLP with **differently** sized hidden layers.
    1. By specifying fixed hidden channel sizes over a number of layers,
       *e.g.*,
       .. code-block:: python
          mlp = MLP(in_channels=16, hidden_channels=32,
                    out_channels=128, num_layers=3)
       creates a three-layer MLP with **equally** sized hidden layers.
    Args:
        channel_list (List[int] or int, optional): List of input, intermediate
            and output channels such that :obj:`len(channel_list) - 1` denotes
            the number of layers of the MLP (default: :obj:`None`)
        in_channels (int, optional): Size of each input sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        hidden_channels (int, optional): Size of each hidden sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        out_channels (int, optional): Size of each output sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        num_layers (int, optional): The number of layers.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        batch_norm (bool, optional): If set to :obj:`False`, will not make use
            of batch normalization. (default: :obj:`True`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        batch_norm_kwargs (Dict[str, Any], optional): Arguments passed to
            :class:`torch.nn.BatchNorm1d` in case :obj:`batch_norm == True`.
            (default: :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the module will not
            learn additive biases. (default: :obj:`True`)
        relu_first (bool, optional): Deprecated in favor of :obj:`act_first`.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: float = 0.,
        act: Callable = F.relu,
        batch_norm: bool = True,
        act_first: bool = False,
        batch_norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        relu_first: bool = False,
    ):
        super().__init__()

        act_first = act_first or relu_first  # Backward compatibility.
        batch_norm_kwargs = batch_norm_kwargs or {}

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            assert num_layers >= 1
            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.dropout = dropout
        self.act = act
        self.act_first = act_first

        self.lins = torch.nn.ModuleList()
        pairwise = zip(channel_list[:-1], channel_list[1:])
        for in_channels, out_channels in pairwise:
            self.lins.append(Linear(in_channels, out_channels, bias=bias))

        self.norms = torch.nn.ModuleList()
        for hidden_channels in channel_list[1:-1]:
            if batch_norm:
                norm = BatchNorm1d(hidden_channels, **batch_norm_kwargs)
            else:
                norm = Identity()
            self.norms.append(norm)

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            if self.act_first:
                x = self.act(x)
            x = norm(x)
            if not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'


class GAAN(BaseDetector):
    """
    GAAN (Generative Adversarial Attributed Network Anomaly
    Detection) is a generative adversarial attribute network anomaly
    detection framework, including a generator module, an encoder
    module, a discriminator module, and uses anomaly evaluation
    measures that consider sample reconstruction error and real sample
    recognition confidence to make predictions. This model is
    transductive only.

    See :cite:`chen2020generative` for details.

    Parameters
    ----------
    noise_dim :  int, optional
        Dimension of the Gaussian random noise. Defaults: ``16``.
    hid_dim :  int, optional
        Hidden dimension of MLP later 3. Defaults: ``64``.
    generator_layers : int, optional
        Number of layers in generator. Defaults: ``2``.
    encoder_layers : int, optional
        Number of layers in encoder. Defaults: ``2``.
    dropout : float, optional
        Dropout rate. Defaults: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Defaults: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Defaults: ``torch.nn.functional.relu``.
    alpha : float, optional
        Loss balance weight for attribute and structure. ``None`` for
        balancing by standard deviation. Default: ``None``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Defaults: ``0.05``.
    lr : float, optional
        Learning rate. Defaults: ``0.005``.
    epoch : int, optional
        Maximum number of training epoch. Defaults: ``10``.
    gpu : int
        GPU Index, -1 for using CPU. Defaults: ``-1``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Defaults: ``False``.

    Examples
    --------
    >>> from pygod.models import GAAN
    >>> model = GAAN()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(None)
    """

    def __init__(self,
                 noise_dim=16,
                 hid_dim=64,
                 generator_layers=4,
                 encoder_layers=4,
                 dropout=0.1,
                 weight_decay=0.01,
                 act=F.gelu,
                 alpha=0.,
                 contamination=0.1,
                 lr=0.0005,
                 epoch=500,
                 gpu=0,
                 verbose=True):
        super(GAAN, self).__init__(contamination=contamination)

        # model param
        self.noise_dim = noise_dim
        self.hid_dim = hid_dim
        self.generator_layers = generator_layers
        self.encoder_layers = encoder_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)
        # self.device = device

        # other param
        self.verbose = verbose
        self.model = None

    def fit(self, loader, in_feats, y_true=None):
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

        self.model = GAAN_Base(in_dim=self.in_feats,
                               noise_dim=self.noise_dim,
                               hid_dim=self.hid_dim,
                               generator_layers=self.generator_layers,
                               encoder_layers=self.encoder_layers,
                               dropout=self.dropout,
                               act=self.act).to(self.device)

        optimizer_ed = torch.optim.Adam(self.model.encoder.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)

        optimizer_g = torch.optim.Adam(self.model.generator.parameters(),
                                       lr=self.lr,
                                       weight_decay=self.weight_decay)

        self.model.train()
        # decision_scores = np.zeros(G.x.shape[0])
        for epoch in range(self.epoch):
            epoch_loss_g = 0
            epoch_loss_ed = 0
            t = 0
            # for sampled_data in loader:
            for data in loader:
                # data = data.to(self.device)

                # Generate noise for constructing fake attribute
                gaussian_noise = torch.randn(data.x.shape[0], self.noise_dim).to(self.device)

                # train the model
                x_, a, a_ = self.model(data.x, gaussian_noise, data.edge_index)

                loss_g = self._loss_func_g(a_[data.edge_index[0], data.edge_index[1]])
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

                loss_ed = self._loss_func_ed(a[data.edge_index[0], data.edge_index[1]],
                                             a_[data.edge_index[0], data.edge_index[1]]
                                             .detach())
                optimizer_ed.zero_grad()
                loss_ed.backward()
                optimizer_ed.step()
                t = t + 1


                epoch_loss_g += loss_g.item()
                epoch_loss_ed += loss_ed.item()
            del gaussian_noise
            gc.collect()


            if self.verbose:
                print("Epoch {:04d}: Loss G {:.4f} | Loss ED {:4f}"
                      .format(epoch, epoch_loss_g / t,
                              epoch_loss_ed / t), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_scores)
                    print(" | AUC {:.4f}".format(auc), end='')
                print()
        save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    }, os.path.join('./train_model/gaan/'))



        return self

    def decision_function(self, data):
        """
        Predict raw anomaly score using the fitted detector.
        Outliers are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        outlier_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        self.model2 = GAAN_Base(in_dim=self.in_feats,
                               noise_dim=self.noise_dim,
                               hid_dim=self.hid_dim,
                               generator_layers=self.generator_layers,
                               encoder_layers=self.encoder_layers,
                               dropout=self.dropout,
                               act=self.act).to(self.device)


        checkpoint = torch.load('./train_model/gaan/model.pth')
        self.model2.load_state_dict(checkpoint['state_dict'])

        self.model2.eval()

        # outlier_scores = torch.zeros(data.num_nodes).to(self.device) 

        # data = data.to(self.device)
        gaussian_noise = torch.randn(data.x.shape[0], self.noise_dim).to(self.device)

        x_, a, a_ = self.model(data.x, gaussian_noise, data.edge_index)

        score = self._score_func(data.x, x_, a, data.edge_index)

        outlier_scores = score.detach().squeeze(0).cpu().numpy()

        return outlier_scores

    def _loss_func_g(self, a_):
        loss_g = F.binary_cross_entropy(a_, torch.ones_like(a_))
        return loss_g

    def _loss_func_ed(self, a, a_):
        loss_r = F.binary_cross_entropy(a, torch.ones_like(a))
        loss_f = F.binary_cross_entropy(a_, torch.zeros_like(a_))
        return (loss_f + loss_r) / 2

    def _score_func(self, x, x_, a, edge_index):
        # attribute reconstruction loss
        diff_attribute = torch.pow(x - x_, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        adj = to_dense_adj(edge_index)[0]
        # structure reconstruction loss
        structure_errors = torch.sum(adj *
            F.binary_cross_entropy(a, torch.ones_like(a), reduction='none')
            , 1)

        score = self.alpha * attribute_errors + (
                1 - self.alpha) * structure_errors

        return score



class GAAN_Base(nn.Module):
    def __init__(self,
                 in_dim,
                 noise_dim,
                 hid_dim,
                 generator_layers,
                 encoder_layers,
                 dropout,
                 act):
        super(GAAN_Base, self).__init__()

        self.generator = MLP(in_channels=noise_dim,
                             hidden_channels=hid_dim,
                             out_channels=in_dim,
                             num_layers=generator_layers,
                             dropout=dropout,
                             act=act)

        self.encoder = MLP(in_channels=in_dim,
                           hidden_channels=hid_dim,
                           out_channels=hid_dim,
                           num_layers=encoder_layers,
                           dropout=dropout,
                           act=act)

    def forward(self, x, noise, edge_index):
        x_ = self.generator(noise)

        z = self.encoder(x)
        z_ = self.encoder(x_)

        a = torch.sigmoid((z @ z.T))
        a_ = torch.sigmoid((z_ @ z_.T))

        return x_, a, a_

def save_checkpoint(state, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    best_file = os.path.join(outdir, 'model.pth')
    torch.save(state, best_file)
