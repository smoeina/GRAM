# -*- coding: utf-8 -*-
""" One-Class Graph Neural Networks for Anomaly Detection in Attributed Networks
"""
# Author: Xueying Ding <xding2@andrew.cmu.edu>
# License: BSD 2 clause

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.utils.validation import check_is_fitted
from torch_sparse import SparseTensor

from base import BaseDetector
from utility import validate_device
from metrics import eval_roc_auc
# from torch_geometric.loader import NeighborLoader
import os


class GCN_base(nn.Module):
    """
    Describe: Backbone GCN module.
    """

    def __init__(self, in_feats, n_hidden, n_layers, dropout, act):
        super(GCN_base, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv(in_feats, n_hidden, bias=False))
        # hidden layers
        for i in range(n_layers):
            self.layers.append(GCNConv(n_hidden, n_hidden, bias=False))
        self.dropout = nn.Dropout(p=dropout)
        self.act = act

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[0:-1]):
            x = layer(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x


class OCGNN(BaseDetector):
    """
    OCGNN (One-Class Graph Neural Networks for Anomaly Detection in
    Attributed Networks) is an anomaly detector that measures the
    distance of anomaly to the centroid, in a similar fashion to the
    support vector machine, but in the embedding space after feeding
    towards several layers of GCN.

    See :cite:`wang2021one` for details.

    Parameters
    ----------
    n_hidden :  int, optional
        Hidden dimension of model. Defaults: `256``.
    n_layers : int, optional
        Dimensions of underlying GCN. Defaults: ``4``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    dropout : float, optional
        Dropout rate. Defaults: ``0.3``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Defaults: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Defaults: ``torch.nn.functional.relu``.
    eps : float, optional
        A small valid number for determining the center and make
        sure it does not collapse to 0. Defaults: ``0.001``.
    nu: float, optional
        Regularization parameter. Defaults: ``0.5`` 
    lr : float, optional
        Learning rate. Defaults: ``0.005``.
    epoch : int, optional
        Maximum number of training epoch. Defaults: ``5``.
    warmup_epoch : int, optional
        Number of epochs to update radius and center in the beginning 
        of training. Defaults: ``2``.
    gpu : int
        GPU Index, -1 for using CPU. Defaults: ``0``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Defaults: ``False``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.

    Examples
    --------
    >>> from pygod.models import AnomalyDAE
    >>> model = OCGNN()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 n_hidden=64,
                 n_layers=4,
                 contamination=0.1,
                 dropout=0.2,
                 lr=5e-4,
                 weight_decay=0,
                 eps=0.001,
                 nu=0.5,
                 gpu=0,
                 epoch=100,
                 warmup_epoch=5,
                 verbose=False,
                 act=F.gelu):
        super(OCGNN, self).__init__(contamination=contamination)
        self.dropout = dropout
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.nu = nu
        self.data_center = 0
        self.radius = 0.0
        self.epoch = epoch
        self.warmup_epoch = warmup_epoch
        self.act = act
        self.device = validate_device(gpu)
        # self.batch_size = batch_size
        # self.num_neigh = num_neigh

        # other param
        self.verbose = verbose
        self.model = None

    def init_center(self, x, edge_index):
        """
        Initialize hypersphere center c as the mean from
        an initial forward pass on the data.
  
        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge indices for the graph data

        Returns
        ----------
        c : torch.Tensor
            The new centroid.
           """
        n_samples = 0
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x, edge_index)
            # get the inputs of the batch
            n_samples = outputs.shape[0]
            c = torch.sum(outputs, dim=0).to(self.device)
        # print(outputs)
        c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be
        # trivially matched with zero weights.
        c[(abs(c) < self.eps) & (c < 0)] = -self.eps
        c[(abs(c) < self.eps) & (c > 0)] = self.eps
        return c

    def get_radius(self, dist):
        """
        Optimally solve for radius R via the (1-nu)-quantile of distances.
        
        Parameters
        ----------
        dist : torch.Tensor
            Distance of the data points, calculated by the loss function.
       
        Returns
        ----------
        r : numpy.array
            New radius.
        """
        radius = np.quantile(np.sqrt(dist.clone().data.cpu().numpy()),
                             1 - self.nu)
        return radius

    def anomaly_scores(self, outputs):
        """
        Calculate the anomaly score given by Euclidean distance to the center.
        
        Parameters
        ----------
        outputs : torch.Tensor
            The output in the reduced space by GCN.

        Returns
        ----------
        dist : torch.Tensor
            Average distance.
        scores : torch.Tensor
            Anomaly scores.
        """
        dist = torch.sum((outputs - self.data_center) ** 2, dim=1)
        scores = dist - self.radius ** 2
        return dist, scores

    def loss_function(self, outputs, update=False):
        """
        Calculate the loss in paper Equation (4)
        
        Parameters
        ----------
        outputs : torch.Tensor
            The output in the reduced space by GCN.
        update : bool, optional (default=False)
            If you need to update the radius, set update=True.

        Returns
        ----------
        dist : torch.Tensor
            Average distance.
        scores : torch.Tensor
            Anomaly scores.
        loss : torch.Tensor
            A combined loss of radius and average scores.
        """

        dist, scores = self.anomaly_scores(outputs)
        loss = self.radius ** 2 + (1 / self.nu) * torch.mean(
            torch.max(torch.zeros_like(scores), scores))
        if update:
            self.radius = torch.tensor(self.get_radius(dist), device = self.device)
        return loss, dist, scores

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

        # initialize the model and optimizer
        self.model = GCN_base(self.in_feats,
                              self.n_hidden,
                              self.n_layers,
                              self.dropout,
                              self.act)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)

        self.data_center = torch.zeros(self.n_hidden, device=self.device)
        self.radius = torch.tensor(0, device=self.device)

        self.model = self.model.to(self.device)
        # training the model
        self.model.train()
        
        for cur_epoch in range(self.epoch):
            epoch_loss = 0
            t = 0
            for data in loader:

                outputs = self.model(data.x, data.edge_index)
                loss, dist, score = self.loss_function(outputs)
                epoch_loss += loss.item()
                if self.warmup_epoch is not None and cur_epoch < self.warmup_epoch:
                    self.data_center = self.init_center(data.x, data.edge_index)
                    self.radius = torch.tensor(self.get_radius(dist),
                                           device=self.device)
                
                # decision_scores[node_idx[:batch_size]] = score.detach()\
                #                                               .cpu().numpy()
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                t = t + 1


            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(cur_epoch,  epoch_loss / t), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_scores)
                    print(" | AUC {:.4f}".format(auc), end='')
                print()
            save_checkpoint({
                    'epoch': cur_epoch,
                    'state_dict': self.model.state_dict(),
                    }, os.path.join('./train_model/ocgnn/'))

        # self.decision_scores_ = decision_scores
        # self._process_decision_scores()
        return self


    def decision_function(self, data):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on distance 
        to the centroid and measurement within the radius
        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        
        Returns
        -------
        anomaly_scores : numpy.array
            The anomaly score of the input samples of shape (n_samples,).
        """
        # self.model2 = DOMINANT_eval(in_dim=self.in_feats,
        #                            hid_dim=self.hid_dim,
        #                            num_layers=self.num_layers,
        #                            dropout=self.dropout,
        #                            act=self.act).to(self.device)


        # checkpoint = torch.load('./train_model/ocgnn/model.pth')
        # self.model2.load_state_dict(checkpoint['state_dict'])

        self.model.eval()

        outputs = self.model(data.x, data.edge_index)
        loss, dist, score = self.loss_function(outputs)

        outlier_scores = score.detach().squeeze(0).cpu().numpy()

        return outlier_scores

def save_checkpoint(state, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    best_file = os.path.join(outdir, 'model.pth')
    torch.save(state, best_file)
