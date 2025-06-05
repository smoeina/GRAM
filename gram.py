import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import  negative_sampling
from torch_geometric.nn import GCNConv
import os

from base import BaseDetector
from utility import validate_device
from metrics import eval_roc_auc


from typing import List, Optional, Union, Callable
from torch.nn import Linear, ModuleList
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge



class GCN(torch.nn.Module):
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



class GRAM(BaseDetector):
    def __init__(self,
                 hid_dim=128,
                 latent_size=64,
                 num_layers=8,
                 dropout=0.0,
                 weight_decay=0.,
                 act=F.gelu,
                 alpha=0.25,
                 contamination=0.00001,
                 lr=5e-4,
                 epoch=700,
                 gpu=0,
                 verbose=True):
        super(GRAM, self).__init__(contamination=contamination)

        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha

        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)

        self.verbose = verbose
        self.model = None

    def fit(self, loader, in_feats, y_true=None):
        self.in_feats = in_feats

        self.model = GRAM_Base(in_dim=self.in_feats,
                                   hid_dim=self.hid_dim,
                                   latent_size=self.latent_size,
                                   num_layers=self.num_layers,
                                   dropout=self.dropout,
                                   act=self.act).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        self.model.train()
        min_loss = 100
        for epoch in range(self.epoch):
            epoch_loss = 0     
            t = 0
            for data in loader:  
                x_, z = self.model(data.x, data.edge_index)
                loss, loss_kl = self.loss_func(data.x, x_, z, data.edge_index)
                epoch_loss += loss_kl.item()


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t = t + 1
            if min_loss > epoch_loss/t:
                min_loss = epoch_loss/t
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }, os.path.join('./train_model/gram/PTC/'))
                            
            if self.verbose:
                print("Epoch {:04d}: Loss {:.6f}"
                      .format(epoch, epoch_loss/t), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_scores)
                    print(" | AUC {:.6f}".format(auc), end='')
                print()

        return self



    def gradcam(self, data):
        self.model2 = GRAM_ad(in_dim=self.in_feats,
                                   hid_dim=self.hid_dim,
                                   latent_size=self.latent_size,
                                   num_layers=self.num_layers,
                                   dropout=self.dropout,
                                   act=self.act).to(self.device)

        checkpoint = torch.load('./train_model/gram/PTC/model.pth')
        self.model2.load_state_dict(checkpoint['state_dict'])

        self.model2.eval()
        outlier_scores = torch.zeros(data.num_nodes).to(self.device)    
        # x = torch.ones(data.num_nodes, self.in_feats).to(self.device)             
        data.x.requires_grad_(True)
        z_latent = self.model2(data.x, data.edge_index)

        for i in range(z_latent.shape[0]):
            z_latent[i].backward(retain_graph=True)
            a = torch.mean(self.model2.h_grads, dim=0).unsqueeze(0)
            outlier_scores = outlier_scores + torch.sum(F.gelu((torch.mul(self.model2.h, a))), dim=1)

        return outlier_scores.detach().cpu().numpy()



    def loss_func(self, x, x_, z, edge_index):
        diff_attribute = torch.pow(x - x_, 2)
        attribute_errors = torch.mean(torch.sqrt(torch.sum(diff_attribute, 1)))
        diff_structure = self.model.recon_loss(z, edge_index)
        structure_errors = diff_structure
        loss_kl = self.model.kl_loss()
        loss = self.alpha * attribute_errors + (1 - self.alpha) * structure_errors + loss_kl
        return loss, diff_structure


class GRAM_Base(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 latent_size,
                 num_layers,
                 dropout,
                 act):
        super(GRAM_Base, self).__init__()

        encoder_layers = int(num_layers / 2)
        decoder_layers = num_layers - encoder_layers
        self.shared_encoder = GCN(in_channels=in_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=encoder_layers,
                                  out_channels=hid_dim,
                                  dropout=dropout,
                                  act=act)
        self.encode_liner1 = ModuleList()
        self.encode_liner1.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))
        self.encode_liner1.append(nn.GELU())
        self.encode_liner1.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))
        self.encode_liner1.append(nn.GELU())
        self.encode_liner1.append(nn.Linear(in_features=hid_dim, out_features=latent_size))
  
        self.encode_liner2 = ModuleList()
        self.encode_liner2.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))
        self.encode_liner2.append(nn.GELU())
        self.encode_liner2.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))
        self.encode_liner2.append(nn.GELU())
        self.encode_liner2.append(nn.Linear(in_features=hid_dim, out_features=latent_size))

        self.decode_liner1 = ModuleList()
        self.decode_liner1.append(nn.Linear(in_features=latent_size, out_features=hid_dim))
        self.decode_liner1.append(nn.GELU())
        self.decode_liner1.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))
        self.decode_liner1.append(nn.GELU())
        self.decode_liner1.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))

        self.decode_liner2 = ModuleList()
        self.decode_liner2.append(nn.Linear(in_features=latent_size, out_features=hid_dim))
        self.decode_liner2.append(nn.GELU())
        self.decode_liner2.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))
        self.decode_liner2.append(nn.GELU())
        self.decode_liner2.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))

        self.attr_decoder = GCN(in_channels=hid_dim,
                                hidden_channels=hid_dim,
                                num_layers=decoder_layers,
                                out_channels=in_dim,
                                dropout=dropout,
                                act=act)

        self.struct_decoder = GCN(in_channels=hid_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=decoder_layers - 1,
                                  out_channels=in_dim,
                                  dropout=dropout,
                                  act=act)

    def forward(self, x, edge_index):
        h = self.shared_encoder(x, edge_index)
        h1 = h
        for layer in self.encode_liner1:
            h1 = layer(h1)
        self.mu = h1
        h2 = h
        for layer in self.encode_liner2:
            h2 = layer(h2)
        self.logstd = h2

        self.logstd = self.logstd.clamp(max=10)
        z = self.mu + torch.randn_like(self.logstd) * torch.exp(self.logstd)

        z1 = z
        for layer in self.decode_liner1:
            z1 = layer(z1)
        z_ = z1

        x_ = self.attr_decoder(z_, edge_index)

        z2 = z
        for layer in self.decode_liner2:
            z2 = layer(z2)
        z_e = self.struct_decoder(z2, edge_index)

        
        return x_, z_e

    def recon_loss(self, z, pos_edge_index,
                   neg_edge_index = None):
        EPS = 1e-15
        pos_value = torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1))
        pos_loss = -torch.log(pos_value+ EPS).mean()


        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        
        neg_value = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))
        neg_loss = -torch.log(1 - neg_value + EPS).mean()
        return pos_loss + neg_loss

    def kl_loss(self, mu = None,
                logstd = None):
        mu = self.mu if mu is None else mu
        logstd = self.logstd if logstd is None else logstd.clamp(
            max=10)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))



class GRAM_ad(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 latent_size,
                 num_layers,
                 dropout,
                 act):
        super(GRAM_ad, self).__init__()

        encoder_layers = int(num_layers / 2)
        decoder_layers = num_layers - encoder_layers

        self.shared_encoder = GCN(in_channels=in_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=encoder_layers,
                                  out_channels=hid_dim,
                                  dropout=dropout,
                                  act=act)
        self.encode_liner1 = ModuleList()
        self.encode_liner1.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))
        self.encode_liner1.append(nn.GELU())
        self.encode_liner1.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))
        self.encode_liner1.append(nn.GELU())
        self.encode_liner1.append(nn.Linear(in_features=hid_dim, out_features=latent_size))
   
        self.encode_liner2 = ModuleList()
        self.encode_liner2.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))
        self.encode_liner2.append(nn.GELU())
        self.encode_liner2.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))
        self.encode_liner2.append(nn.GELU())
        self.encode_liner2.append(nn.Linear(in_features=hid_dim, out_features=latent_size))

        self.decode_liner1 = ModuleList()
        self.decode_liner1.append(nn.Linear(in_features=latent_size, out_features=hid_dim))
        self.decode_liner1.append(nn.GELU())
        self.decode_liner1.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))
        self.decode_liner1.append(nn.GELU())
        self.decode_liner1.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))

        self.decode_liner2 = ModuleList()
        self.decode_liner2.append(nn.Linear(in_features=latent_size, out_features=hid_dim))
        self.decode_liner2.append(nn.GELU())
        self.decode_liner2.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))
        self.decode_liner2.append(nn.GELU())
        self.decode_liner2.append(nn.Linear(in_features=hid_dim, out_features=hid_dim))

        self.attr_decoder = GCN(in_channels=hid_dim,
                                hidden_channels=hid_dim,
                                num_layers=decoder_layers,
                                out_channels=in_dim,
                                dropout=dropout,
                                act=act)

        self.struct_decoder = GCN(in_channels=hid_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=decoder_layers - 1,
                                  out_channels=in_dim,
                                  dropout=dropout,
                                  act=act)

    def activations_hook(self, grad):
        self.h_grads = grad

    def forward(self, x, edge_index):
        with torch.enable_grad():
            self.h = self.shared_encoder(x, edge_index)
        self.h.register_hook(self.activations_hook)
        h1 = self.h
        for layer in self.encode_liner1:
            h1 = layer(h1)
        self.mu = h1

        h2 = self.h
        for layer in self.encode_liner2:
            h2 = layer(h2)
        self.logstd = h2

        self.logstd = self.logstd.clamp(max=10)


        z = self.mu + torch.randn_like(self.logstd) * torch.exp(self.logstd)
        z = torch.sum(z, dim=0)

        return z

def save_checkpoint(state, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    best_file = os.path.join(outdir, 'model.pth')
    torch.save(state, best_file)
