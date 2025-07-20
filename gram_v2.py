import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv, TransformerConv, global_mean_pool
from torch_geometric.utils import negative_sampling
from torch.nn import ModuleList, Linear
from typing import Optional, Callable, Union

class FlexibleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, gnn_type='gatv2', dropout=0.0, act=F.gelu):
        super().__init__()
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.act = act
        self.convs = ModuleList()
        for i in range(num_layers):
            inc = in_channels if i == 0 else hidden_channels
            if gnn_type == 'gatv2':
                self.convs.append(GATv2Conv(inc, hidden_channels))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(inc, hidden_channels))
            elif gnn_type == 'transformer':
                self.convs.append(TransformerConv(inc, hidden_channels))
            else:
                raise ValueError(f'Unknown gnn_type: {gnn_type}')

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GNNV2Anomaly(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, num_layers=4, dropout=0.0, act=F.gelu, alpha=0.5, gnn_type='gatv2', pooling='mean', score_mode='hybrid'):
        super().__init__()
        self.encoder = FlexibleGNN(in_dim, hidden_dim, num_layers, gnn_type, dropout, act)
        self.mu_layer = Linear(hidden_dim, latent_dim)
        self.logstd_layer = Linear(hidden_dim, latent_dim)
        self.decoder_attr = FlexibleGNN(latent_dim, hidden_dim, num_layers, gnn_type, dropout, act)
        self.out_layer = Linear(hidden_dim, in_dim)
        self.alpha = alpha
        self.pooling = pooling
        self.score_mode = score_mode

    def forward(self, x, edge_index, batch=None):
        h = self.encoder(x, edge_index)
        mu = self.mu_layer(h)
        logstd = self.logstd_layer(h).clamp(max=10)
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        z = mu + eps * std
        h_dec = self.decoder_attr(z, edge_index)
        x_rec = self.out_layer(h_dec)
        return x_rec, z, mu, logstd

    def recon_loss(self, z, edge_index):
        EPS = 1e-15
        pos_value = torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1))
        pos_loss = -torch.log(pos_value + EPS)
        neg_edge_index = negative_sampling(edge_index, z.size(0), num_neg_samples=edge_index.size(1))
        neg_value = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))
        neg_loss = -torch.log(1 - neg_value + EPS)
        return pos_loss, neg_loss, neg_edge_index

    def kl_loss(self, mu, logstd):
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def anomaly_score(self, x, x_rec, z, edge_index, batch=None):
        # Attribute error per node (mean squared error)
        attr_err = torch.sqrt(torch.mean((x - x_rec) ** 2, dim=1)).clamp(max=10)
        # Structure error per node (normalized by degree)
        pos_loss, neg_loss, neg_edge_index = self.recon_loss(z, edge_index)
        struct_err = torch.zeros(x.size(0), device=x.device)
        deg = torch.zeros(x.size(0), device=x.device)
        for idx in range(edge_index.size(1)):
            struct_err[edge_index[0, idx]] += pos_loss[idx]
            struct_err[edge_index[1, idx]] += pos_loss[idx]
            deg[edge_index[0, idx]] += 1
            deg[edge_index[1, idx]] += 1
        for idx in range(neg_loss.size(0)):
            struct_err[neg_edge_index[0, idx]] += neg_loss[idx]
            struct_err[neg_edge_index[1, idx]] += neg_loss[idx]
            deg[neg_edge_index[0, idx]] += 1
            deg[neg_edge_index[1, idx]] += 1
        deg = deg.clamp(min=1)
        struct_err = (struct_err / deg).clamp(max=10)
        # Combine errors
        if self.score_mode == 'hybrid':
            score = self.alpha * attr_err + (1 - self.alpha) * struct_err
        elif self.score_mode == 'attr':
            score = attr_err
        elif self.score_mode == 'struct':
            score = struct_err
        else:
            score = attr_err
        # Only pool for graph-level scores
        if batch is not None and batch.max() > 0:
            score = global_mean_pool(score, batch)
        return score

class GNNVariantAnomalyDetector:
    def __init__(self, in_dim, hidden_dim=128, latent_dim=64, num_layers=4, dropout=0.0, alpha=0.5, gnn_type='gatv2', device='cpu', lr=1e-3, weight_decay=5e-4, epochs=300, pooling='mean', score_mode='hybrid'):
        self.model = GNNV2Anomaly(in_dim, hidden_dim, latent_dim, num_layers, dropout, F.gelu, alpha, gnn_type, pooling, score_mode).to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

    def fit(self, train_loader):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(self.epochs):
            total_loss = 0
            for data in train_loader:
                data = data.to(self.device)
                x_rec, z, mu, logstd = self.model(data.x, data.edge_index, getattr(data, 'batch', None))
                attr_err = torch.mean(torch.sqrt(torch.mean((data.x - x_rec) ** 2, dim=1)).clamp(max=10))
                pos_loss, neg_loss, neg_edge_index = self.model.recon_loss(z, data.edge_index)
                # Structure error normalized by degree
                struct_err = torch.zeros(data.x.size(0), device=data.x.device)
                deg = torch.zeros(data.x.size(0), device=data.x.device)
                for idx in range(data.edge_index.size(1)):
                    struct_err[data.edge_index[0, idx]] += pos_loss[idx]
                    struct_err[data.edge_index[1, idx]] += pos_loss[idx]
                    deg[data.edge_index[0, idx]] += 1
                    deg[data.edge_index[1, idx]] += 1
                for idx in range(neg_loss.size(0)):
                    struct_err[neg_edge_index[0, idx]] += neg_loss[idx]
                    struct_err[neg_edge_index[1, idx]] += neg_loss[idx]
                    deg[neg_edge_index[0, idx]] += 1
                    deg[neg_edge_index[1, idx]] += 1
                deg = deg.clamp(min=1)
                struct_err = torch.mean((struct_err / deg).clamp(max=10))
                kl = self.model.kl_loss(mu, logstd)
                loss = self.model.alpha * attr_err + (1 - self.model.alpha) * struct_err + kl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f}')

    def decision_function(self, data, return_node_scores=False):
        self.model.eval()
        with torch.no_grad():
            x_rec, z, _, _ = self.model(data.x, data.edge_index, getattr(data, 'batch', None))
            score = self.model.anomaly_score(data.x, x_rec, z, data.edge_index, getattr(data, 'batch', None))
            return score.cpu().numpy()
