import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv, TransformerConv, global_mean_pool
from torch_geometric.utils import negative_sampling, degree
from torch.nn import ModuleList, Linear, Parameter
import numpy as np


class BilinearDecoder(nn.Module):
    """Optimized bilinear decoder for edge reconstruction"""

    def __init__(self, latent_dim):
        super().__init__()
        self.W = Parameter(torch.randn(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, z, edge_index):
        """Vectorized bilinear transformation"""
        z_i = z[edge_index[0]]
        z_j = z[edge_index[1]]
        # Optimized: z_i^T W z_j using batch matrix multiplication
        edge_logits = torch.sum(z_i * (z_j @ self.W), dim=1)
        return torch.sigmoid(edge_logits)


class MLPDecoder(nn.Module):
    """Optimized MLP decoder"""

    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, edge_index):
        z_cat = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        return torch.sigmoid(self.mlp(z_cat).squeeze())


class FastNegativeSampler:
    """Optimized negative sampling without hard mining for speed"""

    def __init__(self):
        pass

    def sample(self, edge_index, num_nodes, num_neg_samples):
        """Fast vectorized negative sampling"""
        return negative_sampling(
            edge_index, num_nodes,
            num_neg_samples=num_neg_samples,
            method='sparse'
        )


class OptimizedGNN(nn.Module):
    """Streamlined GNN without attention hooks"""

    def __init__(self, in_channels, hidden_channels, num_layers, gnn_type='gatv2', dropout=0.0):
        super().__init__()

        self.convs = ModuleList()
        self.norms = ModuleList()  # Add batch normalization for stability

        for i in range(num_layers):
            inc = in_channels if i == 0 else hidden_channels

            if gnn_type == 'gatv2':
                conv = GATv2Conv(inc, hidden_channels, dropout=dropout, add_self_loops=False)
            elif gnn_type == 'sage':
                conv = SAGEConv(inc, hidden_channels)
            elif gnn_type == 'transformer':
                conv = TransformerConv(inc, hidden_channels, dropout=dropout)
            else:
                raise ValueError(f'Unknown gnn_type: {gnn_type}')

            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ContrastiveLoss(nn.Module):
    """Optimized contrastive loss"""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, edge_index, negative_edge_index):
        # Vectorized similarity computation
        pos_sim = F.cosine_similarity(z[edge_index[0]], z[edge_index[1]]) / self.temperature
        neg_sim = F.cosine_similarity(z[negative_edge_index[0]], z[negative_edge_index[1]]) / self.temperature

        # Simplified InfoNCE loss
        pos_exp = torch.exp(pos_sim)
        neg_exp = torch.exp(neg_sim)

        # Use logsumexp for numerical stability
        neg_sum = torch.sum(neg_exp)
        pos_loss = -torch.log(pos_exp / (pos_exp + neg_sum))

        return pos_loss.mean()


class OptimizedGNNAnomalyVAE(nn.Module):
    """Streamlined VAE without complex features for maximum speed"""

    def __init__(self, in_dim, hidden_dim, latent_dim, num_layers=3, dropout=0.0,
                 alpha=0.5, gnn_type='gatv2', decoder_type='bilinear',
                 use_contrastive=False):
        super().__init__()

        # Simplified encoder
        self.encoder = OptimizedGNN(in_dim, hidden_dim, num_layers, gnn_type, dropout)

        # VAE components
        self.mu_layer = Linear(hidden_dim, latent_dim)
        self.logstd_layer = Linear(hidden_dim, latent_dim)

        # Simplified decoder
        self.decoder = Linear(latent_dim, in_dim)

        # Edge decoder
        if decoder_type == 'bilinear':
            self.edge_decoder = BilinearDecoder(latent_dim)
        elif decoder_type == 'mlp':
            self.edge_decoder = MLPDecoder(latent_dim)
        else:
            self.edge_decoder = None

        # Components
        self.neg_sampler = FastNegativeSampler()
        self.contrastive_loss = ContrastiveLoss() if use_contrastive else None

        self.alpha = alpha

    def encode(self, x, edge_index):
        """Encode to latent space"""
        h = self.encoder(x, edge_index)
        mu = self.mu_layer(h)
        logstd = self.logstd_layer(h).clamp(max=10)
        return mu, logstd

    def reparameterize(self, mu, logstd):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode_edges(self, z, edge_index):
        """Edge reconstruction"""
        if self.edge_decoder is not None:
            return self.edge_decoder(z, edge_index)
        else:
            # Simple inner product
            return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1))

    def forward(self, x, edge_index, batch=None):
        mu, logstd = self.encode(x, edge_index)
        z = self.reparameterize(mu, logstd)
        x_rec = self.decoder(z)

        return {
            'x_rec': x_rec,
            'z': z,
            'mu': mu,
            'logstd': logstd
        }

    def compute_losses(self, x, edge_index, results):
        """Compute all losses efficiently"""
        z = results['z']
        x_rec = results['x_rec']
        mu = results['mu']
        logstd = results['logstd']

        # Attribute reconstruction loss
        attr_loss = F.mse_loss(x_rec, x, reduction='mean')

        # Structure reconstruction loss
        pos_pred = self.decode_edges(z, edge_index)
        pos_loss = -torch.log(pos_pred + 1e-15).mean()

        # Negative sampling
        neg_edge_index = self.neg_sampler.sample(edge_index, x.size(0), edge_index.size(1))
        neg_pred = self.decode_edges(z, neg_edge_index)
        neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()

        struct_loss = pos_loss + neg_loss

        # KL divergence
        kl_loss = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=1))

        # Combined loss
        recon_loss = self.alpha * attr_loss + (1 - self.alpha) * struct_loss
        total_loss = recon_loss + kl_loss

        # Optional contrastive loss
        if self.contrastive_loss is not None:
            contrastive = self.contrastive_loss(z, edge_index, neg_edge_index)
            total_loss += 0.1 * contrastive

        return {
            'total_loss': total_loss,
            'attr_loss': attr_loss,
            'struct_loss': struct_loss,
            'kl_loss': kl_loss,
            'recon_loss': recon_loss
        }

    def anomaly_score(self, x, edge_index, batch=None):
        """Compute anomaly scores efficiently"""
        self.eval()
        with torch.no_grad():
            results = self.forward(x, edge_index, batch)

            # Attribute error
            attr_err = torch.mean((x - results['x_rec']).pow(2), dim=1)

            # Structure error (simplified)
            z = results['z']
            pos_pred = self.decode_edges(z, edge_index)

            # Aggregate structure error per node
            struct_err = torch.zeros(x.size(0), device=x.device)
            edge_errors = -torch.log(pos_pred + 1e-15)

            # Vectorized aggregation
            struct_err.scatter_add_(0, edge_index[0], edge_errors)
            struct_err.scatter_add_(0, edge_index[1], edge_errors)

            # Normalize by degree
            deg = degree(edge_index[0], num_nodes=x.size(0)) + degree(edge_index[1], num_nodes=x.size(0))
            deg = deg.clamp(min=1)
            struct_err = struct_err / deg

            # Combined score
            score = self.alpha * attr_err + (1 - self.alpha) * struct_err

            # Graph-level pooling if needed
            if batch is not None and batch.max() > 0:
                score = global_mean_pool(score, batch)

        return score


class FastGNNAnomalyDetector:
    """Optimized detector class for maximum speed"""

    def __init__(self, in_dim, hidden_dim=64, latent_dim=32, num_layers=3,
                 dropout=0.0, alpha=0.5, gnn_type='gatv2', device='cuda',
                 lr=1e-3, weight_decay=5e-4, epochs=200, decoder_type='bilinear',
                 use_contrastive=False):

        self.model = OptimizedGNNAnomalyVAE(
            in_dim, hidden_dim, latent_dim, num_layers, dropout, alpha,
            gnn_type, decoder_type, use_contrastive
        ).to(device)

        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

        # Use AdamW for better performance
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

    def fit(self, train_loader):
        """Optimized training loop"""
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0

            for data in train_loader:
                data = data.to(self.device)

                # Forward pass
                results = self.model(data.x, data.edge_index, getattr(data, 'batch', None))

                # Compute losses
                losses = self.model.compute_losses(data.x, data.edge_index, results)

                # Backward pass
                self.optimizer.zero_grad()
                losses['total_loss'].backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += losses['total_loss'].item()
                num_batches += 1

            self.scheduler.step()

            # Logging every 50 epochs
            if epoch % 50 == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | LR: {lr:.6f}')

    def predict(self, data):
        """Fast prediction without interpretability features"""
        data = data.to(self.device)
        return self.model.anomaly_score(data.x, data.edge_index, getattr(data, 'batch', None))

    def decision_function(self, data):
        """Compatibility method"""
        with torch.no_grad():
            scores = self.predict(data)
            return scores.cpu().numpy()
