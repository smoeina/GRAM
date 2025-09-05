import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv, TransformerConv, global_mean_pool, SAGPooling
from torch_geometric.utils import negative_sampling, add_self_loops, degree
from torch.nn import ModuleList, Linear, Parameter
from typing import Optional, Callable, Union
import numpy as np
from collections import defaultdict


class AttentionVisualizationMixin:
    """Mixin for attention visualization and interpretability"""

    def __init__(self):
        self.attention_weights = {}
        self.gradients = {}
        self.activations = {}

    def register_hooks(self):
        """Register hooks for gradient and activation capture"""

        def save_gradient(name):
            def hook(grad):
                self.gradients[name] = grad.detach()

            return hook

        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()

            return hook

        # Register hooks for each layer
        for i, conv in enumerate(self.convs):
            conv.register_forward_hook(save_activation(f'conv_{i}'))

    def get_attention_weights(self, layer_idx):
        """Extract attention weights from GAT/Transformer layers"""
        if hasattr(self.convs[layer_idx], 'attention'):
            return self.convs[layer_idx].attention
        return None

    def compute_grad_cam(self, x, edge_index, target_nodes=None):
        """Compute Grad-CAM for node-level interpretability"""
        self.eval()
        x.requires_grad_(True)

        # Forward pass
        h = self.encoder(x, edge_index)

        if target_nodes is None:
            target_nodes = torch.arange(x.size(0))

        # Compute gradients
        grad_outputs = torch.zeros_like(h)
        grad_outputs[target_nodes] = 1.0

        gradients = torch.autograd.grad(
            outputs=h,
            inputs=x,
            grad_outputs=grad_outputs,
            retain_graph=True
        )[0]

        # Compute importance scores
        importance = torch.abs(gradients).mean(dim=1)
        return importance


class BilinearDecoder(nn.Module):
    """Bilinear decoder for better edge reconstruction"""

    def __init__(self, latent_dim):
        super().__init__()
        self.W = Parameter(torch.randn(latent_dim, latent_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, z, edge_index):
        """Compute edge probabilities using bilinear transformation"""
        z_i = z[edge_index[0]]  # Source nodes
        z_j = z[edge_index[1]]  # Target nodes

        # Bilinear transformation: z_i^T W z_j
        edge_logits = torch.sum(z_i * torch.matmul(z_j, self.W), dim=1)
        return torch.sigmoid(edge_logits)


class MLPDecoder(nn.Module):
    """MLP decoder for edge reconstruction"""

    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z, edge_index):
        z_i = z[edge_index[0]]
        z_j = z[edge_index[1]]
        edge_features = torch.cat([z_i, z_j], dim=1)
        return self.mlp(edge_features).squeeze()


class HardNegativeSampler:
    """Smart negative sampling with hard negative mining"""

    def __init__(self, k=5):
        self.k = k  # Number of hardest negatives to keep

    def sample(self, edge_index, z, num_neg_samples):
        """Sample negative edges with focus on hard negatives"""
        # Standard negative sampling
        neg_edge_index = negative_sampling(
            edge_index, z.size(0),
            num_neg_samples=num_neg_samples * 2  # Sample more initially
        )

        # Compute negative edge scores
        z_i = z[neg_edge_index[0]]
        z_j = z[neg_edge_index[1]]
        neg_scores = torch.sigmoid((z_i * z_j).sum(dim=1))

        # Keep hardest negatives (highest scores = hardest to distinguish)
        _, hard_indices = torch.topk(neg_scores, min(num_neg_samples, len(neg_scores)))

        return neg_edge_index[:, hard_indices]


class FlexibleGNN(torch.nn.Module, AttentionVisualizationMixin):
    def __init__(self, in_channels, hidden_channels, num_layers, gnn_type='gatv2', dropout=0.0, act=F.gelu):
        super().__init__()
        AttentionVisualizationMixin.__init__(self)

        self.gnn_type = gnn_type
        self.dropout = dropout
        self.act = act
        self.convs = ModuleList()

        for i in range(num_layers):
            inc = in_channels if i == 0 else hidden_channels
            if gnn_type == 'gatv2':
                self.convs.append(GATv2Conv(inc, hidden_channels, dropout=dropout))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(inc, hidden_channels))
            elif gnn_type == 'transformer':
                self.convs.append(TransformerConv(inc, hidden_channels, dropout=dropout))
            else:
                raise ValueError(f'Unknown gnn_type: {gnn_type}')

        self.register_hooks()

    def forward(self, x, edge_index, return_attention=False):
        attention_weights = []

        for i, conv in enumerate(self.convs):
            if return_attention and hasattr(conv, 'attention'):
                # Store attention before applying activation
                x = conv(x, edge_index)
                if hasattr(conv, '_alpha'):  # GATv2Conv stores attention in _alpha
                    attention_weights.append(conv._alpha)
            else:
                x = conv(x, edge_index)

            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            return x, attention_weights
        return x


class MultiScaleEncoder(nn.Module):
    """Multi-scale encoder with hierarchical pooling"""

    def __init__(self, in_channels, hidden_channels, num_layers, gnn_type='gatv2', dropout=0.0):
        super().__init__()
        self.gnn = FlexibleGNN(in_channels, hidden_channels, num_layers, gnn_type, dropout)
        self.pool = SAGPooling(hidden_channels, ratio=0.8)
        self.global_pool = global_mean_pool

    def forward(self, x, edge_index, batch=None):
        # Node-level embeddings
        node_emb = self.gnn(x, edge_index)

        # Graph-level embeddings through pooling
        if batch is not None:
            # Hierarchical pooling
            pooled_x, pooled_edge_index, _, pooled_batch, _, _ = self.pool(
                node_emb, edge_index, batch=batch
            )
            graph_emb = self.global_pool(pooled_x, pooled_batch)

            # Broadcast graph embedding back to nodes
            graph_emb_expanded = graph_emb[batch]

            # Combine node and graph level information
            multi_scale_emb = torch.cat([node_emb, graph_emb_expanded], dim=1)
        else:
            multi_scale_emb = node_emb

        return multi_scale_emb


class ContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss for latent space"""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, edge_index, negative_edge_index):
        # Positive pairs
        z_i_pos = z[edge_index[0]]
        z_j_pos = z[edge_index[1]]
        pos_sim = F.cosine_similarity(z_i_pos, z_j_pos) / self.temperature

        # Negative pairs
        z_i_neg = z[negative_edge_index[0]]
        z_j_neg = z[negative_edge_index[1]]
        neg_sim = F.cosine_similarity(z_i_neg, z_j_neg) / self.temperature

        # InfoNCE loss
        pos_loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim).sum()))

        return pos_loss.mean()


class AdaptiveWeighting(nn.Module):
    """Adaptive weighting between attribute and structure losses"""

    def __init__(self, initial_alpha=0.5):
        super().__init__()
        self.alpha = Parameter(torch.tensor(initial_alpha))

    def forward(self, attr_loss, struct_loss):
        # Ensure alpha is in [0, 1]
        alpha = torch.sigmoid(self.alpha)
        return alpha * attr_loss + (1 - alpha) * struct_loss, alpha


class GNNV2Anomaly(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, num_layers=4, dropout=0.0,
                 act=F.gelu, alpha=0.5, gnn_type='gatv2', pooling='mean',
                 score_mode='hybrid', decoder_type='bilinear', use_contrastive=True,
                 use_adaptive_alpha=True, use_multi_scale=True):
        super().__init__()

        # Encoder setup
        if use_multi_scale:
            self.encoder = MultiScaleEncoder(in_dim, hidden_dim, num_layers, gnn_type, dropout)
            encoder_out_dim = hidden_dim * 2  # Node + graph embeddings
        else:
            self.encoder = FlexibleGNN(in_dim, hidden_dim, num_layers, gnn_type, dropout, act)
            encoder_out_dim = hidden_dim

        # VAE components with attention
        self.mu_layer = Linear(encoder_out_dim, latent_dim)
        self.logstd_layer = Linear(encoder_out_dim, latent_dim)

        # Two-stage decoder
        self.decoder_attr = FlexibleGNN(latent_dim, hidden_dim, num_layers, gnn_type, dropout, act)
        self.out_layer = Linear(hidden_dim, in_dim)

        # Edge decoder
        if decoder_type == 'bilinear':
            self.edge_decoder = BilinearDecoder(latent_dim)
        elif decoder_type == 'mlp':
            self.edge_decoder = MLPDecoder(latent_dim)
        else:
            self.edge_decoder = None  # Use inner product

        # Auxiliary components
        self.hard_neg_sampler = HardNegativeSampler()

        if use_contrastive:
            self.contrastive_loss = ContrastiveLoss()
        else:
            self.contrastive_loss = None

        if use_adaptive_alpha:
            self.adaptive_alpha = AdaptiveWeighting(alpha)
        else:
            self.adaptive_alpha = None
            self.alpha = alpha

        # Auxiliary task: degree prediction
        self.degree_predictor = Linear(latent_dim, 1)

        self.pooling = pooling
        self.score_mode = score_mode
        self.use_multi_scale = use_multi_scale

    def forward(self, x, edge_index, batch=None, return_attention=False):
        # Encode
        if self.use_multi_scale:
            h = self.encoder(x, edge_index, batch)
        else:
            if return_attention:
                h, attention_weights = self.encoder(x, edge_index, return_attention=True)
            else:
                h = self.encoder(x, edge_index)

        # VAE latent space
        mu = self.mu_layer(h)
        logstd = self.logstd_layer(h).clamp(max=10)
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Decode attributes
        h_dec = self.decoder_attr(z, edge_index)
        x_rec = self.out_layer(h_dec)

        # Auxiliary predictions
        degree_pred = self.degree_predictor(z).squeeze()

        result = {
            'x_rec': x_rec,
            'z': z,
            'mu': mu,
            'logstd': logstd,
            'degree_pred': degree_pred
        }

        if return_attention and not self.use_multi_scale:
            result['attention_weights'] = attention_weights

        return result

    def recon_loss(self, z, edge_index):
        EPS = 1e-15

        if self.edge_decoder is not None:
            # Use learned decoder
            pos_value = self.edge_decoder(z, edge_index)
        else:
            # Use inner product
            pos_value = torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1))

        pos_loss = -torch.log(pos_value + EPS)

        # Hard negative sampling
        neg_edge_index = self.hard_neg_sampler.sample(edge_index, z, edge_index.size(1))

        if self.edge_decoder is not None:
            neg_value = self.edge_decoder(z, neg_edge_index)
        else:
            neg_value = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))

        neg_loss = -torch.log(1 - neg_value + EPS)

        return pos_loss, neg_loss, neg_edge_index

    def kl_loss(self, mu, logstd):
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def degree_loss(self, degree_pred, edge_index):
        """Auxiliary loss for degree prediction"""
        # Compute actual degrees
        actual_degrees = degree(edge_index[0], num_nodes=degree_pred.size(0)).float()
        return F.mse_loss(degree_pred, actual_degrees)

    def anomaly_score(self, x, results, edge_index, batch=None):
        x_rec = results['x_rec']
        z = results['z']

        # Attribute error per node
        attr_err = torch.sqrt(torch.mean((x - x_rec) ** 2, dim=1)).clamp(max=10)

        # Structure error per node
        pos_loss, neg_loss, neg_edge_index = self.recon_loss(z, edge_index)
        struct_err = torch.zeros(x.size(0), device=x.device)
        deg = torch.zeros(x.size(0), device=x.device)

        # Accumulate losses per node
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

        # Combine errors with adaptive or fixed weighting
        if self.adaptive_alpha is not None:
            score, current_alpha = self.adaptive_alpha(attr_err, struct_err)
        else:
            if self.score_mode == 'hybrid':
                score = self.alpha * attr_err + (1 - self.alpha) * struct_err
            elif self.score_mode == 'attr':
                score = attr_err
            elif self.score_mode == 'struct':
                score = struct_err
            else:
                score = attr_err

        # Pool for graph-level scores if needed
        if batch is not None and batch.max() > 0:
            score = global_mean_pool(score, batch)

        return score

    def get_interpretability_scores(self, x, edge_index, target_nodes=None):
        """Get interpretability scores for anomaly explanation"""
        if hasattr(self.encoder, 'compute_grad_cam'):
            grad_cam_scores = self.encoder.compute_grad_cam(x, edge_index, target_nodes)
        else:
            grad_cam_scores = None

        # Get attention weights if available
        results = self.forward(x, edge_index, return_attention=True)
        attention_weights = results.get('attention_weights', None)

        return {
            'grad_cam': grad_cam_scores,
            'attention_weights': attention_weights
        }


class GNNVariantAnomalyDetector:
    def __init__(self, in_dim, hidden_dim=128, latent_dim=64, num_layers=4, dropout=0.0,
                 alpha=0.5, gnn_type='gatv2', device='cpu', lr=1e-3, weight_decay=5e-4,
                 epochs=300, pooling='mean', score_mode='hybrid', decoder_type='bilinear',
                 use_contrastive=True, use_adaptive_alpha=True, use_multi_scale=True,
                 contrastive_weight=0.1, degree_weight=0.05):

        self.model = GNNV2Anomaly(
            in_dim, hidden_dim, latent_dim, num_layers, dropout, F.gelu, alpha,
            gnn_type, pooling, score_mode, decoder_type, use_contrastive,
            use_adaptive_alpha, use_multi_scale
        ).to(device)

        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.contrastive_weight = contrastive_weight
        self.degree_weight = degree_weight

    def fit(self, train_loader):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            total_loss = 0
            total_attr_loss = 0
            total_struct_loss = 0
            total_kl_loss = 0
            total_contrastive_loss = 0
            total_degree_loss = 0

            for data in train_loader:
                data = data.to(self.device)
                results = self.model(data.x, data.edge_index, getattr(data, 'batch', None))

                # Main losses
                attr_err = torch.mean(torch.sqrt(torch.mean((data.x - results['x_rec']) ** 2, dim=1)).clamp(max=10))
                pos_loss, neg_loss, neg_edge_index = self.model.recon_loss(results['z'], data.edge_index)

                # Structure error calculation
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

                # KL divergence
                kl = self.model.kl_loss(results['mu'], results['logstd'])

                # Main loss with adaptive weighting
                if self.model.adaptive_alpha is not None:
                    main_loss, current_alpha = self.model.adaptive_alpha(attr_err, struct_err)
                else:
                    main_loss = self.model.alpha * attr_err + (1 - self.model.alpha) * struct_err

                total_loss_batch = main_loss + kl

                # Additional losses
                if self.model.contrastive_loss is not None:
                    contrastive_loss = self.model.contrastive_loss(results['z'], data.edge_index, neg_edge_index)
                    total_loss_batch += self.contrastive_weight * contrastive_loss
                    total_contrastive_loss += contrastive_loss.item()

                # Degree prediction loss
                degree_loss = self.model.degree_loss(results['degree_pred'], data.edge_index)
                total_loss_batch += self.degree_weight * degree_loss

                # Backpropagation
                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()

                # Logging
                total_loss += total_loss_batch.item()
                total_attr_loss += attr_err.item()
                total_struct_loss += struct_err.item()
                total_kl_loss += kl.item()
                total_degree_loss += degree_loss.item()

            # Print comprehensive loss information
            n_batches = len(train_loader)
            if epoch % 50 == 0:
                print(f'Epoch {epoch + 1:03d} | '
                      f'Total: {total_loss / n_batches:.4f} | '
                      f'Attr: {total_attr_loss / n_batches:.4f} | '
                      f'Struct: {total_struct_loss / n_batches:.4f} | '
                      f'KL: {total_kl_loss / n_batches:.4f} | '
                      f'Contr: {total_contrastive_loss / n_batches:.4f} | '
                      f'Degree: {total_degree_loss / n_batches:.4f}')

                if self.model.adaptive_alpha is not None:
                    current_alpha = torch.sigmoid(self.model.adaptive_alpha.alpha).item()
                    print(f'        | Adaptive α: {current_alpha:.3f}')

    def decision_function(self, data, return_interpretability=False):
        self.model.eval()
        with torch.no_grad():
            results = self.model(data.x, data.edge_index, getattr(data, 'batch', None))
            score = self.model.anomaly_score(data.x, results, data.edge_index, getattr(data, 'batch', None))

            if return_interpretability:
                interpretability = self.model.get_interpretability_scores(data.x, data.edge_index)
                return score.cpu().numpy(), interpretability

            return score.cpu().numpy()

    def explain_anomaly(self, data, node_idx=None):
        """Provide explanations for detected anomalies"""
        self.model.eval()

        # Get anomaly scores and interpretability
        scores, interpretability = self.decision_function(data, return_interpretability=True)

        explanations = {
            'anomaly_scores': scores,
            'grad_cam_importance': interpretability['grad_cam'],
            'attention_weights': interpretability['attention_weights']
        }

        if node_idx is not None:
            explanations['node_specific'] = {
                'score': scores[node_idx] if len(scores.shape) == 1 else scores,
                'grad_cam': interpretability['grad_cam'][node_idx] if interpretability['grad_cam'] is not None else None
            }

        return explanations