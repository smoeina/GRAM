import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, global_add_pool
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.data import Data, Batch
from torch.nn import ModuleList, Linear, LSTM
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class GCN(nn.Module):
    """Graph Convolutional Network module"""
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.0, act=F.gelu):
        super(GCN, self).__init__()
        self.convs = ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            self.convs = ModuleList([GCNConv(in_channels, out_channels)])
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TemporalGRAMEncoder(nn.Module):
    """Temporal encoder using GCN + LSTM architecture"""
    def __init__(self, in_dim, hid_dim, latent_size, num_layers, dropout, act, lstm_hidden_dim=None):
        super(TemporalGRAMEncoder, self).__init__()
        
        self.hid_dim = hid_dim
        self.latent_size = latent_size
        lstm_hidden_dim = lstm_hidden_dim or hid_dim
        
        # GCN encoder for each timestep
        encoder_layers = max(2, num_layers // 2)
        self.gcn_encoder = GCN(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=encoder_layers,
            out_channels=hid_dim,
            dropout=dropout,
            act=act
        )
        
        # LSTM for temporal modeling
        self.lstm = LSTM(
            input_size=hid_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=False
        )
        
        # MLPs for mean and log variance
        self.encode_mu = ModuleList([
            Linear(lstm_hidden_dim, hid_dim),
            nn.GELU(),
            Linear(hid_dim, hid_dim),
            nn.GELU(),
            Linear(hid_dim, latent_size)
        ])
        
        self.encode_logvar = ModuleList([
            Linear(lstm_hidden_dim, hid_dim),
            nn.GELU(),
            Linear(hid_dim, hid_dim),
            nn.GELU(),
            Linear(hid_dim, latent_size)
        ])
        
        self.h = None  # Store embeddings for GradCAM
        self.h_grads = None  # Store gradients for GradCAM
    
    def forward(self, temporal_graphs):
        """
        Args:
            temporal_graphs: List of Data objects for each timestep
        Returns:
            mu, logvar, z: mean, log variance, and sampled latent variables
        """
        batch_size = len(temporal_graphs)
        device = temporal_graphs[0].x.device
        
        # Extract embeddings for each timestep
        temporal_embeddings = []
        
        for graph in temporal_graphs:
            # Apply GCN to each timestep
            h_t = self.gcn_encoder(graph.x, graph.edge_index)
            
            # Global pooling for graph-level representation
            batch = getattr(graph, 'batch', None)
            if batch is None:
                # Single graph case
                h_t_pooled = global_add_pool(h_t, torch.zeros(h_t.size(0), dtype=torch.long, device=device))
            else:
                # Batched graphs case
                h_t_pooled = global_add_pool(h_t, batch)
            
            temporal_embeddings.append(h_t_pooled)
        
        # Stack temporal embeddings: [batch_size, seq_len, hid_dim]
        temporal_embeddings = torch.stack(temporal_embeddings, dim=1)
        
        # Apply LSTM
        lstm_out, (h_n, c_n) = self.lstm(temporal_embeddings)
        
        # Use the final hidden state for latent variable generation
        final_hidden = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]
        
        # Store for GradCAM
        self.h = final_hidden
        if self.h.requires_grad:
            self.h.register_hook(lambda grad: setattr(self, 'h_grads', grad))
        
        # Generate mu and logvar
        mu = final_hidden
        for layer in self.encode_mu:
            mu = layer(mu)
        
        logvar = final_hidden
        for layer in self.encode_logvar:
            logvar = layer(logvar)
        
        # Clamp logvar for numerical stability
        logvar = logvar.clamp(max=10)
        
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        return mu, logvar, z


class TemporalGRAMDecoder(nn.Module):
    """Decoder for reconstructing the final timestep"""
    def __init__(self, latent_size, hid_dim, out_dim, num_layers, dropout, act):
        super(TemporalGRAMDecoder, self).__init__()
        
        decoder_layers = max(2, num_layers - (num_layers // 2))
        
        # MLPs for decoding
        self.decode_attr = ModuleList([
            Linear(latent_size, hid_dim),
            nn.GELU(),
            Linear(hid_dim, hid_dim),
            nn.GELU(),
            Linear(hid_dim, hid_dim)
        ])
        
        self.decode_struct = ModuleList([
            Linear(latent_size, hid_dim),
            nn.GELU(),
            Linear(hid_dim, hid_dim),
            nn.GELU(),
            Linear(hid_dim, hid_dim)
        ])
        
        # GCN decoders
        self.attr_decoder = GCN(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            num_layers=decoder_layers,
            out_channels=out_dim,
            dropout=dropout,
            act=act
        )
        
        self.struct_decoder = GCN(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            num_layers=decoder_layers - 1,
            out_channels=latent_size,
            dropout=dropout,
            act=act
        )
    
    def forward(self, z, final_graph):
        """
        Args:
            z: Latent variables [batch_size, latent_size]
            final_graph: Graph data for the final timestep
        Returns:
            x_recon: Reconstructed node features
            z_struct: Structural embeddings for edge reconstruction
        """
        # Expand z to match number of nodes in final graph
        num_nodes = final_graph.x.size(0)
        if z.dim() == 2 and z.size(0) == 1:
            # Single graph case
            z_expanded = z.repeat(num_nodes, 1)
        else:
            # Handle batched case - this needs proper implementation based on batch sizes
            z_expanded = z.repeat(num_nodes, 1)
        
        # Attribute reconstruction
        z_attr = z_expanded
        for layer in self.decode_attr:
            z_attr = layer(z_attr)
        
        x_recon = self.attr_decoder(z_attr, final_graph.edge_index)
        
        # Structure reconstruction  
        z_struct = z_expanded
        for layer in self.decode_struct:
            z_struct = layer(z_struct)
        
        z_struct = self.struct_decoder(z_struct, final_graph.edge_index)
        
        return x_recon, z_struct


class TemporalGRAM(nn.Module):
    """Main Temporal GRAM model"""
    def __init__(self, 
                 in_dim=1,
                 hid_dim=128, 
                 latent_size=64,
                 num_layers=6,
                 dropout=0.1,
                 act=F.gelu,
                 lstm_hidden_dim=None):
        super(TemporalGRAM, self).__init__()
        
        self.encoder = TemporalGRAMEncoder(
            in_dim=in_dim,
            hid_dim=hid_dim,
            latent_size=latent_size,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            lstm_hidden_dim=lstm_hidden_dim
        )
        
        self.decoder = TemporalGRAMDecoder(
            latent_size=latent_size,
            hid_dim=hid_dim,
            out_dim=in_dim,
            num_layers=num_layers,
            dropout=dropout,
            act=act
        )
        
        self.latent_size = latent_size
        
    def forward(self, temporal_graphs):
        """
        Args:
            temporal_graphs: List of Data objects for each timestep
        Returns:
            x_recon: Reconstructed features for final timestep
            z_struct: Structural embeddings for edge reconstruction
            mu, logvar, z: Latent variables and parameters
        """
        # Encode temporal sequence
        mu, logvar, z = self.encoder(temporal_graphs)
        
        # Decode using final timestep
        final_graph = temporal_graphs[-1]
        x_recon, z_struct = self.decoder(z, final_graph)
        
        return x_recon, z_struct, mu, logvar, z
    
    def kl_loss(self, mu, logvar):
        """KL divergence loss"""
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    def recon_loss(self, z_struct, pos_edge_index, neg_edge_index=None):
        """Reconstruction loss for edges"""
        EPS = 1e-15
        
        # Positive edge loss
        pos_pred = torch.sigmoid((z_struct[pos_edge_index[0]] * z_struct[pos_edge_index[1]]).sum(dim=1))
        pos_loss = -torch.log(pos_pred + EPS).mean()
        
        # Negative edge loss
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z_struct.size(0))
        
        neg_pred = torch.sigmoid((z_struct[neg_edge_index[0]] * z_struct[neg_edge_index[1]]).sum(dim=1))
        neg_loss = -torch.log(1 - neg_pred + EPS).mean()
        
        return pos_loss + neg_loss


class TemporalDataProcessor:
    """Processes temporal graph data for GRAM"""
    
    def __init__(self, num_timesteps=12, feature_type='degree'):
        self.num_timesteps = num_timesteps
        self.feature_type = feature_type
        
    def load_collegemsg_data(self, filepath):
        """Load CollegeMsg dataset"""
        # Read the data
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        node1, node2, timestamp = int(parts[0]), int(parts[1]), int(parts[2])
                        data.append((node1, node2, timestamp))
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(data, columns=['node1', 'node2', 'timestamp'])
        
        return self.process_temporal_data(df)
    
    def process_temporal_data(self, df):
        """Process temporal data into time windows"""
        # Get all unique nodes
        all_nodes = sorted(list(set(df['node1'].tolist() + df['node2'].tolist())))
        num_nodes = len(all_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Divide timestamps into equal windows
        min_time, max_time = df['timestamp'].min(), df['timestamp'].max()
        time_windows = np.linspace(min_time, max_time, self.num_timesteps + 1)
        
        temporal_graphs = []
        
        for t in range(self.num_timesteps):
            start_time, end_time = time_windows[t], time_windows[t + 1]
            
            # Get edges for this time window
            mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
            window_edges = df[mask]
            
            # Create edge index
            if len(window_edges) > 0:
                edge_list = []
                for _, row in window_edges.iterrows():
                    src_idx = node_to_idx[row['node1']]
                    dst_idx = node_to_idx[row['node2']]
                    edge_list.append([src_idx, dst_idx])
                    edge_list.append([dst_idx, src_idx])  # Make undirected
                
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                edge_index = remove_self_loops(edge_index)[0]
                edge_index = add_self_loops(edge_index, num_nodes=num_nodes)[0]
            else:
                # No edges in this window - create self-loops only
                edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
            
            # Generate node features
            node_features = self.generate_node_features(edge_index, num_nodes)
            
            # Create graph
            graph = Data(x=node_features, edge_index=edge_index)
            temporal_graphs.append(graph)
        
        return temporal_graphs, all_nodes, node_to_idx
    
    def generate_node_features(self, edge_index, num_nodes):
        """Generate artificial node features"""
        if self.feature_type == 'degree':
            # Node degree as features
            degrees = torch.zeros(num_nodes, dtype=torch.float)
            for i in range(num_nodes):
                degrees[i] = (edge_index[0] == i).sum().float()
            return degrees.unsqueeze(1)
        
        elif self.feature_type == 'onehot':
            # One-hot encoding
            return torch.eye(num_nodes, dtype=torch.float)
        
        elif self.feature_type == 'random':
            # Random features
            return torch.randn(num_nodes, 10)
        
        else:
            # Default: simple constant features
            return torch.ones(num_nodes, 1, dtype=torch.float)


class TemporalGRAMTrainer:
    """Training and evaluation for Temporal GRAM"""
    
    def __init__(self, model, device='cpu', lr=5e-4, alpha=0.5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.alpha = alpha  # Balance between attribute and structure loss
        
    def compute_loss(self, temporal_graphs):
        """Compute total loss for temporal graphs"""
        x_recon, z_struct, mu, logvar, z = self.model(temporal_graphs)
        
        final_graph = temporal_graphs[-1]
        
        # Attribute reconstruction loss (only for final timestep)
        attr_loss = F.mse_loss(x_recon, final_graph.x)
        
        # Structure reconstruction loss (for final timestep)
        struct_loss = self.model.recon_loss(z_struct, final_graph.edge_index)
        
        # KL divergence loss
        kl_loss = self.model.kl_loss(mu, logvar)
        
        # Total loss
        total_loss = self.alpha * attr_loss + (1 - self.alpha) * struct_loss + kl_loss
        
        return total_loss, attr_loss, struct_loss, kl_loss
    
    def train_epoch(self, temporal_graphs_list):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for temporal_graphs in temporal_graphs_list:
            self.optimizer.zero_grad()
            
            loss, attr_loss, struct_loss, kl_loss = self.compute_loss(temporal_graphs)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(temporal_graphs_list)
    
    def compute_anomaly_scores(self, temporal_graphs):
        """Compute anomaly scores using GradCAM-like approach"""
        self.model.eval()
        
        with torch.enable_grad():
            # Forward pass
            temporal_graphs_copy = [g.clone() for g in temporal_graphs]
            for g in temporal_graphs_copy:
                g.x.requires_grad_(True)
            
            x_recon, z_struct, mu, logvar, z = self.model(temporal_graphs_copy)
            
            # Compute scores based on reconstruction error and gradients
            final_graph = temporal_graphs_copy[-1]
            
            # Reconstruction-based score
            recon_error = torch.sum((x_recon - final_graph.x) ** 2, dim=1)
            
            # Gradient-based score (GradCAM-inspired)
            if hasattr(self.model.encoder, 'h') and self.model.encoder.h_grads is not None:
                h = self.model.encoder.h
                h_grads = self.model.encoder.h_grads
                
                # Compute attention coefficients
                alpha = torch.mean(h_grads, dim=0, keepdim=True)
                
                # Node-level scores
                grad_scores = torch.sum(F.gelu(h * alpha), dim=1)
                
                # Expand to node level (simple approach)
                grad_scores_expanded = grad_scores.repeat(final_graph.x.size(0))
            else:
                grad_scores_expanded = torch.zeros_like(recon_error)
            
            # Combine scores
            anomaly_scores = recon_error + grad_scores_expanded
            
        return anomaly_scores.detach().cpu().numpy()
    
    def evaluate(self, temporal_graphs_list, labels=None):
        """Evaluate model and return metrics"""
        self.model.eval()
        
        all_scores = []
        total_loss = 0
        
        with torch.no_grad():
            for temporal_graphs in temporal_graphs_list:
                # Compute loss
                loss, _, _, _ = self.compute_loss(temporal_graphs)
                total_loss += loss.item()
                
                # Compute anomaly scores
                scores = self.compute_anomaly_scores(temporal_graphs)
                all_scores.extend(scores)
        
        avg_loss = total_loss / len(temporal_graphs_list)
        
        results = {'loss': avg_loss, 'scores': all_scores}
        
        if labels is not None:
            # Compute AUC if labels provided
            try:
                auc = roc_auc_score(labels, all_scores)
                results['auc'] = auc
            except ValueError:
                results['auc'] = 0.5
        
        return results


def create_synthetic_anomalies(temporal_graphs, anomaly_ratio=0.1):
    """Create synthetic anomalies for testing"""
    labels = []
    modified_graphs = []
    
    for i, temporal_graphs_seq in enumerate(temporal_graphs):
        if i < len(temporal_graphs) * (1 - anomaly_ratio):
            # Normal sample
            labels.extend([0] * temporal_graphs_seq[-1].x.size(0))
            modified_graphs.append(temporal_graphs_seq)
        else:
            # Create anomaly by modifying final graph
            anomalous_seq = [g.clone() for g in temporal_graphs_seq]
            final_graph = anomalous_seq[-1]
            
            # Add noise to node features
            noise = torch.randn_like(final_graph.x) * 0.5
            final_graph.x = final_graph.x + noise
            
            # Mark nodes as anomalous
            labels.extend([1] * final_graph.x.size(0))
            modified_graphs.append(anomalous_seq)
    
    return modified_graphs, labels


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize data processor
    processor = TemporalDataProcessor(num_timesteps=12, feature_type='degree')
    
    # Load and process data (example with synthetic data if CollegeMsg not available)
    print("Processing temporal data...")
    
    # For demonstration, create dummy temporal graphs
    # In practice, use: temporal_graphs, nodes, node_map = processor.load_collegemsg_data('CollegeMsg.txt')
    
    num_nodes = 50
    dummy_temporal_graphs = []
    for t in range(12):
        # Create random graph for each timestep
        num_edges = np.random.randint(50, 200)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn(num_nodes, 1)
        dummy_temporal_graphs.append(Data(x=x, edge_index=edge_index))
    
    # Initialize model
    model = TemporalGRAM(
        in_dim=1,
        hid_dim=64,
        latent_size=32,
        num_layers=4,
        dropout=0.1
    )
    
    # Initialize trainer
    trainer = TemporalGRAMTrainer(model, device=device, lr=1e-3, alpha=0.5)
    
    # Prepare training data (single sequence for demo)
    training_data = [dummy_temporal_graphs]
    
    print("Training model...")
    for epoch in range(5):
        loss = trainer.train_epoch(training_data)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    # Evaluate
    print("Computing anomaly scores...")
    results = trainer.evaluate(training_data)
    print(f"Evaluation loss: {results['loss']:.4f}")
    print(f"Anomaly scores shape: {len(results['scores'])}")
    
    print("Temporal GRAM model ready for use!")
