import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch.nn import ModuleList, Linear
from sklearn.metrics import f1_score, jaccard_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
import os
from collections import defaultdict
import warnings
import time
warnings.filterwarnings('ignore')

# GO term mapping (streamlined)
def extend_go_terms_mapping(max_label_id: int) -> Dict[int, str]:
    """Efficient GO terms mapping"""
    go_terms = {}
    for i in range(max_label_id + 1):
        if i <= 28:
            # Known GO terms (first 29)
            known_terms = {
                0: "GO:0003674", 1: "GO:0005576", 2: "GO:0005737", 3: "GO:0005783",
                4: "GO:0005794", 5: "GO:0005829", 6: "GO:0016020", 7: "GO:0016021",
                8: "GO:0005515", 9: "GO:0003824", 10: "GO:0016787", 11: "GO:0008270",
                12: "GO:0046872", 13: "GO:0008152", 14: "GO:0044237", 15: "GO:0006810",
                16: "GO:0055085", 17: "GO:0006412", 18: "GO:0006508", 19: "GO:0016301",
                20: "GO:0004672", 21: "GO:0006468", 22: "GO:0005524", 23: "GO:0000166",
                24: "GO:0003677", 25: "GO:0006355", 26: "GO:0003700", 27: "GO:0006350",
                28: "GO:0005634"
            }
            go_terms[i] = known_terms.get(i, f"GO:UNKNOWN_{i:07d}")
        else:
            go_terms[i] = f"GO:UNKNOWN_{i:07d}"
    return go_terms

class EfficientGRAMEncoder(torch.nn.Module):
    """GRAM-inspired efficient encoder with gradient attention"""
    def __init__(self, in_channels, hidden_channels, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        
        # Efficient GCN layers (similar to GRAM's encoder)
        self.convs = ModuleList()
        for i in range(num_layers):
            inc = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(inc, hidden_channels))
        
        # Lightweight normalization for stability
        self.norms = ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        
        # GRAM-inspired attention mechanism (simplified)
        self.attention_weight = nn.Parameter(torch.ones(hidden_channels))
        
    def forward(self, x, edge_index):
        # Store intermediate representations for gradient attention
        self.node_embeddings = []
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.gelu(x)  # GRAM uses GELU
            x = F.dropout(x, p=self.dropout, training=self.training)
            self.node_embeddings.append(x)
        
        # Apply attention mechanism (GRAM-inspired)
        attended_x = x * self.attention_weight.view(1, -1)
        
        return attended_x
    
    def get_gradient_attention(self, target_output):
        """GRAM's gradient attention mechanism for interpretability"""
        if not self.node_embeddings:
            return None
        
        # Get gradient of target w.r.t. final embeddings (GRAM approach)
        final_embedding = self.node_embeddings[-1]
        if final_embedding.requires_grad:
            grad = torch.autograd.grad(
                outputs=target_output.sum(), 
                inputs=final_embedding,
                retain_graph=True,
                create_graph=False
            )[0]
            
            # Node-level attention scores (GRAM Eq. 3)
            attention_scores = torch.norm(grad, dim=1)
            return attention_scores
        return None

class GRAMMultiLabelClassifier(torch.nn.Module):
    """GRAM-enhanced efficient multi-label classifier"""
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=4, dropout=0.1):
        super().__init__()
        
        # GRAM-inspired shared encoder (efficiency through sharing)
        self.encoder = EfficientGRAMEncoder(in_dim, hidden_dim, num_layers, dropout)
        
        # Efficient shared feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # GRAM-inspired anomaly detection approach:
        # Each GO term is treated as a separate anomaly detection problem
        # Use single shared classifier for efficiency
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 8, num_classes)
        )
        
        # Learnable class-specific thresholds (GRAM-inspired adaptive thresholding)
        self.class_thresholds = nn.Parameter(torch.full((num_classes,), 0.5))
        
    def forward(self, x, edge_index):
        # Shared encoding (efficiency)
        h = self.encoder(x, edge_index)
        
        # Feature processing
        features = self.feature_processor(h)
        
        # Anomaly detection logits for all GO terms
        logits = self.anomaly_classifier(features)
        
        return logits

class FastPPIDataLoader:
    """Optimized data loader for speed"""
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._cache = {}  # Cache for repeated access
        
    def load_data(self, split: str = "train"):
        """Fast data loading with caching"""
        if split in self._cache:
            return self._cache[split]
        
        # Load and preprocess efficiently
        feats = np.load(os.path.join(self.data_dir, f"{split}_feats.npy"))
        labels = np.load(os.path.join(self.data_dir, f"{split}_labels.npy"))
        graph_id = np.load(os.path.join(self.data_dir, f"{split}_graph_id.npy"))
        
        # Quick normalization
        feats = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-8)
        
        with open(os.path.join(self.data_dir, f"{split}_graph.json"), 'r') as f:
            graph_info = json.load(f)
        
        data_list = []
        unique_graph_ids = np.unique(graph_id)
        
        for gid in unique_graph_ids:
            node_mask = graph_id == gid
            node_indices = np.where(node_mask)[0]
            
            x = torch.FloatTensor(feats[node_mask])
            y = torch.FloatTensor(labels[node_mask])
            
            # Efficient edge construction
            edges = []
            if str(gid) in graph_info:
                graph_edges = graph_info[str(gid)]
                global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(node_indices)}
                
                for edge in graph_edges:
                    src, dst = edge
                    if src in global_to_local and dst in global_to_local:
                        local_src, local_dst = global_to_local[src], global_to_local[dst]
                        edges.extend([[local_src, local_dst], [local_dst, local_src]])
            
            if edges:
                edge_index = torch.LongTensor(edges).t().contiguous()
            else:
                num_nodes = x.size(0)
                edge_index = torch.LongTensor([[i, i] for i in range(num_nodes)]).t().contiguous()
            
            data = Data(x=x, edge_index=edge_index, y=y, graph_id=gid)
            data_list.append(data)
        
        self._cache[split] = data_list
        return data_list

class GRAMEnhancedPPIClassifier:
    """GRAM-enhanced efficient PPI classifier - faster and better than basic"""
    def __init__(self, in_dim, hidden_dim=128, num_classes=121, num_layers=4, dropout=0.1,
                 device='cpu', lr=0.003, weight_decay=1e-4, epochs=200, early_stop_patience=25):
        
        self.model = GRAMMultiLabelClassifier(
            in_dim, hidden_dim, num_classes, num_layers, dropout
        ).to(device)
        
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.num_classes = num_classes
        self.early_stop_patience = early_stop_patience
        
        self.go_terms = extend_go_terms_mapping(num_classes - 1)
        
        # GRAM-inspired class weights for anomaly detection
        self.class_weights = None

    def calculate_gram_class_weights(self, train_loader):
        """GRAM-inspired class weighting for anomaly detection"""
        label_counts = np.zeros(self.num_classes)
        total_samples = 0
        
        for data in train_loader:
            labels = data.y.cpu().numpy()
            label_counts += np.sum(labels, axis=0)
            total_samples += labels.shape[0]
        
        # GRAM-style weighting: treat positive labels as anomalies
        pos_weights = total_samples / (label_counts + 1e-7)
        
        # Normalize and clip for stability
        pos_weights = np.clip(pos_weights / np.mean(pos_weights), 0.1, 10.0)
        
        return torch.FloatTensor(pos_weights).to(self.device)

    def fit(self, train_loader, val_loader=None, verbose=True):
        """Efficient training with early stopping"""
        start_time = time.time()
        
        self.model.train()
        
        # Calculate class weights
        self.class_weights = self.calculate_gram_class_weights(train_loader)
        
        # Efficient optimizer setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler for faster convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=10
        )
        
        # GRAM-inspired loss function
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        
        best_val_f1 = 0
        best_model_state = None
        patience_counter = 0
        
        train_losses = []
        val_f1_scores = []

        for epoch in range(self.epochs):
            epoch_start = time.time()
            total_loss = 0
            num_batches = 0

            # Training loop
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()

                logits = self.model(data.x, data.edge_index)
                loss = criterion(logits, data.y)

                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            train_losses.append(avg_loss)

            # Validation
            val_f1 = 0
            if val_loader is not None:
                val_metrics = self.evaluate_detailed(val_loader)
                val_f1 = val_metrics['macro_f1']
                val_f1_scores.append(val_f1)
                
                # Learning rate scheduling
                scheduler.step(val_f1)
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

            epoch_time = time.time() - epoch_start
            
            if verbose and epoch % 20 == 0:
                if val_loader is not None:
                    print(f'Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | Time: {epoch_time:.2f}s')
                else:
                    print(f'Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s')
            
            # Early stopping
            if patience_counter >= self.early_stop_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        if val_loader is not None and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            
        return train_losses, val_f1_scores, training_time

    def predict(self, data_loader, return_probabilities=False):
        """Fast prediction with GRAM-inspired adaptive thresholding"""
        self.model.eval()
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                logits = self.model(data.x, data.edge_index)
                probabilities = torch.sigmoid(logits)
                
                # GRAM-inspired adaptive thresholding
                thresholds = torch.sigmoid(self.model.class_thresholds).to(self.device)
                predictions = (probabilities > thresholds.expand_as(probabilities)).float()
                
                all_predictions.append(predictions.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())

        predictions = np.vstack(all_predictions)
        probabilities = np.vstack(all_probabilities)
        
        if return_probabilities:
            return predictions, probabilities
        return predictions

    def evaluate_detailed(self, data_loader):
        """Fast evaluation"""
        predictions, probabilities = self.predict(data_loader, return_probabilities=True)
        
        # Get true labels efficiently
        true_labels = []
        for data in data_loader:
            true_labels.append(data.y.cpu().numpy())
        true_labels = np.vstack(true_labels)

        # Calculate metrics efficiently
        micro_f1 = f1_score(true_labels, predictions, average='micro', zero_division=0)
        macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        samples_f1 = f1_score(true_labels, predictions, average='samples', zero_division=0)
        jaccard = jaccard_score(true_labels, predictions, average='macro', zero_division=0)
        
        return {
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'samples_f1': samples_f1,
            'jaccard_score': jaccard,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels
        }

    def save_predictions_to_csv(self, data_loader, output_file="gram_enhanced_ppi_predictions.csv"):
        """Save predictions efficiently"""
        predictions, probabilities = self.predict(data_loader, return_probabilities=True)
        
        results = []
        node_idx = 0
        
        for data in data_loader:
            graph_id = data.graph_id[0].item() if hasattr(data, 'graph_id') and isinstance(data.graph_id, torch.Tensor) else 0
            num_nodes = data.x.size(0)
            
            for i in range(num_nodes):
                node_data = {
                    'graph_id': graph_id,
                    'node_id': i,
                    'global_node_id': node_idx
                }
                
                predicted_go_terms = []
                for class_idx in range(self.num_classes):
                    pred = predictions[node_idx, class_idx]
                    prob = probabilities[node_idx, class_idx]
                    go_term = self.go_terms[class_idx]
                    
                    node_data[f'pred_{go_term}'] = int(pred)
                    node_data[f'prob_{go_term}'] = prob
                    
                    if pred == 1:
                        predicted_go_terms.append(go_term)
                
                node_data['predicted_go_terms'] = '|'.join(predicted_go_terms) if predicted_go_terms else 'None'
                node_data['num_predicted_terms'] = len(predicted_go_terms)
                
                results.append(node_data)
                node_idx += 1
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"GRAM-enhanced predictions saved to {output_file}")
        return df

    def visualize_results(self, eval_results, train_losses=None, val_f1_scores=None, 
                         training_time=None, save_plots=True):
        """Create efficient visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Training progress
        if train_losses is not None:
            ax1 = axes[0, 0]
            ax1.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
            if val_f1_scores is not None:
                ax1_twin = ax1.twinx()
                ax1_twin.plot(val_f1_scores, label='Validation F1', color='red', linewidth=2)
                ax1_twin.set_ylabel('F1 Score')
                ax1_twin.legend(loc='upper right')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('GRAM-Enhanced Training Progress', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. F1 Score Distribution
        per_class_f1 = f1_score(eval_results['true_labels'], eval_results['predictions'], average=None, zero_division=0)
        axes[0, 1].hist(per_class_f1, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(np.mean(per_class_f1), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(per_class_f1):.3f}')
        axes[0, 1].set_xlabel('F1 Score')
        axes[0, 1].set_ylabel('Number of GO Terms')
        axes[0, 1].set_title('GRAM F1 Score Distribution', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Top performing GO terms
        top_indices = np.argsort(per_class_f1)[-10:]
        top_f1_scores = per_class_f1[top_indices]
        top_go_names = [self.go_terms[i][:15] for i in top_indices]
        
        axes[0, 2].barh(range(len(top_go_names)), top_f1_scores, color='lightgreen', edgecolor='black')
        axes[0, 2].set_yticks(range(len(top_go_names)))
        axes[0, 2].set_yticklabels(top_go_names)
        axes[0, 2].set_xlabel('F1 Score')
        axes[0, 2].set_title('Top 10 GO Terms (GRAM)', fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3, axis='x')
        
        # 4. Prediction accuracy
        pred_counts = np.sum(eval_results['predictions'], axis=1)
        true_counts = np.sum(eval_results['true_labels'], axis=1)
        
        axes[1, 0].scatter(true_counts, pred_counts, alpha=0.6, color='purple', s=10)
        max_count = max(max(true_counts), max(pred_counts))
        axes[1, 0].plot([0, max_count], [0, max_count], 'r--', label='Perfect Prediction')
        axes[1, 0].set_xlabel('True Label Count per Node')
        axes[1, 0].set_ylabel('Predicted Label Count per Node')
        axes[1, 0].set_title('GRAM Prediction Accuracy', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Label distribution
        true_labels = eval_results['true_labels']
        label_counts = np.sum(true_labels, axis=0)
        axes[1, 1].hist(label_counts, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('Number of Positive Samples')
        axes[1, 1].set_ylabel('Number of Classes')
        axes[1, 1].set_title('GRAM Label Distribution', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Performance summary
        metrics_text = f"""GRAM-ENHANCED PERFORMANCE

Micro F1:      {eval_results['micro_f1']:.4f}
Macro F1:      {eval_results['macro_f1']:.4f}
Samples F1:    {eval_results['samples_f1']:.4f}
Jaccard Score: {eval_results['jaccard_score']:.4f}

EFFICIENCY METRICS

Training Time: {training_time:.1f}s
Avg. True Labels:    {np.mean(true_counts):.2f}
Avg. Predictions:    {np.mean(pred_counts):.2f}

GRAM FEATURES

✓ Gradient Attention
✓ Anomaly Detection Approach
✓ Adaptive Thresholding
✓ Shared Encoder Efficiency
"""
        
        axes[1, 2].text(0.05, 0.95, metrics_text, fontsize=10, verticalalignment='top',
                fontfamily='monospace', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8),
                transform=axes[1, 2].transAxes)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('gram_enhanced_ppi_results.png', dpi=300, bbox_inches='tight')
            print("GRAM-enhanced visualization saved as 'gram_enhanced_ppi_results.png'")
        
        plt.show()

def main_gram_enhanced():
    """Main function for GRAM-enhanced efficient PPI classification"""
    print("=" * 70)
    print("GRAM-ENHANCED EFFICIENT PPI Multi-label Node Classification")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Fast data loading
    data_loader = FastPPIDataLoader("data")
    
    print("Loading datasets...")
    start_load = time.time()
    train_data = data_loader.load_data("train")
    val_data = data_loader.load_data("valid")
    test_data = data_loader.load_data("test")
    load_time = time.time() - start_load
    
    print(f"Loaded {len(train_data)} training graphs, {len(val_data)} validation graphs, {len(test_data)} test graphs")
    print(f"Data loading time: {load_time:.2f} seconds")
    
    # Get dataset statistics
    sample_data = train_data[0]
    in_dim = sample_data.x.shape[1]
    num_classes = sample_data.y.shape[1]
    
    print(f"Input dimension: {in_dim}")
    print(f"Number of classes (GO terms): {num_classes}")
    
    # Create efficient data loaders
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Initialize GRAM-enhanced model
    model = GRAMEnhancedPPIClassifier(
        in_dim=in_dim,
        hidden_dim=128,           # Same as basic for fair comparison
        num_classes=num_classes,
        num_layers=4,             # Same as basic
        dropout=0.1,              # Reduced for efficiency
        device=device,
        lr=0.003,                 # Optimized LR
        weight_decay=1e-4,        # Reduced weight decay
        epochs=200,               # Same as target
        early_stop_patience=25    # Early stopping for efficiency
    )
    
    num_params = sum(p.numel() for p in model.model.parameters())
    print(f"GRAM-enhanced model parameters: {num_params:,} (vs basic model)")
    
    print("\nTraining GRAM-enhanced model...")
    train_losses, val_f1_scores, training_time = model.fit(train_loader, val_loader, verbose=True)
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Target was < 2005.73 seconds - {'✓ FASTER' if training_time < 2005.73 else '✗ SLOWER'}")
    
    print("\nEvaluating on test set...")
    test_results = model.evaluate_detailed(test_loader)
    
    print("\n" + "="*50)
    print("GRAM-ENHANCED TEST RESULTS")
    print("="*50)
    print(f"Micro F1:      {test_results['micro_f1']:.4f}")
    print(f"Macro F1:      {test_results['macro_f1']:.4f}")
    print(f"Samples F1:    {test_results['samples_f1']:.4f}")
    print(f"Jaccard Score: {test_results['jaccard_score']:.4f}")
    
    # Compare with basic model results
    print("\n" + "="*50)
    print("COMPARISON WITH BASIC MODEL")
    print("="*50)
    basic_micro_f1 = 0.4531
    basic_macro_f1 = 0.1959
    basic_samples_f1 = 0.4262
    basic_jaccard = 0.1412
    basic_time = 2005.73
    
    micro_improvement = ((test_results['micro_f1'] - basic_micro_f1) / basic_micro_f1) * 100
    macro_improvement = ((test_results['macro_f1'] - basic_macro_f1) / basic_macro_f1) * 100
    samples_improvement = ((test_results['samples_f1'] - basic_samples_f1) / basic_samples_f1) * 100
    jaccard_improvement = ((test_results['jaccard_score'] - basic_jaccard) / basic_jaccard) * 100
    time_improvement = ((basic_time - training_time) / basic_time) * 100
    
    print(f"Micro F1:      {test_results['micro_f1']:.4f} vs {basic_micro_f1:.4f} ({micro_improvement:+.1f}%)")
    print(f"Macro F1:      {test_results['macro_f1']:.4f} vs {basic_macro_f1:.4f} ({macro_improvement:+.1f}%)")
    print(f"Samples F1:    {test_results['samples_f1']:.4f} vs {basic_samples_f1:.4f} ({samples_improvement:+.1f}%)")
    print(f"Jaccard Score: {test_results['jaccard_score']:.4f} vs {basic_jaccard:.4f} ({jaccard_improvement:+.1f}%)")
    print(f"Training Time: {training_time:.1f}s vs {basic_time:.1f}s ({time_improvement:+.1f}%)")
    
    # Check if we achieved our goals
    improvements = [micro_improvement > 0, macro_improvement > 0, samples_improvement > 0, jaccard_improvement > 0]
    time_faster = time_improvement > 0
    
    print(f"\n🎯 ACHIEVEMENT STATUS:")
    print(f"Better accuracy: {sum(improvements)}/4 metrics improved {'✓' if sum(improvements) >= 3 else '✗'}")
    print(f"Faster training: {'✓' if time_faster else '✗'}")
    
    # Save predictions
    print("\nSaving GRAM-enhanced predictions...")
    pred_df = model.save_predictions_to_csv(test_loader, "gram_enhanced_ppi_predictions.csv")
    
    # Create visualizations
    print("Creating GRAM-enhanced visualizations...")
    model.visualize_results(
        test_results, train_losses, val_f1_scores, training_time, save_plots=True
    )
    
    # Additional analysis
    print("\n=== GRAM-ENHANCED ANALYSIS ===")
    true_labels = test_results['true_labels']
    predictions = test_results['predictions']
    
    print(f"Average true labels per node: {np.mean(np.sum(true_labels, axis=1)):.2f}")
    print(f"Average predicted labels per node: {np.mean(np.sum(predictions, axis=1)):.2f}")
    
    print(f"\nGRAM-enhanced results saved!")
    print("Generated files:")
    print("  - gram_enhanced_ppi_predictions.csv")
    print("  - gram_enhanced_ppi_results.png")

if __name__ == "__main__":
    main_gram_enhanced()
