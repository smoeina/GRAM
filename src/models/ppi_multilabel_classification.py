import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv, TransformerConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
from torch.nn import ModuleList, Linear
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix, f1_score, jaccard_score
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Callable, Union, List, Dict
import os
from collections import defaultdict
import warnings
import time
import psutil
warnings.filterwarnings('ignore')

# GO term mapping (basic mapping - can be extended)
GO_TERMS = {
    0: "GO:0003674",  # molecular_function
    1: "GO:0005576",  # extracellular region
    2: "GO:0005737",  # cytoplasm
    3: "GO:0005783",  # endoplasmic reticulum
    4: "GO:0005794",  # Golgi apparatus
    5: "GO:0005829",  # cytosol
    6: "GO:0016020",  # membrane
    7: "GO:0016021",  # integral component of membrane
    8: "GO:0005515",  # protein binding
    9: "GO:0003824",  # catalytic activity
    10: "GO:0016787", # hydrolase activity
    11: "GO:0008270", # zinc ion binding
    12: "GO:0046872", # metal ion binding
    13: "GO:0008152", # metabolic process
    14: "GO:0044237", # cellular metabolic process
    15: "GO:0006810", # transport
    16: "GO:0055085", # transmembrane transport
    17: "GO:0006412", # translation
    18: "GO:0006508", # proteolysis
    19: "GO:0016301", # kinase activity
    20: "GO:0004672", # protein kinase activity
    21: "GO:0006468", # protein phosphorylation
    22: "GO:0005524", # ATP binding
    23: "GO:0000166", # nucleotide binding
    24: "GO:0003677", # DNA binding
    25: "GO:0006355", # regulation of transcription
    26: "GO:0003700", # sequence-specific DNA binding transcription factor activity
    27: "GO:0006350", # transcription
    28: "GO:0005634",  # nucleus
    # Add more GO terms as needed
}

def extend_go_terms_mapping(max_label_id: int) -> Dict[int, str]:
    """Extend GO terms mapping for unknown labels"""
    extended_mapping = GO_TERMS.copy()
    for i in range(max_label_id + 1):
        if i not in extended_mapping:
            extended_mapping[i] = f"GO:UNKNOWN_{i:07d}"
    return extended_mapping

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

class MultiLabelGNNClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=4, dropout=0.0, act=F.gelu,
                 gnn_type='gatv2'):
        """
        Multi-label GNN Classifier for node-level classification
        """
        super().__init__()
        self.encoder = FlexibleGNN(in_dim, hidden_dim, num_layers, gnn_type, dropout, act)
        
        # Multi-label classification head with sigmoid activation
        self.classifier = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 4, num_classes),
            nn.Sigmoid()  # Sigmoid for multi-label classification
        )

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        logits = self.classifier(h)
        return logits

class PPIDataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        
    def load_data(self, split: str = "train"):
        """Load PPI data from .npy and .json files"""
        # Load features, labels, and graph info
        feats = np.load(os.path.join(self.data_dir, f"{split}_feats.npy"))
        labels = np.load(os.path.join(self.data_dir, f"{split}_labels.npy"))
        graph_id = np.load(os.path.join(self.data_dir, f"{split}_graph_id.npy"))
        
        # Load graph structure
        with open(os.path.join(self.data_dir, f"{split}_graph.json"), 'r') as f:
            graph_info = json.load(f)
        
        # Convert to PyTorch geometric format
        data_list = []
        
        # Get unique graph IDs
        unique_graph_ids = np.unique(graph_id)
        
        for gid in unique_graph_ids:
            # Get nodes for this graph
            node_mask = graph_id == gid
            node_indices = np.where(node_mask)[0]
            
            # Extract features and labels for this graph
            x = torch.FloatTensor(feats[node_mask])
            y = torch.FloatTensor(labels[node_mask])
            
            # Build edge index from graph_info
            edges = []
            if str(gid) in graph_info:
                graph_edges = graph_info[str(gid)]
                # Map global node indices to local indices
                global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(node_indices)}
                
                for edge in graph_edges:
                    src, dst = edge
                    if src in global_to_local and dst in global_to_local:
                        edges.append([global_to_local[src], global_to_local[dst]])
                        edges.append([global_to_local[dst], global_to_local[src]])  # Undirected
            
            if edges:
                edge_index = torch.LongTensor(edges).t().contiguous()
            else:
                # Create self-loops if no edges
                num_nodes = x.size(0)
                edge_index = torch.LongTensor([[i, i] for i in range(num_nodes)]).t().contiguous()
            
            data = Data(x=x, edge_index=edge_index, y=y, graph_id=gid)
            data_list.append(data)
        
        return data_list

class PPIMultiLabelClassifier:
    def __init__(self, in_dim, hidden_dim=128, num_classes=121, num_layers=4, dropout=0.2,
                 gnn_type='gatv2', device='cpu', lr=1e-3, weight_decay=5e-4, epochs=300,
                 threshold=0.5):
        """
        PPI Multi-label Classification wrapper
        """
        self.model = MultiLabelGNNClassifier(
            in_dim, hidden_dim, num_classes, num_layers, dropout, F.gelu, gnn_type
        ).to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.num_classes = num_classes
        self.threshold = threshold
        self.go_terms = extend_go_terms_mapping(num_classes - 1)

    def fit(self, train_loader, val_loader=None, verbose=True):
        """Train the model"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCELoss()  # Binary Cross Entropy for multi-label
        
        best_val_f1 = 0
        best_model_state = None
        
        train_losses = []
        val_f1_scores = []

        for epoch in range(self.epochs):
            total_loss = 0
            total_samples = 0

            # Training loop
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()

                logits = self.model(data.x, data.edge_index)
                loss = criterion(logits, data.y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_samples += data.y.size(0)

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Validation
            val_f1 = 0
            if val_loader is not None:
                val_metrics = self.evaluate_detailed(val_loader)
                val_f1 = val_metrics['macro_f1']
                val_f1_scores.append(val_f1)
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = self.model.state_dict().copy()

            if verbose and epoch % 10 == 0:
                if val_loader is not None:
                    print(f'Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}')
                else:
                    print(f'Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f}')

        # Load best model if validation was used
        if val_loader is not None and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return train_losses, val_f1_scores

    def predict(self, data_loader, return_probabilities=False):
        """Make predictions"""
        self.model.eval()
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                probabilities = self.model(data.x, data.edge_index)
                predictions = (probabilities > self.threshold).float()
                
                all_predictions.append(predictions.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())

        predictions = np.vstack(all_predictions)
        probabilities = np.vstack(all_probabilities)
        
        if return_probabilities:
            return predictions, probabilities
        return predictions

    def evaluate_detailed(self, data_loader):
        """Get detailed multi-label classification metrics"""
        predictions, probabilities = self.predict(data_loader, return_probabilities=True)
        
        # Get true labels
        true_labels = []
        for data in data_loader:
            true_labels.append(data.y.cpu().numpy())
        true_labels = np.vstack(true_labels)

        # Calculate metrics
        micro_f1 = f1_score(true_labels, predictions, average='micro')
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        samples_f1 = f1_score(true_labels, predictions, average='samples')
        jaccard = jaccard_score(true_labels, predictions, average='macro')
        
        # Calculate per-class metrics
        per_class_f1 = f1_score(true_labels, predictions, average=None)
        
        return {
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'samples_f1': samples_f1,
            'jaccard_score': jaccard,
            'per_class_f1': per_class_f1,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels
        }

    def save_predictions_to_csv(self, data_loader, output_file="ppi_predictions.csv"):
        """Save predictions to CSV with GO term labels"""
        predictions, probabilities = self.predict(data_loader, return_probabilities=True)
        
        results = []
        node_idx = 0
        
        for data in data_loader:
            if hasattr(data, 'graph_id'):
                graph_id = data.graph_id
                if isinstance(graph_id, (list, tuple)):
                    graph_id = graph_id[0]
                elif isinstance(graph_id, torch.Tensor):
                    graph_id = graph_id.item()
            else:
                graph_id = 0
            num_nodes = data.x.size(0)
            
            for i in range(num_nodes):
                node_data = {
                    'graph_id': graph_id,
                    'node_id': i,
                    'global_node_id': node_idx
                }
                
                # Add predictions and probabilities for each GO term
                predicted_go_terms = []
                for class_idx in range(self.num_classes):
                    pred = predictions[node_idx, class_idx]
                    prob = probabilities[node_idx, class_idx]
                    go_term = self.go_terms.get(class_idx, f"GO:UNKNOWN_{class_idx}")
                    
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
        print(f"Predictions saved to {output_file}")
        return df

    def visualize_results(self, eval_results, save_plots=True):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. F1 scores distribution
        per_class_f1 = eval_results['per_class_f1']
        axes[0, 0].hist(per_class_f1, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribution of Per-Class F1 Scores')
        axes[0, 0].set_xlabel('F1 Score')
        axes[0, 0].set_ylabel('Number of Classes')
        axes[0, 0].axvline(np.mean(per_class_f1), color='red', linestyle='--', label=f'Mean: {np.mean(per_class_f1):.3f}')
        axes[0, 0].legend()
        
        # 2. Top performing GO terms
        top_indices = np.argsort(per_class_f1)[-10:]
        top_f1_scores = per_class_f1[top_indices]
        top_go_terms = [self.go_terms[i] for i in top_indices]
        
        axes[0, 1].barh(range(len(top_go_terms)), top_f1_scores, color='lightgreen')
        axes[0, 1].set_yticks(range(len(top_go_terms)))
        axes[0, 1].set_yticklabels([f'{term[:15]}...' if len(term) > 15 else term for term in top_go_terms])
        axes[0, 1].set_title('Top 10 GO Terms by F1 Score')
        axes[0, 1].set_xlabel('F1 Score')
        
        # 3. Label distribution
        true_labels = eval_results['true_labels']
        label_counts = np.sum(true_labels, axis=0)
        axes[0, 2].hist(label_counts, bins=20, alpha=0.7, color='orange')
        axes[0, 2].set_title('Label Distribution')
        axes[0, 2].set_xlabel('Number of Positive Samples')
        axes[0, 2].set_ylabel('Number of Classes')
        
        # 4. Prediction vs True label counts per sample
        pred_counts = np.sum(eval_results['predictions'], axis=1)
        true_counts = np.sum(true_labels, axis=1)
        
        axes[1, 0].scatter(true_counts, pred_counts, alpha=0.6, color='purple')
        axes[1, 0].plot([0, max(true_counts)], [0, max(true_counts)], 'r--', label='Perfect Prediction')
        axes[1, 0].set_title('Predicted vs True Label Counts per Node')
        axes[1, 0].set_xlabel('True Label Count')
        axes[1, 0].set_ylabel('Predicted Label Count')
        axes[1, 0].legend()
        
        # 5. Confusion matrix for most frequent labels
        most_frequent_labels = np.argsort(label_counts)[-5:]
        cm_data = []
        
        for label_idx in most_frequent_labels:
            true_binary = true_labels[:, label_idx]
            pred_binary = eval_results['predictions'][:, label_idx]
            
            tn = np.sum((true_binary == 0) & (pred_binary == 0))
            fp = np.sum((true_binary == 0) & (pred_binary == 1))
            fn = np.sum((true_binary == 1) & (pred_binary == 0))
            tp = np.sum((true_binary == 1) & (pred_binary == 1))
            
            cm_data.append([tn, fp, fn, tp])
        
        # Show confusion matrix for the most frequent label
        if cm_data:
            cm = np.array([[cm_data[0][0], cm_data[0][1]], [cm_data[0][2], cm_data[0][3]]])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_title(f'Confusion Matrix: {self.go_terms[most_frequent_labels[0]][:20]}')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('True')
        
        # 6. Performance metrics summary
        metrics_text = f"""
        Micro F1: {eval_results['micro_f1']:.3f}
        Macro F1: {eval_results['macro_f1']:.3f}
        Samples F1: {eval_results['samples_f1']:.3f}
        Jaccard Score: {eval_results['jaccard_score']:.3f}
        
        Avg. labels per node: {np.mean(true_counts):.2f}
        Avg. predictions per node: {np.mean(pred_counts):.2f}
        """
        
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Performance Summary')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('ppi_classification_results.png', dpi=300, bbox_inches='tight')
            print("Visualization saved as 'ppi_classification_results.png'")
        
        plt.show()

def main():
    """Main function to run PPI multi-label classification"""
    print("Starting PPI Multi-label Node Classification...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_loader = PPIDataLoader("data")
    
    print("Loading datasets...")
    train_data = data_loader.load_data("train")
    val_data = data_loader.load_data("valid")
    test_data = data_loader.load_data("test")
    
    print(f"Loaded {len(train_data)} training graphs, {len(val_data)} validation graphs, {len(test_data)} test graphs")
    
    # Get dataset statistics
    sample_data = train_data[0]
    in_dim = sample_data.x.shape[1]
    num_classes = sample_data.y.shape[1]
    
    print(f"Input dimension: {in_dim}")
    print(f"Number of classes (GO terms): {num_classes}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Initialize model
    model = PPIMultiLabelClassifier(
        in_dim=in_dim,
        hidden_dim=256,
        num_classes=num_classes,
        num_layers=4,
        dropout=0.2,
        gnn_type='gatv2',
        device=device,
        lr=0.001,
        weight_decay=5e-4,
        epochs=200,
        threshold=0.5
    )
    
    print("Training model...")
    train_losses, val_f1_scores = model.fit(train_loader, val_loader, verbose=True)
    
    print("Evaluating on test set...")
    test_results = model.evaluate_detailed(test_loader)
    
    print("\n=== Test Results ===")
    print(f"Micro F1: {test_results['micro_f1']:.4f}")
    print(f"Macro F1: {test_results['macro_f1']:.4f}")
    print(f"Samples F1: {test_results['samples_f1']:.4f}")
    print(f"Jaccard Score: {test_results['jaccard_score']:.4f}")
    
    # Save predictions
    print("\nSaving predictions...")
    pred_df = model.save_predictions_to_csv(test_loader, "ppi_test_predictions.csv")
    
    # Create visualizations
    print("Creating visualizations...")
    model.visualize_results(test_results, save_plots=True)
    
    # Additional analysis
    print("\n=== Additional Analysis ===")
    true_labels = test_results['true_labels']
    predictions = test_results['predictions']
    
    print(f"Average number of true labels per node: {np.mean(np.sum(true_labels, axis=1)):.2f}")
    print(f"Average number of predicted labels per node: {np.mean(np.sum(predictions, axis=1)):.2f}")
    print(f"Most frequent GO terms (by true labels):")
    
    label_counts = np.sum(true_labels, axis=0)
    top_labels = np.argsort(label_counts)[-10:]
    
    for idx in reversed(top_labels):
        go_term = model.go_terms[idx]
        count = label_counts[idx]
        f1_score = test_results['per_class_f1'][idx]
        print(f"  {go_term}: {count} samples, F1={f1_score:.3f}")
    
    print(f"\nResults saved to:")
    print(f"  - Predictions: ppi_test_predictions.csv") 
    print(f"  - Visualization: ppi_classification_results.png")

if __name__ == "__main__":
    main()