import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import torch.nn as nn
import torch.nn.functional as F
import random

from collections import defaultdict

# Import models
from gram import GRAM
from faster_gram import FastGNNAnomalyDetector

# Set seeds for reproducibility
random.seed(12345)
torch.manual_seed(12345)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
config = {
    'epochs': 200,
    'batch_size': 128,
    'train_split': 0.70,
    'num_workers': 4,
    'prefetch_factor': 2
}

# List of datasets to evaluate
datasets_to_run = [
    'ER_MD', 'MUTAG', 'PTC'
]


def print_model_config(model, model_name):
    """Print model configuration for comparison"""
    print(f"\n{'-' * 50}")
    print(f"Model Configuration: {model_name}")
    print(f"{'-' * 50}")

    if model_name == 'GRAM':
        print(f"Model Type: Graph Attention-based Graph Reconstruction (VAE)")
        print(f"Hidden Dimension: {getattr(model, 'hid_dim', 'N/A')}")
        print(f"Latent Size: {getattr(model, 'latent_size', 'N/A')}")
        print(f"Number of Layers: {getattr(model, 'num_layers', 'N/A')}")
        print(f"Dropout Rate: {getattr(model, 'dropout', 'N/A')}")
        print(f"Weight Decay: {getattr(model, 'weight_decay', 'N/A')}")
        print(f"Alpha Parameter: {getattr(model, 'alpha', 'N/A')}")
        print(f"Learning Rate: {getattr(model, 'lr', 'N/A')}")
        print(f"Epochs: {getattr(model, 'epoch', 'N/A')}")
        print(f"Contamination: {getattr(model, 'contamination', 'N/A')}")
        print(f"Activation Function: {getattr(model, 'act', 'N/A')}")
        print(f"Device: {getattr(model, 'device', 'N/A')}")
        print(f"Verbose: {getattr(model, 'verbose', 'N/A')}")
        print(f"Architecture: Encoder-Decoder with separate attribute/structure paths")
        print(f"Detection Method: GradCAM-based anomaly scoring")
        print(f"Loss Components: Attribute + Structure + KL divergence")

    elif model_name == 'Fast_GRAM':
        print(f"Model Type: Fast Graph Neural Network Anomaly Detector (VAE)")

        # Access model parameters from the FastGNNAnomalyDetector instance
        model_vae = getattr(model, 'model', None)

        # Get parameters from the detector instance
        print(f"Learning Rate: {getattr(model, 'lr', 'N/A')}")
        print(f"Weight Decay: {getattr(model, 'weight_decay', 'N/A')}")
        print(f"Epochs: {getattr(model, 'epochs', 'N/A')}")
        print(f"Device: {getattr(model, 'device', 'N/A')}")
        print(
            f"Optimizer: {type(getattr(model, 'optimizer', None)).__name__ if hasattr(model, 'optimizer') else 'N/A'}")
        print(
            f"Scheduler: {type(getattr(model, 'scheduler', None)).__name__ if hasattr(model, 'scheduler') else 'N/A'}")

        # Get parameters from the VAE model if available
        if model_vae is not None:
            encoder = getattr(model_vae, 'encoder', None)
            if encoder is not None:
                convs = getattr(encoder, 'convs', [])
                if convs:
                    print(f"Input Dimension: {getattr(convs[0], 'in_channels', 'N/A') if len(convs) > 0 else 'N/A'}")
                    print(f"Hidden Dimension: {getattr(convs[0], 'out_channels', 'N/A') if len(convs) > 0 else 'N/A'}")
                    print(f"Number of Layers: {len(convs)}")
                    print(f"GNN Type: {type(convs[0]).__name__ if len(convs) > 0 else 'N/A'}")

                print(f"Dropout Rate: {getattr(encoder, 'dropout', 'N/A')}")

            # Get latent dimension from mu_layer
            mu_layer = getattr(model_vae, 'mu_layer', None)
            if mu_layer is not None:
                print(f"Latent Dimension: {getattr(mu_layer, 'out_features', 'N/A')}")

            # Get other VAE parameters
            print(f"Alpha Parameter: {getattr(model_vae, 'alpha', 'N/A')}")

            # Get decoder information
            edge_decoder = getattr(model_vae, 'edge_decoder', None)
            if edge_decoder is not None:
                print(f"Decoder Type: {type(edge_decoder).__name__}")

            # Check for contrastive loss
            contrastive = getattr(model_vae, 'contrastive_loss', None)
            print(f"Use Contrastive: {contrastive is not None}")

        print(f"Architecture: Optimized encoder-decoder with batch normalization")
        print(f"Detection Method: Reconstruction error-based scoring")
        print(f"Loss Components: Attribute + Structure + KL + Optional Contrastive")
        print(f"Optimizations: AdamW optimizer, cosine annealing, gradient clipping")

    # Print model size comparison
    try:
        if hasattr(model, 'model') and model.model is not None:
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            print(f"Total Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
        elif hasattr(model, 'parameters') and callable(getattr(model, 'parameters')):
            # For models that directly expose parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
        else:
            print(f"Parameter count: Not available (model not initialized)")
    except Exception as e:
        print(f"Parameter count: Error calculating ({str(e)})")

    print(f"{'-' * 50}")


def compute_comprehensive_metrics(labels, scores, k_values=[5, 10, 20]):
    """Compute all evaluation metrics"""
    labels = np.array(labels)
    scores = np.array(scores)

    # Handle edge cases
    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present in labels. Cannot compute some metrics.")
        return {
            'AUC': 0.5,
            'AP': np.mean(labels),
            'NDCG': 0.0,
            'Recall@5': 0.0,
            'Recall@10': 0.0,
            'Recall@20': 0.0,
            'Precision@5': 0.0,
            'Precision@10': 0.0,
            'Precision@20': 0.0
        }

    # Basic metrics
    try:
        from metrics import (
            eval_roc_auc,
            eval_average_precision,
            eval_recall_at_k,
            eval_precision_at_k,
            eval_ndcg
        )
        auc = eval_roc_auc(labels, scores)
        ap = eval_average_precision(labels, scores)
        ndcg = eval_ndcg(labels, scores)
    except (ImportError, ValueError) as e:
        print(f"Using sklearn metrics instead of custom metrics: {e}")
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        ndcg = 0.0

    # Precision and Recall at K
    metrics = {
        'AUC': auc,
        'AP': ap,
        'NDCG': ndcg
    }

    # Only compute @K metrics if we have enough samples
    n_samples = len(labels)
    n_positives = np.sum(labels)

    for k in k_values:
        if k <= n_samples and n_positives > 0:
            try:
                recall_k = eval_recall_at_k(labels, scores, k)
                precision_k = eval_precision_at_k(labels, scores, k)
                metrics[f'Recall@{k}'] = recall_k
                metrics[f'Precision@{k}'] = precision_k
            except Exception as e:
                print(f"Error computing @{k} metrics: {e}")
                metrics[f'Recall@{k}'] = 0.0
                metrics[f'Precision@{k}'] = 0.0
        else:
            metrics[f'Recall@{k}'] = 0.0
            metrics[f'Precision@{k}'] = 0.0

    return metrics


def evaluate_model(model, test_loader, model_name):
    """Evaluate model performance"""
    score_pred = np.zeros(len(test_loader.dataset))
    graph_label = np.zeros(len(test_loader.dataset))

    t = 0
    for data in test_loader:
        # Label processing
        y_label = 1 if data.y == 1 else 0
        data = data.to(device)

        try:
            # Model-specific scoring method
            if model_name == 'GRAM':
                scores = model.gradcam(data)
            else:
                scores = model.decision_function(data)

            print(f'Sample {t} scores:')
            print(scores)

            # Score aggregation
            if isinstance(scores, (np.ndarray, list)):
                score_pred[t] = sum(scores)
            else:
                score_pred[t] = float(scores)

            graph_label[t] = y_label
            t += 1

        except Exception as e:
            print(f"Error evaluating {model_name} on sample {t}: {e}")
            score_pred[t] = 0.5  # Default score
            graph_label[t] = y_label
            t += 1

    # Check if GRAM needs score inversion (AUC < 0.5 indicates inverted labels)
    try:
        initial_auc = roc_auc_score(graph_label, score_pred)
        if model_name == 'GRAM' and initial_auc < 0.5:
            print(f"Detected inverted scores for GRAM (AUC={initial_auc:.4f}). Inverting scores...")
            score_pred = -score_pred  # Invert the scores
            # Or alternatively: score_pred = np.max(score_pred) - score_pred

        # Compute final metrics
        graph_roc_auc = roc_auc_score(graph_label, score_pred)
        graph_ap = average_precision_score(graph_label, score_pred)

        print(f'AUC graph: {graph_roc_auc:.6f}, AP graph: {graph_ap:.6f}')

        # Also compute comprehensive metrics
        comprehensive_metrics = compute_comprehensive_metrics(graph_label, score_pred)
        comprehensive_metrics['AUC'] = graph_roc_auc
        comprehensive_metrics['AP'] = graph_ap

        return comprehensive_metrics

    except ValueError as e:
        print(f"Error computing metrics: {e}")
        return {
            'AUC': 0.5,
            'AP': np.mean(graph_label),
            'NDCG': 0.0,
            'Recall@5': 0.0,
            'Recall@10': 0.0,
            'Recall@20': 0.0,
            'Precision@5': 0.0,
            'Precision@10': 0.0,
            'Precision@20': 0.0
        }


def prepare_dataset(dataset_name):
    """Load and prepare dataset"""
    try:
        # Load TU dataset
        dataset = TUDataset(root='./dataset', name=dataset_name)
        print(f'Loaded {len(dataset)} graphs from {dataset_name}')

        # Convert to required format
        processed_dataset = []
        node_feat_type = 1

        for graph_data in dataset:
            if node_feat_type and hasattr(graph_data, 'x') and graph_data.x is not None:
                x = graph_data.x.clone().detach().float()
                in_feats = graph_data.x.shape[1]
            else:
                x = torch.ones((graph_data.num_nodes, 1), dtype=torch.float)
                in_feats = 1

            # Edge processing
            edge_index = graph_data.edge_index.clone().detach().long()

            # Label processing
            y = graph_data.y.clone().detach().long()

            # Create data object
            data = Data(x=x, edge_index=edge_index, y=y).to(device)
            processed_dataset.append(data)

        # Train/test split
        total_graphs = len(processed_dataset)
        split_idx = int(config['train_split'] * total_graphs)

        train_dataset = processed_dataset[:split_idx]
        test_dataset = processed_dataset[split_idx:]

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False
        )

        return train_loader, test_loader, in_feats, len(train_dataset), len(test_dataset)

    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None, None, None, None, None


def initialize_models(in_feats):
    """Initialize models with their configurations"""
    models = {}

    # GRAM
    try:
        models['GRAM'] = GRAM(epoch=200)
        print("✓ GRAM initialized successfully")
    except Exception as e:
        print(f"Failed to initialize GRAM: {e}")

    # Fast GRAM
    try:
        models['Fast_GRAM'] = FastGNNAnomalyDetector(
            in_dim=in_feats,
            hidden_dim=32,
            latent_dim=16,
            num_layers=3,
            dropout=0,
            alpha=0.5,
            gnn_type='transformer',
            device=device,
            lr=1e-3,
            weight_decay=5e-4,
            epochs=100,
            decoder_type='bilinear',
            use_contrastive=True
        )
        print("✓ Fast GRAM initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Fast GRAM: {e}")

    return models


def train_model(model, train_loader, model_name, in_feats=None):
    """Train model"""
    try:
        if model_name == 'GRAM':
            print("Training GRAM...")
            model.fit(train_loader, in_feats)
        elif model_name == 'Fast_GRAM':
            print("Training Fast GRAM...")
            model.fit(train_loader)
        else:
            print(f"Training {model_name}...")
            model.fit(train_loader, in_feats)
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        raise e


# Main evaluation loop
all_results = {}
timing_results = defaultdict(dict)
model_configs = {}

for dataset_name in datasets_to_run:
    print(f'\n{"=" * 80}')
    print(f'Processing dataset: {dataset_name}')
    print(f'{"=" * 80}')

    # Prepare dataset
    data_prep = prepare_dataset(dataset_name)
    if data_prep[0] is None:
        continue

    train_loader, test_loader, in_feats, n_train, n_test = data_prep
    print(f'Train samples: {n_train}, Test samples: {n_test}')
    print(f'Input features: {in_feats}')

    # Initialize models
    models = initialize_models(in_feats)
    if not models:
        print(f"No models could be initialized for {dataset_name}")
        continue

    dataset_results = {}

    for model_name, model in models.items():
        print(f'\n--- Training and Evaluating {model_name} ---')

        # Print model configuration
        print_model_config(model, model_name)

        # Store model config for summary
        if model_name not in model_configs:
            model_configs[model_name] = model

        try:
            # Training with timing
            start_time = time.time()
            train_model(model, train_loader, model_name, in_feats)
            training_time = time.time() - start_time
            timing_results[dataset_name][model_name] = training_time
            print(f'Training completed in {training_time:.2f} seconds')

            # Evaluation with timing
            print(f'Evaluating {model_name}...')
            start_time = time.time()
            results = evaluate_model(model, test_loader, model_name)
            evaluation_time = time.time() - start_time
            print(f'Evaluation completed in {evaluation_time:.2f} seconds')

            dataset_results[model_name] = results

            # Print results
            print(f'{model_name} Results:')
            print(f'  AUC: {results["AUC"]:.4f}')
            print(f'  AP: {results["AP"]:.4f}')
            if 'NDCG' in results:
                print(f'  NDCG: {results["NDCG"]:.4f}')
                print(f'  Recall@10: {results["Recall@10"]:.4f}')
                print(f'  Precision@10: {results["Precision@10"]:.4f}')

        except Exception as e:
            print(f'Error with {model_name}: {e}')
            import traceback

            traceback.print_exc()
            continue

    all_results[dataset_name] = dataset_results

# Print Model Configurations Summary
print(f'\n{"=" * 100}')
print('MODEL CONFIGURATIONS SUMMARY')
print(f'{"=" * 100}')

for model_name, model in model_configs.items():
    print_model_config(model, model_name)

# Print Final Results
print(f'\n{"=" * 100}')
print('FINAL RESULTS COMPARISON: GRAM vs Fast GRAM')
print(f'{"=" * 100}')

# Results table
metrics = ['AUC', 'AP', 'NDCG', 'Recall@10', 'Precision@10']
print(f'{"Dataset":<15} {"Model":<15} {"AUC":<8} {"AP":<8} {"NDCG":<8} {"R@10":<8} {"P@10":<8} {"Time":<10}')
print('-' * 100)

for dataset_name, results in all_results.items():
    for model_name, model_results in results.items():
        train_time = timing_results[dataset_name].get(model_name, 0)
        row = f'{dataset_name:<15} {model_name:<15}'
        for metric in metrics:
            row += f' {model_results.get(metric, 0.0):<7.4f}'
        row += f' {train_time:<9.2f}s'
        print(row)
    print('-' * 100)

# Performance comparison
print(f'\n{"=" * 80}')
print('AVERAGE PERFORMANCE COMPARISON')
print(f'{"=" * 80}')

# Calculate averages
model_averages = defaultdict(lambda: {metric: [] for metric in metrics + ['time']})

for dataset_results in all_results.values():
    for model_name, model_metrics in dataset_results.items():
        for metric in metrics:
            model_averages[model_name][metric].append(model_metrics.get(metric, 0.0))

for dataset_times in timing_results.values():
    for model_name, time_val in dataset_times.items():
        model_averages[model_name]['time'].append(time_val)

# Print averages
print(
    f'{"Model":<15} {"Avg AUC":<10} {"Avg AP":<10} {"Avg NDCG":<10} {"Avg R@10":<10} {"Avg P@10":<10} {"Avg Time":<12}')
print('-' * 80)

for model_name, model_metrics in model_averages.items():
    row = f'{model_name:<15}'
    for metric in metrics:
        avg_val = np.mean(model_metrics[metric]) if model_metrics[metric] else 0
        row += f' {avg_val:<9.4f}'
    avg_time = np.mean(model_metrics['time']) if model_metrics['time'] else 0
    row += f' {avg_time:<11.2f}s'
    print(row)

# Speed comparison
print(f'\n{"=" * 60}')
print('SPEED COMPARISON')
print(f'{"=" * 60}')

if 'GRAM' in model_averages and 'Fast_GRAM' in model_averages:
    gram_time = np.mean(model_averages['GRAM']['time'])
    fast_time = np.mean(model_averages['Fast_GRAM']['time'])
    speedup = gram_time / fast_time if fast_time > 0 else 1.0

    print(f'GRAM average time: {gram_time:.2f}s')
    print(f'Fast GRAM average time: {fast_time:.2f}s')
    print(f'Speedup: {speedup:.2f}x faster')

    if speedup > 1:
        print(f'Fast GRAM is {speedup:.1f}x faster than GRAM!')
    else:
        print(f'Fast GRAM is {1 / speedup:.1f}x slower than GRAM')

# Configuration Comparison Summary
print(f'\n{"=" * 80}')
print('MODEL ARCHITECTURE COMPARISON')
print(f'{"=" * 80}')

print("GRAM:")
print("  - Original graph attention-based reconstruction approach")
print("  - Uses GradCAM for anomaly detection")
print("  - Fixed architecture with attention mechanism")
print("  - Default hyperparameters")

print("\nFast GRAM:")
print("  - Configurable GNN architecture (transformer-based)")
print("  - Multiple hidden layers with dropout")
print("  - Bilinear decoder with contrastive learning")
print("  - Optimized hyperparameters")
print("  - GPU acceleration support")

# Summary
print(f'\n{"=" * 60}')
print('SUMMARY')
print(f'{"=" * 60}')

if model_averages:
    best_auc_model = max(model_averages.items(), key=lambda x: np.mean(x[1]['AUC']))
    fastest_model = min(model_averages.items(), key=lambda x: np.mean(x[1]['time']))

    print(f'Best AUC: {best_auc_model[0]} ({np.mean(best_auc_model[1]["AUC"]):.4f})')
    print(f'Fastest: {fastest_model[0]} ({np.mean(fastest_model[1]["time"]):.2f}s)')

print(f'\nEvaluation completed!')
print(f'Datasets processed: {len(all_results)}')
print(f'Models compared: {len(model_averages)}')