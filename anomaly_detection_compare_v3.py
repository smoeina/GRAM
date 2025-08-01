import os                      # ← NEW
import contextlib              # ← NEW
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import time
from collections import defaultdict
# Import models
from gram import GRAM
from gram_v2 import GNNVariantAnomalyDetector
from gram_v3 import GNNVariantAnomalyDetector as GRAMv3  # Enhanced version

# Import custom metrics
from metrics import (
    eval_roc_auc,
    eval_average_precision,
    eval_recall_at_k,
    eval_precision_at_k,
    eval_ndcg
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------- Runtime perf knobs ----------
# Limit/coordinate CPU threads so dataloader doesn't thrash the CPU
try:
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    torch.set_num_interop_threads(min(8, os.cpu_count() or 8))
except Exception:
    pass

if device.type == 'cuda':
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    # PyTorch 2.x: Slightly lower precision matmuls often speed up kernels
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

# Configuration
config = {
    'epochs': 100,
    'batch_size': 16,
    'train_split': 0.70,
    'early_stopping_patience': 50,
    'min_improvement': 1e-4,
    # ---- NEW: pipeline knobs (tuned for CUDA) ----
    'num_workers': max(2, (os.cpu_count() or 4) // 2),
    'prefetch_factor': 4,
    'persistent_workers': True
}
def _loader_kwargs_for_device():
    """Return DataLoader kwargs tuned for current device."""
    if device.type == 'cuda':
        kw = dict(
            num_workers=config['num_workers'],
            pin_memory=True,
            persistent_workers=config['persistent_workers']
        )
        # prefetch_factor is only valid when num_workers > 0
        if config['num_workers'] > 0:
            kw['prefetch_factor'] = config['prefetch_factor']
        # PyTorch ≥2.0 supports pin_memory_device
        if 'pin_memory_device' in DataLoader.__init__.__code__.co_varnames:
            kw['pin_memory_device'] = 'cuda'
        return kw
    # CPU-only: fewer workers prevents oversubscription
    return dict(num_workers=max(1, (os.cpu_count() or 2) // 4), pin_memory=False)
def _place_model_on_device(model, device):
    """Best-effort move of custom wrappers/modules to target device."""
    # Many custom wrappers expose .model (nn.Module) and/or .to()
    try:
        if hasattr(model, 'to'):
            model.to(device)
    except Exception:
        pass
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'to'):
            model.model.to(device)
    except Exception:
        pass
    try:
        if hasattr(model, 'device'):
            model.device = device
    except Exception:
        pass
# List of datasets to evaluate
datasets_to_run = [
    'ER_MD','MUTAG','highschool_ct1'
]

# Model configurations for GRAM v3
gram_v3_configs = [
    {
        'name': 'GRAM_v3_Standard',
        'config': {
            'gnn_type': 'gatv2',
            'decoder_type': 'bilinear',
            'use_contrastive': True,
            'use_adaptive_alpha': True,
            'use_multi_scale': True,
            'hidden_dim': 128,
            'latent_dim': 64,
            'num_layers': 6,
            'dropout': 0.1,
            'alpha': 0.5,
            'lr': 5e-4,
            'weight_decay': 1e-5,
            'contrastive_weight': 0.1,
            'degree_weight': 0.05
        }
    },
    {
        'name': 'GRAM_v3_Transformer',
        'config': {
            'gnn_type': 'transformer',
            'decoder_type': 'mlp',
            'use_contrastive': True,
            'use_adaptive_alpha': True,
            'use_multi_scale': False,  # Transformer already captures multi-scale info
            'hidden_dim': 128,
            'latent_dim': 64,
            'num_layers': 4,
            'dropout': 0.15,
            'alpha': 0.3,
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'contrastive_weight': 0.15,
            'degree_weight': 0.03
        }
    },
    {
        'name': 'GRAM_v3_Lightweight',
        'config': {
            'gnn_type': 'sage',
            'decoder_type': 'bilinear',
            'use_contrastive': False,
            'use_adaptive_alpha': True,
            'use_multi_scale': False,
            'hidden_dim': 64,
            'latent_dim': 32,
            'num_layers': 3,
            'dropout': 0.05,
            'alpha': 0.4,
            'lr': 1e-3,
            'weight_decay': 5e-5,
            'contrastive_weight': 0.0,
            'degree_weight': 0.02
        }
    }
]


def compute_comprehensive_metrics(labels, scores, k_values=[5, 10, 20]):
    """Compute all evaluation metrics including custom ones from metrics.py"""
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
        auc = eval_roc_auc(labels, scores)
        ap = eval_average_precision(labels, scores)
        ndcg = eval_ndcg(labels, scores)
    except ValueError as e:
        print(f"Error computing basic metrics: {e}")
        auc, ap, ndcg = 0.0, 0.0, 0.0

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


def evaluate_model_with_interpretability(model, test_loader, model_name, device):
    """Enhanced evaluation with interpretability analysis and comprehensive metrics"""
    model.model.eval() if hasattr(model, 'model') else model.eval()

    score_pred = []
    graph_labels = []
    interpretability_scores = []

    def _run_decision_function(m, d, allow_interp):
        if allow_interp:
            try:
                out = m.decision_function(d, return_interpretability=True)
                if isinstance(out, tuple) and len(out) == 2:
                    return out  # (scores, interp)
                return m.decision_function(d), None
            except TypeError:
                return m.decision_function(d), None
        else:
            return m.decision_function(d), None

    # ---- NEW: autocast context only on CUDA ----
    autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' else contextlib.nullcontext()

    with torch.no_grad():
        # keep this outside the loop
        autocast_eval = (torch.autocast(device_type='cuda', dtype=torch.float16)
                         if device.type == 'cuda' else contextlib.nullcontext())

        for data in test_loader:
            y_label = int(data.y.item())
            data = data.to(device, non_blocking=(device.type == 'cuda'))

            try:
                scores, interp = None, None
                prefer_gradcam = model_name.lower() == 'gram_original'

                # ---- 1) Prefer GRAD-CAM for GRAM_Original (needs gradients, run in FP32) ----
                if prefer_gradcam and hasattr(model, 'gradcam'):
                    with torch.enable_grad(), torch.cuda.amp.autocast(
                            enabled=False) if device.type == 'cuda' else contextlib.nullcontext():
                        if hasattr(model, 'zero_grad'):
                            model.zero_grad(set_to_none=True)
                        if hasattr(model, 'model') and hasattr(model.model, 'zero_grad'):
                            model.model.zero_grad(set_to_none=True)
                        scores = model.gradcam(data)

                # ---- 2) Try decision_function under no_grad (cheap, no gradients) ----
                if scores is None and hasattr(model, 'decision_function'):
                    allow_interp = 'v3' in model_name.lower()
                    with torch.no_grad(), autocast_eval:
                        scores, interp = _run_decision_function(model, data, allow_interp)

                # ---- 3) Fallback to GRAD-CAM if needed (again needs gradients) ----
                if scores is None and hasattr(model, 'gradcam'):
                    with torch.enable_grad(), torch.cuda.amp.autocast(
                            enabled=False) if device.type == 'cuda' else contextlib.nullcontext():
                        if hasattr(model, 'zero_grad'):
                            model.zero_grad(set_to_none=True)
                        if hasattr(model, 'model') and hasattr(model.model, 'zero_grad'):
                            model.model.zero_grad(set_to_none=True)
                        scores = model.gradcam(data)

                # ---- aggregation (unchanged) ----
                if scores is None:
                    print(f"Warning: {model_name} returned None scores for sample {len(score_pred)}")
                    graph_score = 0.5
                else:
                    if isinstance(scores, np.ndarray):
                        graph_score = float(scores.sum()) if scores.size > 1 else float(scores)
                    elif isinstance(scores, torch.Tensor):
                        graph_score = float(scores.sum().detach().cpu()) if scores.numel() > 1 else float(
                            scores.detach().cpu())
                    elif isinstance(scores, (int, float)):
                        graph_score = float(scores)
                    else:
                        print(f"Warning: Unexpected score type {type(scores)} from {model_name}")
                        graph_score = 0.5

                interpretability_scores.append(interp)

            except Exception as e:
                print(f"Error evaluating {model_name} on sample {len(score_pred)}: {e}")
                import traceback;
                traceback.print_exc()
                graph_score = 0.5
                interpretability_scores.append(None)

            score_pred.append(graph_score)
            graph_labels.append(y_label)

    score_pred = np.array(score_pred)
    graph_labels = np.array(graph_labels)

    if np.isnan(score_pred).any():
        print(f"Warning: NaN detected in {model_name} scores. Replacing with median.")
        median_score = np.nanmedian(score_pred)
        if np.isnan(median_score):
            median_score = 0.5
        score_pred = np.nan_to_num(score_pred, nan=median_score)

    print(f"Score statistics for {model_name}:")
    print(f"  Min: {np.min(score_pred):.4f}, Max: {np.max(score_pred):.4f}")
    print(f"  Mean: {np.mean(score_pred):.4f}, Std: {np.std(score_pred):.4f}")
    print(f"  Unique values: {len(np.unique(score_pred))}")

    metrics = compute_comprehensive_metrics(graph_labels, score_pred)
    metrics.update({'scores': score_pred, 'labels': graph_labels, 'interpretability': interpretability_scores})
    return metrics



def train_with_early_stopping(model, train_loader, patience=50, min_improvement=1e-4):
    """Train model with early stopping and better error handling"""
    if hasattr(model, 'fit'):
        try:
            # For our custom models
            print("Training with standard fit method...")
            model.fit(train_loader)
            print("Training completed successfully.")
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            raise e
    else:
        raise ValueError("Model does not have a fit method")


def test_model_interface(model, test_data, model_name):
    """Test model interface to ensure it works correctly"""
    print(f"Testing interface for {model_name}...")

    try:
        used_any = False

        if hasattr(model, 'decision_function'):
            print(f"  ✓ Has decision_function method")
            test_scores = model.decision_function(test_data)
            print(f"  ✓ decision_function returns: {type(test_scores)}")
            if test_scores is not None:
                shape_or_val = getattr(test_scores, 'shape', None)
                print(f"    Shape/Value: {shape_or_val if shape_or_val is not None else test_scores}")
                used_any = True
            else:
                print(f"    Warning: decision_function returned None")

        # Try gradcam either when preferred for GRAM_Original or when decision_function was None
        if hasattr(model, 'gradcam') and (model_name.lower() == 'gram_original' or not used_any):
            print(f"  ✓ Has gradcam method (fallback check)")
            gc_scores = model.gradcam(test_data)
            print(f"  ✓ gradcam returns: {type(gc_scores)}")
            if gc_scores is not None:
                shape_or_val = getattr(gc_scores, 'shape', None)
                print(f"    Shape/Value: {shape_or_val if shape_or_val is not None else gc_scores}")
                used_any = True
            else:
                print(f"    Warning: gradcam returned None")

        if not used_any:
            print(f"  ✗ No valid scoring output (both methods returned None)")

    except Exception as e:
        print(f"  ✗ Interface test failed: {e}")
        import traceback; traceback.print_exc()

    print("Interface test completed.\n")


def prepare_dataset(dataset_name):
    """Load and prepare dataset with proper error handling"""
    try:
        dataset = TUDataset(root='./dataset', name=dataset_name)
        print(f'Loaded {len(dataset)} graphs from {dataset_name}')

        # Features
        if hasattr(dataset[0], 'x') and dataset[0].x is not None:
            in_feats = dataset[0].x.shape[1]
            print(f'Dataset has {in_feats} node features')
        else:
            in_feats = 1
            print('Dataset has no node features, adding dummy features')
            for data in dataset:
                data.x = torch.ones((data.num_nodes, in_feats))

        # Shuffle & split
        dataset = dataset.shuffle()
        split_idx = int(config['train_split'] * len(dataset))
        train_dataset = dataset[:split_idx]
        test_dataset = dataset[split_idx:]

        # ---- NEW: tuned loader kwargs ----
        loader_kwargs = _loader_kwargs_for_device()

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            **loader_kwargs
        )
        # keep test batch_size=1 to stay simple per-graph
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            **loader_kwargs
        )

        return train_loader, test_loader, in_feats, len(train_dataset), len(test_dataset)

    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None, None, None, None, None



def initialize_models(in_feats):
    """Initialize all models for comparison"""
    models = {}

    # Original GRAM
    try:
        models['GRAM_Original'] = GRAM(epoch=100)
    except Exception as e:
        print(f"Failed to initialize GRAM Original: {e}")

    # GRAM v2
    try:
        models['GRAM_v2'] = GNNVariantAnomalyDetector(
            in_dim=in_feats,
            hidden_dim=128,
            latent_dim=64,
            num_layers=6,
            dropout=0.1,
            alpha=0.25,
            gnn_type='gatv2',
            device=device,
            lr=5e-4,
            weight_decay=1e-5,
            epochs=config['epochs']
        )
    except Exception as e:
        print(f"Failed to initialize GRAM v2: {e}")

    # GRAM v3 variants
    for variant in gram_v3_configs:
        try:
            variant_config = variant['config'].copy()
            variant_config.update({
                'in_dim': in_feats,
                'device': device,
                'epochs': config['epochs']
            })
            models[variant['name']] = GRAMv3(**variant_config)
        except Exception as e:
            print(f"Failed to initialize {variant['name']}: {e}")

    return models


# Main evaluation loop
all_results = {}
timing_results = defaultdict(dict)

for dataset_name in datasets_to_run:
    print(f'\n{"=" * 60}')
    print(f'Processing dataset: {dataset_name}')
    print(f'{"=" * 60}')

    # Prepare dataset
    data_prep = prepare_dataset(dataset_name)
    if data_prep[0] is None:
        continue

    train_loader, test_loader, in_feats, n_train, n_test = data_prep
    print(f'Train samples: {n_train}, Test samples: {n_test}')

    # Initialize models
    models = initialize_models(in_feats)
    if not models:
        print(f"No models could be initialized for {dataset_name}")
        continue
    for _m in models.values():
        _place_model_on_device(_m, device)
    # Train and evaluate each model
    dataset_results = {}

    # Get a test sample for interface testing
    test_sample = next(iter(test_loader)).to(device, non_blocking=(device.type == 'cuda'))


    for model_name, model in models.items():
        print(f'\n--- Training {model_name} ---')

        try:
            # Training
            start_time = time.time()

            if model_name == 'GRAM_Original':
                model.fit(train_loader, in_feats)
            else:
                train_with_early_stopping(model, train_loader,
                                          patience=config['early_stopping_patience'],
                                          min_improvement=config['min_improvement'])

            training_time = time.time() - start_time
            timing_results[dataset_name][model_name] = training_time
            print(f'Training completed in {training_time:.2f} seconds')

            # Test model interface before evaluation
            test_model_interface(model, test_sample, model_name)

            # Evaluation
            print(f'Evaluating {model_name}...')
            start_time = time.time()

            results = evaluate_model_with_interpretability(model, test_loader, model_name, device)

            evaluation_time = time.time() - start_time
            print(f'Evaluation completed in {evaluation_time:.2f} seconds')

            dataset_results[model_name] = results

            # Print comprehensive results
            print(f'{model_name} Results:')
            print(f'  AUC: {results["AUC"]:.4f}')
            print(f'  AP: {results["AP"]:.4f}')
            print(f'  NDCG: {results["NDCG"]:.4f}')
            print(f'  Recall@5: {results["Recall@5"]:.4f}')
            print(f'  Recall@10: {results["Recall@10"]:.4f}')
            print(f'  Precision@5: {results["Precision@5"]:.4f}')
            print(f'  Precision@10: {results["Precision@10"]:.4f}')

        except Exception as e:
            print(f'Error with {model_name}: {e}')
            import traceback

            traceback.print_exc()
            continue

    all_results[dataset_name] = dataset_results

# Print Final Results
print(f'\n{"=" * 120}')
print('FINAL RESULTS SUMMARY')
print(f'{"=" * 120}')

# Comprehensive results table
metric_names = ['AUC', 'AP', 'NDCG', 'Recall@5', 'Recall@10', 'Precision@5', 'Precision@10']
header = f'{"Dataset":<15} {"Model":<20}'
for metric in metric_names:
    header += f' {metric:<10}'
header += f' {"Time":<10}'
print(header)
print('-' * 120)

for dataset_name, results in all_results.items():
    for model_name, metrics in results.items():
        train_time = timing_results[dataset_name].get(model_name, 0)
        row = f'{dataset_name:<15} {model_name:<20}'
        for metric in metric_names:
            row += f' {metrics.get(metric, 0.0):<10.4f}'
        row += f' {train_time:<10.2f}s'
        print(row)

# Average performance comparison
print(f'\n{"=" * 100}')
print('AVERAGE PERFORMANCE ACROSS DATASETS')
print(f'{"=" * 100}')

model_averages = defaultdict(lambda: {metric: [] for metric in metric_names + ['time']})

for dataset_results in all_results.values():
    for model_name, metrics in dataset_results.items():
        for metric in metric_names:
            model_averages[model_name][metric].append(metrics.get(metric, 0.0))

for dataset_times in timing_results.values():
    for model_name, time_val in dataset_times.items():
        model_averages[model_name]['time'].append(time_val)

# Print averages table
header = f'{"Model":<20}'
for metric in metric_names:
    header += f' {"Avg " + metric:<12}'
header += f' {"Avg Time":<12}'
print(header)
print('-' * 100)

for model_name, metrics in model_averages.items():
    row = f'{model_name:<20}'
    for metric in metric_names:
        avg_val = np.mean(metrics[metric]) if metrics[metric] else 0
        row += f' {avg_val:<12.4f}'
    avg_time = np.mean(metrics['time']) if metrics['time'] else 0
    row += f' {avg_time:<12.2f}s'
    print(row)

# Best model identification for each metric
print(f'\n{"=" * 60}')
print('BEST PERFORMING MODELS BY METRIC')
print(f'{"=" * 60}')

for metric in metric_names:
    try:
        best_model = max(model_averages.items(),
                         key=lambda x: np.mean(x[1][metric]) if x[1][metric] else 0)
        best_score = np.mean(best_model[1][metric]) if best_model[1][metric] else 0
        print(f'Best {metric:<12}: {best_model[0]:<20} ({best_score:.4f})')
    except (ValueError, KeyError):
        print(f'Best {metric:<12}: Unable to determine')

# Statistical significance testing (if multiple datasets)
if len(all_results) > 1:
    print(f'\n{"=" * 60}')
    print('STATISTICAL ANALYSIS')
    print(f'{"=" * 60}')

    from scipy import stats

    # Compare top 2 models on AUC
    auc_scores = {model: [results[model]['AUC'] for results in all_results.values()
                          if model in results]
                  for model in model_averages.keys()}

    # Find top 2 models by average AUC
    top_models = sorted(auc_scores.items(),
                        key=lambda x: np.mean(x[1]), reverse=True)[:2]

    if len(top_models) == 2:
        model1_name, model1_scores = top_models[0]
        model2_name, model2_scores = top_models[1]

        if len(model1_scores) > 1 and len(model2_scores) > 1:
            try:
                t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
                print(f'T-test between {model1_name} and {model2_name}:')
                print(f'  T-statistic: {t_stat:.4f}')
                print(f'  P-value: {p_value:.4f}')
                print(f'  Significant difference: {"Yes" if p_value < 0.05 else "No"}')
            except Exception as e:
                print(f'Statistical test failed: {e}')

# Create detailed performance report
print(f'\n{"=" * 60}')
print('DETAILED PERFORMANCE BREAKDOWN')
print(f'{"=" * 60}')

for dataset_name, results in all_results.items():
    print(f'\n--- {dataset_name} ---')

    # Sort models by AUC for this dataset
    sorted_models = sorted(results.items(), key=lambda x: x[1]['AUC'], reverse=True)

    for rank, (model_name, metrics) in enumerate(sorted_models, 1):
        print(f'{rank}. {model_name}:')
        for metric in ['AUC', 'AP', 'NDCG']:
            print(f'   {metric}: {metrics.get(metric, 0.0):.4f}')
        print(f'   Training Time: {timing_results[dataset_name].get(model_name, 0):.2f}s')

# Save comprehensive results to file
import json
import scipy.stats

results_summary = {
    'config': config,
    'gram_v3_configs': gram_v3_configs,
    'results': {},
    'timing': dict(timing_results),
    'averages': {},
    'best_models': {}
}

# Process results for JSON serialization
for dataset, results in all_results.items():
    results_summary['results'][dataset] = {}
    for model, metrics in results.items():
        # Only include serializable metrics
        serializable_metrics = {k: float(v) for k, v in metrics.items()
                                if k not in ['scores', 'labels', 'interpretability']
                                and isinstance(v, (int, float, np.number))}
        results_summary['results'][dataset][model] = serializable_metrics

# Calculate and store averages
for model, metrics in model_averages.items():
    results_summary['averages'][model] = {}
    for metric_name in metric_names + ['time']:
        if metrics[metric_name]:
            results_summary['averages'][model][metric_name] = {
                'mean': float(np.mean(metrics[metric_name])),
                'std': float(np.std(metrics[metric_name])),
                'min': float(np.min(metrics[metric_name])),
                'max': float(np.max(metrics[metric_name]))
            }
        else:
            results_summary['averages'][model][metric_name] = {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0
            }

# Store best models for each metric
for metric in metric_names:
    try:
        best_model = max(model_averages.items(),
                         key=lambda x: np.mean(x[1][metric]) if x[1][metric] else 0)
        results_summary['best_models'][metric] = {
            'model': best_model[0],
            'score': float(np.mean(best_model[1][metric])) if best_model[1][metric] else 0.0
        }
    except (ValueError, KeyError):
        results_summary['best_models'][metric] = {'model': 'Unknown', 'score': 0.0}

with open('comprehensive_gram_evaluation_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f'\nComprehensive results saved to comprehensive_gram_evaluation_results.json')

# Generate LaTeX table for paper
print(f'\n{"=" * 60}')
print('LATEX TABLE FOR PAPER')
print(f'{"=" * 60}')

print("\\begin{tabular}{l|cccc|cc}")
print("\\hline")
print("Dataset & Model & AUC & AP & NDCG & R@10 & P@10 & Time \\\\")
print("\\hline")

for dataset_name, results in all_results.items():
    dataset_printed = False
    # Sort by AUC descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]['AUC'], reverse=True)
    for model_name, metrics in sorted_results:
        dataset_col = dataset_name if not dataset_printed else ""
        train_time = timing_results[dataset_name].get(model_name, 0)
        print(f"{dataset_col} & {model_name} & "
              f"{metrics['AUC']:.3f} & {metrics['AP']:.3f} & {metrics['NDCG']:.3f} & "
              f"{metrics['Recall@10']:.3f} & {metrics['Precision@10']:.3f} & "
              f"{train_time:.1f}s \\\\")
        dataset_printed = True
    print("\\hline")

print("\\end{tabular}")

print(f'\nEvaluation completed successfully!')
print(f'Total datasets processed: {len(all_results)}')
print(f'Total models evaluated: {len(model_averages)}')
print(f'Best overall model (AUC): {results_summary["best_models"]["AUC"]["model"]}')