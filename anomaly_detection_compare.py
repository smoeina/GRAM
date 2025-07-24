import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.metrics import roc_auc_score, average_precision_score

from gram import GRAM
from gram_v2 import GNNVariantAnomalyDetector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
number_of_epoches = 200

# List of datasets to evaluate
datasets_to_run = [
    'MUTAG', 'NCI1', 'PROTEINS',
    'IMDB-BINARY', 'REDDIT-BINARY',
    'IMDB-MULTI', 'REDDIT-MULTI-5K', 'SYNTHETIC'
]

all_results = {}

for dataset_name in datasets_to_run:
    print(f'\n\n=== Processing dataset: {dataset_name} ===')

    # Load dataset
    try:
        dataset = TUDataset(root='./dataset', name=dataset_name)
    except Exception as e:
        print(f"Could not load dataset {dataset_name}: {e}")
        continue

    print(f'Loaded {len(dataset)} graphs.')

    # Shuffle dataset for splitting
    dataset = dataset.shuffle()
    split_idx = int(0.75 * len(dataset))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Determine input feature size
    if hasattr(dataset[0], 'x') and dataset[0].x is not None:
        in_feats = dataset[0].x.shape[1]
    else:
        in_feats = 1
        for data in dataset:
            data.x = torch.ones((data.num_nodes, in_feats))  # Add dummy features

    # Initialize models
    gram_model = GRAM(epoch=100)
    gnn_variant_model = GNNVariantAnomalyDetector(
        in_dim=in_feats,
        hidden_dim=128,
        latent_dim=64,
        num_layers=8,
        dropout=0.0,
        alpha=0.25,
        gnn_type='gatv2',
        device=device,
        lr=5e-4,
        weight_decay=0.,
        epochs=number_of_epoches
    )

    # Train models
    print('Training GRAM...')
    gram_model.fit(train_loader, in_feats)
    print('Training GNNVariant...')
    gnn_variant_model.fit(train_loader)

    # Evaluate
    dataset_results = {}
    for name, model, score_fn in [
        ('GRAM', gram_model, lambda data: model.gradcam(data)),
        ('GNNVariant', gnn_variant_model, lambda data: model.decision_function(data))
    ]:
        score_pred = np.zeros(len(test_dataset))
        graph_label = np.zeros(len(test_dataset))
        for t, data in enumerate(test_loader):
            y_label = int(data.y.item())
            data = data.to(device)
            scores = score_fn(data)
            score_pred[t] = np.sum(scores)
            graph_label[t] = y_label

        if np.isnan(score_pred).any():
            print(f"Warning: NaN detected in {name} score_pred. Replacing NaNs with 0.")
            score_pred = np.nan_to_num(score_pred, nan=0.0)

        auc = roc_auc_score(graph_label, score_pred)
        ap = average_precision_score(graph_label, score_pred)
        dataset_results[name] = {'AUC': auc, 'AP': ap}
        print(f'{name} on {dataset_name}: AUC={auc:.4f}, AP={ap:.4f}')

    all_results[dataset_name] = dataset_results

# Print Summary
print('\n\n=== FINAL RESULTS SUMMARY ===')
for dataset_name, result in all_results.items():
    print(f'\nDataset: {dataset_name}')
    for model_name, metrics in result.items():
        print(f'  {model_name}: AUC={metrics["AUC"]:.4f}, AP={metrics["AP"]:.4f}')
