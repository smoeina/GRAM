# Python
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from gram import GRAM
from gram_v2 import GNNVariantAnomalyDetector
from sklearn.metrics import roc_auc_score, average_precision_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
number_of_epoches = 200

# Load MUTAG dataset
dataset = TUDataset(root='./dataset', name='MUTAG')
print(f'Loaded {len(dataset)} graphs.')

# Set node feature dimension
if hasattr(dataset[0], 'x') and dataset[0].x is not None:
    in_feats = dataset[0].x.shape[1]
else:
    in_feats = 1  # fallback if no node features

# Split dataset (adjust indices as needed)
train_loader = DataLoader(dataset[:150], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset[150:], batch_size=1)
test_dataset = dataset[150:]

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
gram_model.fit(train_loader, in_feats)
gnn_variant_model.fit(train_loader)

# Evaluate models
results = {}
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
    results[name] = {'AUC': auc, 'AP': ap}
    print(f'{name}: AUC={auc:.4f}, AP={ap:.4f}')

print('Summary:')
for name, res in results.items():
    print(f'{name}: AUC={res["AUC"]:.4f}, AP={res["AP"]:.4f}')