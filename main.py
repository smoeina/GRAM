import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from gram import GRAM
from dominant import DOMINANT
from conad import CONAD
from gram_v2 import GNNVariant
from guide import GUIDE
from gcnae import GCNAE
from gaan import GAAN
from ocgnn import OCGNN
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from util import load_data

random.seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(12345)

name_dataset = 'PTC'
degree_as_tag = 'store_true'
graphs, num_classes = load_data(name_dataset, degree_as_tag)

node_feat_type = 1
if node_feat_type == 0:
    in_feats = 1

dataset = []
for graph in graphs:
    if node_feat_type:
        x = torch.tensor(graph.node_features.clone().detach(), dtype=torch.float)
        in_feats = graph.node_features.shape[1]
        adj_feats = graph.node_features.shape[0]
    else:
        x = torch.ones((graph.node_features.shape[0], in_feats), dtype=torch.float)
    edge_index = torch.tensor(graph.edge_mat.clone().detach().T, dtype=torch.long).t().contiguous()
    y = torch.tensor([graph.label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y).to(device)
    dataset.append(data)

train_dataset = dataset[167:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = dataset[137:167]
test_loader = DataLoader(test_dataset, batch_size=1)

method = 'gnn_variant'  # <-- set active method

if method == 'gram':
    model = GRAM()
elif method == 'dominant':
    model = DOMINANT()
elif method == 'conad':
    model = CONAD()
elif method == 'guide':
    model = GUIDE()
elif method == 'gcnae':
    model = GCNAE()
elif method == 'gaan':
    model = GAAN()
elif method == 'ocgnn':
    model = OCGNN()
elif method == 'gnn_variant':
    model = GNNVariant(
        in_channels=in_feats,
        hidden_channels=64,
        num_layers=3,
        out_channels=num_classes,
        dropout=0.5,
        gnn_type='gatv2'
    ).to(device)

# Train model
if method in ['gram', 'dominant', 'conad', 'gcnae', 'gaan', 'ocgnn']:
    model.fit(train_loader, in_feats)
elif method == 'guide':
    model.fit(train_loader, in_feats, adj_feats)
elif method == 'gnn_variant':
    model.fit(train_loader, device=device)

# Test model
t = 0
score_pred = np.zeros(len(test_dataset))
graph_label = np.zeros(len(test_dataset))

for data in test_loader:
    y_label = 1 if data.y == 1 else 0
    data = data.to(device)

    if method == 'gram':
        scores = model.gradcam(data)
    elif method in ['dominant', 'conad', 'guide', 'gcnae', 'gaan', 'ocgnn']:
        scores = model.decision_function(data)
    elif method == 'gnn_variant':
        scores = model.decision_function(data)
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    print('scores')
    print(scores)
    score_pred[t] = sum(scores)
    graph_label[t] = y_label
    t += 1

graph_roc_auc = roc_auc_score(graph_label, score_pred)
graph_ap = average_precision_score(graph_label, score_pred)

print(f'AUC graph: {graph_roc_auc:.6f}, AP graph: {graph_ap:.6f}')
