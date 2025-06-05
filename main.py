import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from gram import GRAM
from dominant import DOMINANT
from conad import CONAD
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


method = 'gram'
# method = 'dominant'

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



if method == 'gram':
    model.fit(train_loader, in_feats)
elif method == 'dominant':
    model.fit(train_loader, in_feats)
elif method == 'conad':
    model.fit(train_loader, in_feats)
elif method == 'guide':
    model.fit(train_loader, in_feats, adj_feats)
elif method == 'gcnae':
    model.fit(train_loader, in_feats)
elif method == 'gaan':
    model.fit(train_loader, in_feats)
elif method == 'ocgnn':
    model.fit(train_loader, in_feats)

t = 0
score_pred = np.zeros(len(test_dataset))
graph_label = np.zeros(len(test_dataset))

for data in test_loader:
    if data.y == 1:
        y_label = 1
    else:
        y_label = 0

    if method == 'gram':
        data = data.to(device)
        scores = model.gradcam(data)
        print('scores')
        print(scores)
        score_pred[t] = sum(scores)
        graph_label[t] = y_label
    elif method == 'dominant':
        data = data.to(device)
        scores = model.decision_function(data)
        print('scores')
        print(scores)
        score_pred[t] = sum(scores)
        graph_label[t] = y_label
    elif method == 'conad':
        data = data.to(device)
        scores = model.decision_function(data)
        print('scores')
        print(scores)
        score_pred[t] = sum(scores)
        graph_label[t] = y_label
    elif method == 'guide':
        data = data.to(device)
        scores = model.decision_function(data)
        print('scores')
        print(scores)
        score_pred[t] = sum(scores)
        graph_label[t] = y_label
    elif method == 'gcnae':
        data = data.to(device)
        scores = model.decision_function(data)
        print('scores')
        print(scores)
        score_pred[t] = sum(scores)
        graph_label[t] = y_label
    elif method == 'gaan':
        data = data.to(device)
        scores = model.decision_function(data)
        print('scores')
        print(scores)
        score_pred[t] = sum(scores)
        graph_label[t] = y_label
    elif method == 'ocgnn':
        data = data.to(device)
        scores = model.decision_function(data)
        print('scores')
        print(scores)
        score_pred[t] = sum(scores)
        graph_label[t] = y_label
    t += 1

graph_roc_auc = roc_auc_score(graph_label, score_pred)
graph_ap = average_precision_score(graph_label, score_pred)

print(f'AUC graph: {graph_roc_auc:.6f}, AP graph: {graph_ap:.6f}')

