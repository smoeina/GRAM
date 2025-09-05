import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv, TransformerConv, global_mean_pool, global_max_pool, global_add_pool
from torch.nn import ModuleList, Linear
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Optional, Callable, Union
import numpy as np


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


class GNNClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=4, dropout=0.0, act=F.gelu,
                 gnn_type='gatv2', pooling='mean', task_type='node'):
        """
        GNN Classifier for node-level or graph-level classification

        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_layers: Number of GNN layers
            dropout: Dropout rate
            act: Activation function
            gnn_type: Type of GNN ('gatv2', 'sage', 'transformer')
            pooling: Pooling method for graph-level tasks ('mean', 'max', 'add')
            task_type: 'node' for node classification, 'graph' for graph classification
        """
        super().__init__()
        self.encoder = FlexibleGNN(in_dim, hidden_dim, num_layers, gnn_type, dropout, act)
        self.pooling = pooling
        self.task_type = task_type

        # Classification head
        self.classifier = nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, num_classes)
        )

        # Initialize pooling function for graph-level tasks
        if task_type == 'graph':
            if pooling == 'mean':
                self.pool_fn = global_mean_pool
            elif pooling == 'max':
                self.pool_fn = global_max_pool
            elif pooling == 'add':
                self.pool_fn = global_add_pool
            else:
                raise ValueError(f'Unknown pooling: {pooling}')

    def forward(self, x, edge_index, batch=None):
        # Encode node features
        h = self.encoder(x, edge_index)

        # Apply pooling for graph-level tasks
        if self.task_type == 'graph' and batch is not None:
            h = self.pool_fn(h, batch)

        # Classification
        logits = self.classifier(h)
        return logits


class GNNClassificationModel:
    def __init__(self, in_dim, hidden_dim=128, num_classes=2, num_layers=4, dropout=0.0,
                 gnn_type='gatv2', device='cpu', lr=1e-3, weight_decay=5e-4, epochs=300,
                 pooling='mean', task_type='node'):
        """
        GNN Classification wrapper

        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_layers: Number of GNN layers
            dropout: Dropout rate
            gnn_type: Type of GNN ('gatv2', 'sage', 'transformer')
            device: Device to run on
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            epochs: Number of training epochs
            pooling: Pooling method for graph-level tasks ('mean', 'max', 'add')
            task_type: 'node' for node classification, 'graph' for graph classification
        """
        self.model = GNNClassifier(
            in_dim, hidden_dim, num_classes, num_layers, dropout, F.gelu,
            gnn_type, pooling, task_type
        ).to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.num_classes = num_classes
        self.task_type = task_type

    def fit(self, train_loader, val_loader=None, verbose=True):
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            verbose: Whether to print training progress
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        best_model_state = None

        for epoch in range(self.epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0

            # Training loop
            for data in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()

                logits = self.model(data.x, data.edge_index, getattr(data, 'batch', None))

                if self.task_type == 'node':
                    # For node classification, use train mask if available
                    if hasattr(data, 'train_mask'):
                        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
                        pred = logits[data.train_mask].argmax(dim=-1)
                        correct = (pred == data.y[data.train_mask]).sum().item()
                        samples = data.train_mask.sum().item()
                    else:
                        loss = criterion(logits, data.y)
                        pred = logits.argmax(dim=-1)
                        correct = (pred == data.y).sum().item()
                        samples = data.y.size(0)
                else:
                    # For graph classification
                    loss = criterion(logits, data.y)
                    pred = logits.argmax(dim=-1)
                    correct = (pred == data.y).sum().item()
                    samples = data.y.size(0)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_correct += correct
                total_samples += samples

            train_acc = total_correct / total_samples
            avg_loss = total_loss / len(train_loader)

            # Validation
            val_acc = 0
            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()

            if verbose:
                if val_loader is not None:
                    print(
                        f'Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
                else:
                    print(f'Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}')

        # Load best model if validation was used
        if val_loader is not None and best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def predict(self, data_loader):
        """
        Make predictions on data

        Args:
            data_loader: Data loader containing test data

        Returns:
            numpy array of predicted class labels
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                logits = self.model(data.x, data.edge_index, getattr(data, 'batch', None))

                if self.task_type == 'node':
                    if hasattr(data, 'test_mask'):
                        pred = logits[data.test_mask].argmax(dim=-1)
                    else:
                        pred = logits.argmax(dim=-1)
                else:
                    pred = logits.argmax(dim=-1)

                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions)

    def predict_proba(self, data_loader):
        """
        Get prediction probabilities

        Args:
            data_loader: Data loader containing test data

        Returns:
            numpy array of prediction probabilities
        """
        self.model.eval()
        probabilities = []

        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                logits = self.model(data.x, data.edge_index, getattr(data, 'batch', None))

                if self.task_type == 'node':
                    if hasattr(data, 'test_mask'):
                        probs = F.softmax(logits[data.test_mask], dim=-1)
                    else:
                        probs = F.softmax(logits, dim=-1)
                else:
                    probs = F.softmax(logits, dim=-1)

                probabilities.append(probs.cpu().numpy())

        return np.concatenate(probabilities)

    def evaluate(self, data_loader):
        """
        Evaluate model accuracy

        Args:
            data_loader: Data loader containing evaluation data

        Returns:
            accuracy score
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                logits = self.model(data.x, data.edge_index, getattr(data, 'batch', None))

                if self.task_type == 'node':
                    if hasattr(data, 'val_mask'):
                        pred = logits[data.val_mask].argmax(dim=-1)
                        correct = (pred == data.y[data.val_mask]).sum().item()
                        samples = data.val_mask.sum().item()
                    elif hasattr(data, 'test_mask'):
                        pred = logits[data.test_mask].argmax(dim=-1)
                        correct = (pred == data.y[data.test_mask]).sum().item()
                        samples = data.test_mask.sum().item()
                    else:
                        pred = logits.argmax(dim=-1)
                        correct = (pred == data.y).sum().item()
                        samples = data.y.size(0)
                else:
                    pred = logits.argmax(dim=-1)
                    correct = (pred == data.y).sum().item()
                    samples = data.y.size(0)

                total_correct += correct
                total_samples += samples

        return total_correct / total_samples

    def get_detailed_metrics(self, data_loader):
        """
        Get detailed classification metrics

        Args:
            data_loader: Data loader containing evaluation data

        Returns:
            dictionary containing accuracy, precision, recall, f1-score
        """
        predictions = self.predict(data_loader)

        # Get true labels
        true_labels = []
        for data in data_loader:
            if self.task_type == 'node':
                if hasattr(data, 'test_mask'):
                    true_labels.append(data.y[data.test_mask].cpu().numpy())
                else:
                    true_labels.append(data.y.cpu().numpy())
            else:
                true_labels.append(data.y.cpu().numpy())

        true_labels = np.concatenate(true_labels)

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }