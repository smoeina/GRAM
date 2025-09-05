import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCN, global_add_pool
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class GRAM_Base(nn.Module):
    def __init__(self, in_dim, hid_dim, latent_size, num_layers, dropout, act):
        super(GRAM_Base, self).__init__()
        
        encoder_layers = int(num_layers / 2)
        self.shared_encoder = GCN(in_channels=in_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=encoder_layers,
                                  out_channels=hid_dim,
                                  dropout=dropout,
                                  act=act)
        
        self.encode_liner1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, latent_size)
        )
        
        self.encode_liner2 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(), 
            nn.Linear(hid_dim, latent_size)
        )
        
    def forward(self, x, edge_index, batch=None):
        h = self.shared_encoder(x, edge_index)
        
        mu = self.encode_liner1(h)
        logstd = self.encode_liner2(h).clamp(max=10)
        
        z = mu + torch.randn_like(logstd) * torch.exp(logstd)
        
        if batch is not None:
            z_global = global_add_pool(z, batch)
        else:
            z_global = torch.sum(z, dim=0, keepdim=True)
            
        return z_global, h

class GRAM_Detector:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = None
        self.model_path = model_path
        
    def load_model(self, in_dim, hid_dim=64, latent_size=32, num_layers=4, dropout=0.1, act='relu'):
        self.model = GRAM_Base(in_dim, hid_dim, latent_size, num_layers, dropout, act).to(self.device)
        
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"Loaded GRAM model from {self.model_path}")
            except:
                print(f"Failed to load model, using random weights")
        else:
            print(f"Model file not found, using random weights")
            
        self.model.eval()
        
    def detect_anomalies(self, graphs, threshold_percentile=85):
        all_features = []
        all_scores = []
        all_labels = []
        all_graph_features = []
        
        with torch.no_grad():
            for data in graphs:
                data = data.to(self.device)
                z_global, h = self.model(data.x, data.edge_index)
                
                # Multiple anomaly scoring methods
                mean_h = torch.mean(h, dim=0, keepdim=True)
                std_h = torch.std(h, dim=0, keepdim=True) + 1e-8
                
                # 1. Distance-based anomaly score
                distances = torch.norm(h - mean_h, dim=1)
                
                # 2. Z-score based anomaly score
                z_scores = torch.norm((h - mean_h) / std_h, dim=1)
                
                # 3. Reconstruction error simulation
                recon_error = torch.sum((h - mean_h) ** 2, dim=1)
                
                # 4. Node degree anomaly
                if data.edge_index.shape[1] > 0:
                    degrees = torch.bincount(data.edge_index[0], minlength=data.x.shape[0]).float()
                    degree_anomaly = torch.abs(degrees - torch.mean(degrees))
                else:
                    degree_anomaly = torch.zeros(data.x.shape[0])
                
                # Combined anomaly score
                scores = (0.3 * distances + 0.3 * z_scores + 0.3 * recon_error + 0.1 * degree_anomaly).cpu().numpy()
                
                # Enhanced features
                enhanced_features = torch.cat([
                    h,
                    distances.unsqueeze(1),
                    z_scores.unsqueeze(1),
                    recon_error.unsqueeze(1),
                    degree_anomaly.unsqueeze(1)
                ], dim=1).cpu().numpy()
                
                all_features.append(enhanced_features)
                all_scores.append(scores)
                all_labels.append(data.y.cpu().numpy())
                all_graph_features.append(z_global.cpu().numpy())
        
        # Determine threshold
        flat_scores = np.concatenate(all_scores)
        threshold = np.percentile(flat_scores, threshold_percentile)
        
        # Identify anomalies
        anomalous_data = []
        anomalous_labels = []
        
        for i, scores in enumerate(all_scores):
            anomaly_mask = scores > threshold
            if np.any(anomaly_mask):
                anomalous_data.append(all_features[i][anomaly_mask])
                anomalous_labels.append(all_labels[i][anomaly_mask])
        
        if anomalous_data:
            X_anomaly = np.vstack(anomalous_data)
            y_anomaly = np.concatenate(anomalous_labels)
            return X_anomaly, y_anomaly, threshold, all_scores
        else:
            return None, None, threshold, all_scores

class PTCDataLoader:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        
    def load_data(self):
        # Read files
        edges = self._read_edges()
        graph_indicator = self._read_graph_indicator()
        graph_labels = self._read_graph_labels()
        node_labels = self._read_node_labels()
        
        # Create graphs
        graphs = self._create_graphs(edges, graph_indicator, graph_labels, node_labels)
        return graphs
    
    def _read_edges(self):
        edge_file = os.path.join(self.data_dir, "PTC_MR_A.txt")
        edges = []
        
        if os.path.exists(edge_file):
            with open(edge_file, 'r') as f:
                for line in f:
                    line = line.strip().replace(',', ' ')
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            src, dst = int(parts[0]) - 1, int(parts[1]) - 1
                            edges.append([src, dst])
                        except:
                            continue
        return edges
    
    def _read_graph_indicator(self):
        file_path = os.path.join(self.data_dir, "PTC_MR_graph_indicator.txt")
        with open(file_path, 'r') as f:
            return [int(line.strip()) - 1 for line in f if line.strip()]
    
    def _read_graph_labels(self):
        file_path = os.path.join(self.data_dir, "PTC_MR_graph_labels.txt")
        with open(file_path, 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]
    
    def _read_node_labels(self):
        file_path = os.path.join(self.data_dir, "PTC_MR_node_labels.txt")
        with open(file_path, 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]
    
    def _create_graphs(self, edges, graph_indicator, graph_labels, node_labels):
        graphs = []
        unique_graphs = sorted(set(graph_indicator))
        
        for graph_id in unique_graphs:
            # Get nodes for this graph
            node_indices = [i for i, g in enumerate(graph_indicator) if g == graph_id]
            node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
            
            # Get edges for this graph
            graph_edges = []
            for src, dst in edges:
                if src in node_mapping and dst in node_mapping:
                    graph_edges.append([node_mapping[src], node_mapping[dst]])
            
            # Create edge index
            if graph_edges:
                edge_index = torch.tensor(graph_edges, dtype=torch.long).t()
            else:
                edge_index = torch.tensor([[], []], dtype=torch.long)
            
            # Create node features (one-hot encoding of node labels)
            num_nodes = len(node_indices)
            unique_node_labels = sorted(set(node_labels))
            num_features = len(unique_node_labels)
            
            x = torch.zeros(num_nodes, num_features + 5)  # +5 for additional features
            for i, node_idx in enumerate(node_indices):
                label = node_labels[node_idx]
                if label in unique_node_labels:
                    x[i, unique_node_labels.index(label)] = 1.0
                # Add random features
                x[i, num_features:] = torch.randn(5) * 0.1
            
            # Node labels for this graph
            y = torch.tensor([node_labels[i] for i in node_indices], dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, y=y, 
                       graph_label=torch.tensor([graph_labels[graph_id]]))
            graphs.append(data)
        
        return graphs

class AnomalyClassifier:
    def __init__(self):
        self.classifiers = {}
        self.scaler = StandardScaler()
        self._initialize_classifiers()
        
    def _initialize_classifiers(self):
        """Initialize multiple classifiers for comparison"""
        self.classifiers = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=50, learning_rate=1.0, random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42,
                early_stopping=True, validation_fraction=0.1
            ),
            'SVM': SVC(
                kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42, solver='lbfgs'
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5, weights='distance'
            ),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10, random_state=42, min_samples_split=5
            )
        }
        
    def train(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        
        print(f"Training {len(self.classifiers)} classifiers...")
        print(f"Training data shape: {X_scaled.shape}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        for name, clf in self.classifiers.items():
            try:
                clf.fit(X_scaled, y_train)
                print(f"✓ Trained {name}")
            except Exception as e:
                print(f"✗ Failed to train {name}: {e}")
                del self.classifiers[name]
    
    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        predictions = {}
        probabilities = {}
        
        for name, clf in self.classifiers.items():
            try:
                predictions[name] = clf.predict(X_scaled)
                if hasattr(clf, 'predict_proba'):
                    probabilities[name] = clf.predict_proba(X_scaled)
            except Exception as e:
                print(f"Failed to predict with {name}: {e}")
        
        return predictions, probabilities
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation for all classifiers"""
        X_scaled = self.scaler.fit_transform(X)
        cv_results = {}
        
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        for name, clf in self.classifiers.items():
            try:
                scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1_weighted')
                cv_results[name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
                print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
            except Exception as e:
                print(f"CV failed for {name}: {e}")
        
        return cv_results
    
    def evaluate(self, y_true, predictions):
        results = {}
        
        for name, y_pred in predictions.items():
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            
            results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1
            }
        
        return results

class ResultsVisualizer:
    def __init__(self, save_dir='./results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        plt.style.use('default')
        
    def plot_anomaly_distribution(self, all_scores, threshold):
        """Plot anomaly score distribution"""
        plt.figure(figsize=(15, 5))
        
        # Combined histogram
        plt.subplot(1, 3, 1)
        flat_scores = np.concatenate(all_scores)
        plt.hist(flat_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot by graph
        plt.subplot(1, 3, 2)
        if len(all_scores) > 1:
            plt.boxplot([scores for scores in all_scores if len(scores) > 0])
            plt.axhline(threshold, color='red', linestyle='--', linewidth=2)
            plt.xlabel('Graph Index')
            plt.ylabel('Anomaly Score')
            plt.title('Anomaly Scores by Graph')
            plt.grid(True, alpha=0.3)
        
        # Anomaly count per graph
        plt.subplot(1, 3, 3)
        anomaly_counts = [np.sum(scores > threshold) for scores in all_scores]
        plt.bar(range(len(anomaly_counts)), anomaly_counts, alpha=0.7, color='orange')
        plt.xlabel('Graph Index')
        plt.ylabel('Number of Anomalies')
        plt.title('Anomalies per Graph')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'anomaly_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results, cv_results=None):
        """Plot comprehensive model comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Prepare data
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # 1. Bar plot of all metrics
        x = np.arange(len(models))
        width = 0.2
        
        ax = axes[0, 0]
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            ax.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. F1-Score ranking
        ax = axes[0, 1]
        f1_scores = [results[model]['f1_score'] for model in models]
        sorted_indices = np.argsort(f1_scores)[::-1]
        sorted_models = [models[i] for i in sorted_indices]
        sorted_f1 = [f1_scores[i] for i in sorted_indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_models)))
        bars = ax.barh(sorted_models, sorted_f1, color=colors, alpha=0.8)
        ax.set_xlabel('F1-Score')
        ax.set_title('Models Ranked by F1-Score')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, sorted_f1):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', ha='left', va='center')
        
        # 3. Accuracy vs F1-Score scatter
        ax = axes[0, 2]
        accuracies = [results[model]['accuracy'] for model in models]
        ax.scatter(accuracies, f1_scores, s=100, alpha=0.7, c=range(len(models)), cmap='tab10')
        
        for i, model in enumerate(models):
            ax.annotate(model, (accuracies[i], f1_scores[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('F1-Score')
        ax.set_title('Accuracy vs F1-Score')
        ax.grid(True, alpha=0.3)
        
        # 4. Cross-validation results (if available)
        if cv_results:
            ax = axes[1, 0]
            cv_models = list(cv_results.keys())
            cv_means = [cv_results[model]['mean'] for model in cv_models]
            cv_stds = [cv_results[model]['std'] for model in cv_models]
            
            ax.bar(cv_models, cv_means, yerr=cv_stds, capsize=5, alpha=0.8, color='lightblue')
            ax.set_ylabel('CV F1-Score')
            ax.set_title('Cross-Validation Results')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 5. Metric heatmap
        ax = axes[1, 1]
        metric_matrix = np.array([[results[model][metric] for metric in metrics] for model in models])
        
        im = ax.imshow(metric_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_title('Performance Heatmap')
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(metrics)):
                ax.text(j, i, f'{metric_matrix[i, j]:.3f}', 
                       ha='center', va='center', color='black', fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        
        # 6. Performance summary table
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')
        
        # Create table data
        table_data = []
        for model in models:
            row = [model] + [f"{results[model][metric]:.3f}" for metric in metrics]
            table_data.append(row)
        
        table = ax.table(cellText=table_data,
                        colLabels=['Model'] + [m.capitalize() for m in metrics],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 2)
        ax.set_title('Performance Summary Table')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, y_true, predictions, top_n=4):
        """Plot confusion matrices for top performing models"""
        # Get top performing models
        results = {}
        for name, y_pred in predictions.items():
            f1 = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[2]
            results[name] = f1
        
        top_models = sorted(results.keys(), key=lambda k: results[k], reverse=True)[:top_n]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, model in enumerate(top_models):
            if i >= 4:
                break
                
            cm = confusion_matrix(y_true, predictions[model])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       cbar_kws={'label': 'Count'})
            axes[i].set_title(f'{model}\nF1: {results[model]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results_table(self, results, cv_results=None):
        """Save results to CSV file"""
        df = pd.DataFrame(results).T
        df = df.round(4)
        
        if cv_results:
            cv_df = pd.DataFrame({k: [v['mean'], v['std']] for k, v in cv_results.items()}, 
                                index=['CV_Mean', 'CV_Std']).T
            df = df.join(cv_df)
        
        df.to_csv(os.path.join(self.save_dir, 'results_table.csv'))
        print(f"Results saved to {os.path.join(self.save_dir, 'results_table.csv')}")
        
        return df


def main():
    print("PTC GRAM Anomaly Classification - Optimized")
    print("=" * 50)

    # Load data
    print("Loading PTC data...")
    loader = PTCDataLoader("./data")
    graphs = loader.load_data()
    print(f"Loaded {len(graphs)} graphs")

    # Initialize GRAM detector
    print("Initializing GRAM detector...")
    detector = GRAM_Detector("./train_model/gram/PTC/model.pth",
                             device='cuda' if torch.cuda.is_available() else 'cpu')
    detector.load_model(in_dim=graphs[0].x.shape[1], hid_dim=128, latent_size=64)

    # Initialize visualizer
    visualizer = ResultsVisualizer()

    # Detect anomalies
    print("Detecting anomalies...")
    X_anomaly, y_anomaly, threshold, all_scores = detector.detect_anomalies(graphs, threshold_percentile=75)

    print(f"Anomaly threshold: {threshold:.3f}")

    # Visualize anomaly distribution
    visualizer.plot_anomaly_distribution(all_scores, threshold)

    if X_anomaly is not None and len(X_anomaly) > 20:
        print(f"Found {len(X_anomaly)} anomalies")
        print(f"Feature dimension: {X_anomaly.shape[1]}")
        print(f"Class distribution: {np.bincount(y_anomaly)}")

        # حذف کلاس‌هایی با تعداد نمونه کمتر از 2
        min_samples = 2
        unique_classes, counts = np.unique(y_anomaly, return_counts=True)
        valid_classes = unique_classes[counts >= min_samples]
        mask = np.isin(y_anomaly, valid_classes)
        X_anomaly = X_anomaly[mask]
        y_anomaly = y_anomaly[mask]
        print(f"Filtered class distribution: {np.bincount(y_anomaly)}")

        # بررسی اینکه آیا بعد از فیلتر کردن، داده کافی برای تقسیم وجود دارد
        if len(np.unique(y_anomaly)) < 2:
            print(
                "Error: After filtering, less than 2 classes remain. Try lowering min_samples or adjusting threshold_percentile.")
            return None, None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_anomaly, y_anomaly, test_size=0.3, random_state=42, stratify=y_anomaly
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Initialize and train classifiers
        classifier = AnomalyClassifier()

        # Cross-validation
        print("\n" + "=" * 50)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 50)
        cv_results = classifier.cross_validate(X_anomaly, y_anomaly, cv=5)

        # Train on full training set
        print("\n" + "=" * 50)
        print("TRAINING CLASSIFIERS")
        print("=" * 50)
        classifier.train(X_train, y_train)

        # Make predictions
        print("\n" + "=" * 50)
        print("MAKING PREDICTIONS")
        print("=" * 50)
        predictions, probabilities = classifier.predict(X_test)

        # Evaluate
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        results = classifier.evaluate(y_test, predictions)

        # Print detailed results
        print(f"\n{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 70)
        for name, metrics in results.items():
            print(f"{name:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
                  f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f}")

        # Find best model
        best_model = max(results.keys(), key=lambda k: results[k]['f1_score'])
        best_f1 = results[best_model]['f1_score']
        print(f"\nBest Model: {best_model} (F1-Score: {best_f1:.3f})")

        # Visualize results
        print("\n" + "=" * 50)
        print("GENERATING VISUALIZATIONS")
        print("=" * 50)

        # Model comparison charts
        visualizer.plot_model_comparison(results, cv_results)

        # Confusion matrices for top models
        visualizer.plot_confusion_matrices(y_test, predictions)

        # Save results table
        results_df = visualizer.save_results_table(results, cv_results)

        # Print final summary
        print("\n" + "=" * 50)
        print("FINAL SUMMARY")
        print("=" * 50)
        print(f"Dataset: PTC ({len(graphs)} graphs)")
        print(f"Anomalies detected: {len(X_anomaly)}")
        print(f"Feature dimensions: {X_anomaly.shape[1]}")
        print(f"Models tested: {len(results)}")
        print(f"Best performing model: {best_model}")
        print(f"Best F1-Score: {best_f1:.3f}")

        # Top 3 models
        top_3 = sorted(results.keys(), key=lambda k: results[k]['f1_score'], reverse=True)[:3]
        print(f"\nTop 3 Models:")
        for i, model in enumerate(top_3, 1):
            f1 = results[model]['f1_score']
            acc = results[model]['accuracy']
            print(f"  {i}. {model}: F1={f1:.3f}, Acc={acc:.3f}")

        print(f"\nResults and visualizations saved to './results/' directory")

        return results, cv_results

    else:
        n_found = len(X_anomaly) if X_anomaly is not None else 0
        print(f"Insufficient anomalies found ({n_found})")
        print("Try lowering threshold_percentile or adjusting model parameters")
        return None, None
if __name__ == "__main__":
    out = main()
    # If main() returns None (e.g., no anomalies found), print a helpful hint.
    if out is None:
        print("Try lowering threshold_percentile")
    else:
        results, cv_results = out