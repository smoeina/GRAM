# NXTGRAM: An Interpretable Approach for Graph Anomaly Detection using Gradient Attention Maps

A comprehensive framework for graph anomaly detection featuring multiple model variants, advanced interpretability, and state-of-the-art performance on graph datasets.

---

## 🔍 Overview

**GRAM** (GRAM: An Interpretable Approach for Graph Anomaly Detection using Gradient Attention Maps) is a unified framework that provides multiple implementations for graph anomaly detection, ranging from the original VAE-based approach to advanced variants with modern GNN architectures and interpretability features.

### Model Variants:

- ✅ **GRAM**: Original VAE-based model with GCN encoder and dual decoders
- ⚡ **GRAM v2**: Enhanced version with flexible GNN backbones (GATv2, SAGE, Transformer)
- 🚀 **GRAM v3**: Advanced variant with multi-scale encoding, contrastive learning, and adaptive weighting
- 🏃 **Fast_GRAM**: Lightweight variant optimized for speed with TransformerConv + Bilinear decoder
- 📈 **Temporal GRAM**: Extension for time-evolving graphs with GCN + LSTM + VAE
- 🧪 **Baseline Models**: OCGNN, DOMINANT, GAAN, GCNAE, CoNAD for comparison

---

## 📁 Project Structure

```
GRAM/
├── src/
│   ├── models/
│   │   ├── gram.py                    # Original GRAM implementation
│   │   ├── gram_v2.py                 # Enhanced GRAM with flexible GNNs
│   │   ├── gram_v3.py                 # Advanced GRAM with multi-scale & contrastive learning
│   │   ├── faster_gram.py             # Lightweight Fast_GRAM variant
│   │   ├── temporal_gram.py           # Temporal extension for dynamic graphs
│   │   ├── main.py                    # Main training/evaluation script
│   │   ├── base.py                    # Base classes and utilities
│   │   └── [baseline_models].py       # OCGNN, DOMINANT, GAAN, GCNAE, CoNAD
│   ├── utils/
│   │   ├── metrics.py                 # Comprehensive evaluation metrics
│   │   ├── util.py                    # Utility functions
│   │   └── flexible_dataset_adapter.py # Dataset loading utilities
│   └── examples/
│       ├── guide.py                   # Usage examples
│       ├── simple_ptc_gram_2.py       # PTC dataset analysis with classic ML comparison
│       ├── read files pt.py           # Data file reader utilities
│       └── read files.py              # General file reading utilities
├── tests/
│   ├── anomaly_detection_compare_v3.py # Comprehensive model comparison
│   ├── compare_faster_gram_v2.py      # GRAM vs Fast_GRAM benchmarking
│   ├── test_classification.py         # Classification evaluation
│   └── [other_test_files].py          # Additional test scripts
├── dataset/
│   └── PTC/                           # Sample dataset with train/test splits
├── train_model/                       # Saved model checkpoints
├── results/                           # Analysis results and visualizations
│   ├── anomaly_distribution_classicML_gram.png    # Anomaly score distribution
│   ├── confusion_matrices_classicML_gram.png      # Confusion matrices for top models
│   └── model_comparison_classicML_gram.png        # Comprehensive model comparison
├── notebooks/
│   └── Playground.ipynb               # Interactive experimentation
├── evaluation_results.png             # Performance visualization
├── training_and_data.png              # Training curves
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.0+

### Install Dependencies

```bash
pip install torch torch-geometric
pip install scikit-learn numpy pandas matplotlib
pip install scipy  # For statistical analysis
```

### Quick Setup

```bash
git clone https://github.com/your-username/GRAM.git
cd GRAM
pip install -r requirements.txt  # If available
```

---

## ⚙️ Usage

### 1. Basic Model Training

#### Original GRAM
```bash
python src/models/main.py --model GRAM --dataset PTC
```

#### GRAM v3 (Recommended)
```python
from src.models.gram_v3 import GNNVariantAnomalyDetector

model = GNNVariantAnomalyDetector(
    in_dim=features_dim,
    hidden_dim=128,
    latent_dim=64,
    num_layers=6,
    gnn_type='gatv2',  # or 'transformer', 'sage'
    decoder_type='bilinear',
    use_contrastive=True,
    use_adaptive_alpha=True,
    use_multi_scale=True
)

model.fit(train_loader)
scores = model.decision_function(test_data)
```

### 2. Comprehensive Model Comparison

```bash
python tests/anomaly_detection_compare_v3.py
```

This script evaluates all model variants across multiple datasets and provides:
- Performance metrics (AUC, AP, NDCG, Recall@K, Precision@K)
- Training time analysis
- Statistical significance testing
- Comprehensive results export

### 3. Fast_GRAM Benchmarking

```bash
python tests/compare_faster_gram_v2.py
```

### 4. Temporal Graph Analysis

```python
from src.models.temporal_gram import TemporalGRAM, TemporalGRAMTrainer

model = TemporalGRAM(
    in_dim=1,
    hid_dim=64,
    latent_dim=32,
    num_layers=4,
    dropout=0.1
)

trainer = TemporalGRAMTrainer(model, device='cuda')
trainer.train_epoch(training_data)
results = trainer.evaluate(training_data)
```

### 5. PTC Dataset Analysis with Classic ML Comparison

```bash
python src/examples/simple_ptc_gram_2.py
```

This comprehensive analysis script provides:

- **GRAM-based Anomaly Detection**: Uses pre-trained GRAM model for anomaly scoring
- **Classic ML Comparison**: Tests 10 different classifiers on anomaly features
- **Comprehensive Evaluation**: Cross-validation, confusion matrices, and performance metrics
- **Rich Visualizations**: Anomaly distributions, model comparisons, and confusion matrices

**Features:**
- Multiple anomaly scoring methods (distance-based, z-score, reconstruction error, degree anomaly)
- Enhanced feature extraction with anomaly scores
- 10 classic ML classifiers: Random Forest, Gradient Boosting, Extra Trees, AdaBoost, Neural Network, SVM, Logistic Regression, K-NN, Naive Bayes, Decision Tree
- 5-fold cross-validation with statistical analysis
- Comprehensive visualization suite

---

## 📊 Model Features

### GRAM v3 (Latest & Most Advanced)

**Architecture:**
- **Multi-scale Encoder**: Hierarchical pooling with SAGPooling + global mean pooling
- **Flexible GNN Backbone**: GATv2, TransformerConv, or SAGE
- **Advanced Decoders**: Bilinear or MLP edge reconstruction
- **VAE Latent Space**: Probabilistic encoding with KL divergence

**Advanced Features:**
- **Contrastive Learning**: InfoNCE-style loss for better representations
- **Adaptive Weighting**: Learnable balance between attribute and structure losses
- **Hard Negative Sampling**: Focus on challenging negative examples
- **Degree Prediction**: Auxiliary task for structural understanding
- **Interpretability**: Grad-CAM and attention weight visualization

**Configuration Options:**
```python
config = {
    'gnn_type': 'gatv2',           # 'gatv2', 'transformer', 'sage'
    'decoder_type': 'bilinear',    # 'bilinear', 'mlp', None
    'use_contrastive': True,       # Enable contrastive learning
    'use_adaptive_alpha': True,    # Learnable loss weighting
    'use_multi_scale': True,       # Multi-scale encoding
    'hidden_dim': 128,             # Hidden layer dimension
    'latent_dim': 64,              # Latent space dimension
    'num_layers': 6,               # Number of GNN layers
    'dropout': 0.1,                # Dropout rate
    'alpha': 0.5,                  # Attribute/structure balance
    'contrastive_weight': 0.1,     # Contrastive loss weight
    'degree_weight': 0.05          # Degree prediction weight
}
```

### Fast_GRAM (Speed Optimized)

**Optimizations:**
- **TransformerConv**: Modern attention-based convolution
- **Bilinear Decoder**: Efficient edge reconstruction
- **Optimized Negative Sampling**: Fast sampling without hard mining
- **Reduced Model Complexity**: Fewer parameters, faster training

### Temporal GRAM

**Features:**
- **Temporal Encoder**: GCN snapshots + LSTM aggregation
- **Time-aware Reconstruction**: Attribute and structure across time
- **GradCAM Interpretability**: Temporal anomaly explanation
- **Synthetic Anomaly Injection**: Robust evaluation framework

---

## 📈 Performance Results

### GRAM v3 Variants (MUTAG Dataset)

| Model | AUC | AP | NDCG | R@10 | P@10 | Time (s) |
|-------|-----|----|----- |------|------|----------|
| GRAM v3 Standard | 0.867 | 0.904 | 0.978 | 0.294 | 1.000 | 4.27 |
| GRAM v3 Transformer | 0.845 | 0.891 | 0.965 | 0.282 | 0.950 | 3.89 |
| GRAM v3 Lightweight | 0.823 | 0.876 | 0.952 | 0.271 | 0.900 | 2.15 |

### Model Comparison Across Datasets

| Dataset | Model | AUC | AP | NDCG | R@10 | P@10 |
|---------|-------|-----|----|----- |------|------|
| MUTAG | GRAM v3 | 0.867 | 0.904 | 0.978 | 0.294 | 1.000 |
| MUTAG | GRAM v2 | 0.845 | 0.891 | 0.965 | 0.282 | 0.950 |
| MUTAG | Original GRAM | 0.705 | 0.797 | 0.948 | 0.265 | 0.900 |
| PTC | GRAM v3 | 0.734 | 0.812 | 0.923 | 0.245 | 0.875 |

### Temporal GRAM Results

From synthetic temporal graph evaluation:
```json
{
  "AUC": 0.664,
  "Precision": 0.766,
  "Recall": 0.573,
  "F1": 0.655,
  "Accuracy": 0.699
}
```

### PTC Dataset Analysis Results

**Anomaly Distribution Analysis:**
![Anomaly Distribution](./results/anomaly_distribution_classicML_gram.png)

**Model Performance Comparison:**
![Model Comparison](./results/model_comparison_classicML_gram.png)

**Confusion Matrices for Top Models:**
![Confusion Matrices](./results/confusion_matrices_classicML_gram.png)

**Key Findings:**
- **Best Performing Models**: Random Forest and Gradient Boosting consistently achieve highest F1-scores
- **Anomaly Detection**: GRAM successfully identifies anomalous patterns in PTC molecular graphs
- **Feature Quality**: Enhanced features combining GRAM embeddings with anomaly scores provide strong discriminative power
- **Cross-Validation**: Robust performance across 5-fold CV with low variance

---

## 🧠 Key Innovations

### 1. Multi-Scale Graph Encoding
- **Hierarchical Pooling**: SAGPooling for local structure + global mean pooling
- **Node-Graph Fusion**: Combines node-level and graph-level representations
- **Adaptive Receptive Fields**: Captures both local and global patterns

### 2. Advanced Loss Functions
- **Adaptive Weighting**: Learnable balance between attribute and structure reconstruction
- **Contrastive Learning**: InfoNCE loss for better latent representations
- **Hard Negative Mining**: Focus on challenging examples for robust training

### 3. Interpretability & Explainability
- **Grad-CAM Integration**: Visualize important nodes for anomaly detection
- **Attention Weight Analysis**: Understand model focus across GNN layers
- **Multi-modal Explanations**: Combine gradient and attention-based insights

### 4. Flexible Architecture
- **Modular Design**: Easy to swap GNN backbones, decoders, and loss functions
- **Configuration-driven**: Extensive hyperparameter control
- **Extensible Framework**: Simple to add new model variants

### 5. Classic ML Integration
- **Hybrid Approach**: Combines deep learning anomaly detection with classic ML classification
- **Multiple Scoring Methods**: Distance-based, z-score, reconstruction error, and degree anomaly
- **Comprehensive Evaluation**: 10 different classifiers with cross-validation
- **Rich Visualizations**: Anomaly distributions, model comparisons, and confusion matrices

---

## 🔧 API Reference

### GRAM v3 Model

```python
class GNNVariantAnomalyDetector:
    def __init__(self, in_dim, hidden_dim=128, latent_dim=64, 
                 num_layers=4, gnn_type='gatv2', decoder_type='bilinear',
                 use_contrastive=True, use_adaptive_alpha=True, 
                 use_multi_scale=True, **kwargs):
        """
        Initialize GRAM v3 model
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
            num_layers: Number of GNN layers
            gnn_type: GNN backbone ('gatv2', 'transformer', 'sage')
            decoder_type: Edge decoder type ('bilinear', 'mlp', None)
            use_contrastive: Enable contrastive learning
            use_adaptive_alpha: Enable adaptive loss weighting
            use_multi_scale: Enable multi-scale encoding
        """
    
    def fit(self, train_loader):
        """Train the model on training data"""
    
    def decision_function(self, data, return_interpretability=False):
        """
        Compute anomaly scores
        
        Args:
            data: Graph data
            return_interpretability: Return interpretability scores
            
        Returns:
            anomaly_scores: Node or graph-level anomaly scores
            interpretability: Optional interpretability information
        """
    
    def explain_anomaly(self, data, node_idx=None):
        """
        Provide detailed anomaly explanations
        
        Returns:
            explanations: Dictionary with scores, Grad-CAM, attention weights
        """
```

### Evaluation Metrics

```python
from src.utils.metrics import (
    eval_roc_auc,           # ROC-AUC score
    eval_average_precision, # Average Precision
    eval_recall_at_k,       # Recall@K
    eval_precision_at_k,    # Precision@K
    eval_ndcg              # Normalized Discounted Cumulative Gain
)
```

### PTC Analysis Classes

```python
class GRAM_Detector:
    def __init__(self, model_path, device='cpu'):
        """Initialize GRAM detector with pre-trained model"""
    
    def load_model(self, in_dim, hid_dim=64, latent_size=32, num_layers=4, dropout=0.1, act='relu'):
        """Load pre-trained GRAM model"""
    
    def detect_anomalies(self, graphs, threshold_percentile=85):
        """
        Detect anomalies using multiple scoring methods
        
        Returns:
            X_anomaly: Anomaly features
            y_anomaly: Anomaly labels
            threshold: Detection threshold
            all_scores: All anomaly scores
        """

class AnomalyClassifier:
    def __init__(self):
        """Initialize 10 different classic ML classifiers"""
    
    def train(self, X_train, y_train):
        """Train all classifiers on anomaly features"""
    
    def predict(self, X_test):
        """Make predictions with all classifiers"""
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation for all classifiers"""
    
    def evaluate(self, y_true, predictions):
        """Evaluate all classifiers and return metrics"""

class ResultsVisualizer:
    def __init__(self, save_dir='./results'):
        """Initialize visualizer with save directory"""
    
    def plot_anomaly_distribution(self, all_scores, threshold):
        """Plot anomaly score distribution analysis"""
    
    def plot_model_comparison(self, results, cv_results=None):
        """Plot comprehensive model comparison charts"""
    
    def plot_confusion_matrices(self, y_true, predictions, top_n=4):
        """Plot confusion matrices for top performing models"""
```

---

## 🧪 Examples

### 1. Quick Start with GRAM v3

```python
import torch
from torch_geometric.loader import DataLoader
from src.models.gram_v3 import GNNVariantAnomalyDetector

# Load your dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

# Initialize model
model = GNNVariantAnomalyDetector(
    in_dim=node_features_dim,
    hidden_dim=128,
    latent_dim=64,
    gnn_type='gatv2',
    device='cuda'
)

# Train
model.fit(train_loader)

# Evaluate
for data in test_loader:
    scores = model.decision_function(data)
    print(f"Anomaly scores: {scores}")
```

### 2. Interpretability Analysis

```python
# Get detailed explanations
explanations = model.explain_anomaly(test_data)

print("Anomaly Scores:", explanations['anomaly_scores'])
print("Grad-CAM Importance:", explanations['grad_cam_importance'])
print("Attention Weights:", explanations['attention_weights'])
```

### 3. Custom Configuration

```python
# Custom GRAM v3 configuration
custom_model = GNNVariantAnomalyDetector(
    in_dim=features_dim,
    hidden_dim=256,           # Larger hidden dimension
    latent_dim=128,           # Larger latent space
    num_layers=8,             # Deeper network
    gnn_type='transformer',   # Use TransformerConv
    decoder_type='mlp',       # Use MLP decoder
    use_contrastive=True,     # Enable contrastive learning
    use_adaptive_alpha=True,  # Enable adaptive weighting
    use_multi_scale=False,    # Disable multi-scale (Transformer already captures this)
    dropout=0.2,              # Higher dropout
    lr=1e-4,                  # Lower learning rate
    epochs=500                # More training epochs
)
```

### 4. PTC Dataset Analysis with Classic ML

```python
from src.examples.simple_ptc_gram_2 import GRAM_Detector, AnomalyClassifier, PTCDataLoader

# Load PTC dataset
loader = PTCDataLoader("./data")
graphs = loader.load_data()

# Initialize GRAM detector with pre-trained model
detector = GRAM_Detector("./train_model/gram/PTC/model.pth", device='cuda')
detector.load_model(in_dim=graphs[0].x.shape[1], hid_dim=128, latent_size=64)

# Detect anomalies with multiple scoring methods
X_anomaly, y_anomaly, threshold, all_scores = detector.detect_anomalies(
    graphs, threshold_percentile=75
)

# Train multiple classic ML classifiers
classifier = AnomalyClassifier()
classifier.train(X_train, y_train)

# Evaluate and compare performance
predictions, probabilities = classifier.predict(X_test)
results = classifier.evaluate(y_test, predictions)

# Generate comprehensive visualizations
visualizer = ResultsVisualizer()
visualizer.plot_anomaly_distribution(all_scores, threshold)
visualizer.plot_model_comparison(results)
visualizer.plot_confusion_matrices(y_test, predictions)
```

---

## 📊 Evaluation & Benchmarking

### Comprehensive Evaluation Script

The `anomaly_detection_compare_v3.py` script provides:

1. **Multi-dataset Evaluation**: Tests on MUTAG, PTC, and other graph datasets
2. **Statistical Analysis**: T-tests for significance testing
3. **Performance Metrics**: AUC, AP, NDCG, Recall@K, Precision@K
4. **Timing Analysis**: Training and inference time comparison
5. **Results Export**: JSON and LaTeX table generation

### Running Full Evaluation

```bash
python tests/anomaly_detection_compare_v3.py
```

Output includes:
- Detailed performance tables
- Statistical significance tests
- Best model identification per metric
- Comprehensive results JSON file
- LaTeX table for papers

---

## 🔬 Research & Citation

If you use GRAM in your research, please cite:

```bibtex
@article{gram2024,
  title={GRAM: Graph Reconstruction-based Anomaly Model for Graph Anomaly Detection},
  author={[Your Name]},
  journal={[Journal/Conference]},
  year={2024}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** tests for new functionality
4. **Ensure** all tests pass
5. **Submit** a pull request

### Development Setup

```bash
git clone https://github.com/your-username/GRAM.git
cd GRAM
pip install -e .  # Install in development mode
pytest tests/     # Run tests
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PyTorch Geometric team for the excellent GNN framework
- Original GRAM paper authors for the foundational work
- Contributors and users who provided feedback and improvements

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/smoeina/GRAM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/smoeina/GRAM/discussions)
- **Documentation**: [Wiki](https://github.com/smoeina/GRAM/wiki)

---

*Last updated: 2024*