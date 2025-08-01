#!/usr/bin/env python3
"""
CollegeMsg Dataset Example for Temporal GRAM

This script demonstrates how to use the Temporal GRAM model with the CollegeMsg dataset.
It includes data loading, preprocessing, training, and anomaly detection evaluation.

Usage:
    python collegemsg_example.py --data_path CollegeMsg.txt --epochs 100 --device cuda
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
from temporal_gram import (
    TemporalGRAM, 
    TemporalDataProcessor, 
    TemporalGRAMTrainer,
    create_synthetic_anomalies
)
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')


class CollegeMsgExperiment:
    """Complete experiment pipeline for CollegeMsg dataset"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # Initialize data processor
        self.processor = TemporalDataProcessor(
            num_timesteps=config['num_timesteps'],
            feature_type=config['feature_type']
        )
        
        print(f"Initialized experiment with device: {self.device}")
        print(f"Configuration: {json.dumps(config, indent=2)}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess CollegeMsg data"""
        print("\n" + "="*50)
        print("LOADING AND PREPROCESSING DATA")
        print("="*50)
        
        start_time = time.time()
        
        # Load data
        print(f"Loading data from: {self.config['data_path']}")
        temporal_graphs, nodes, node_map = self.processor.load_collegemsg_data(
            self.config['data_path']
        )
        
        # Store data info
        self.temporal_graphs = temporal_graphs
        self.nodes = nodes
        self.node_map = node_map
        self.num_nodes = len(nodes)
        
        load_time = time.time() - start_time
        
        # Print data statistics
        print(f"\nData Statistics:")
        print(f"  Total nodes: {self.num_nodes}")
        print(f"  Time steps: {len(temporal_graphs)}")
        print(f"  Loading time: {load_time:.2f} seconds")
        
        # Analyze temporal graphs
        edge_counts = []
        node_degrees = []
        
        for t, graph in enumerate(temporal_graphs):
            num_edges = graph.edge_index.size(1)
            edge_counts.append(int(num_edges))  # Convert to int
            
            # Calculate average degree
            degrees = torch.zeros(graph.x.size(0))
            for i in range(graph.x.size(0)):
                degrees[i] = (graph.edge_index[0] == i).sum()
            avg_degree = degrees.mean().item()
            node_degrees.append(float(avg_degree))  # Convert to float
            
            print(f"  Timestep {t+1}: {num_edges} edges, avg degree: {avg_degree:.2f}")
        
        # Store results with proper type conversion
        self.results['data_stats'] = {
            'num_nodes': int(self.num_nodes),
            'num_timesteps': int(len(temporal_graphs)),
            'edge_counts': edge_counts,  # Already converted to int above
            'avg_degrees': node_degrees,  # Already converted to float above
            'loading_time': float(load_time),
            'total_edges': int(sum(edge_counts)),
            'max_edges_per_timestep': int(max(edge_counts)),
            'min_edges_per_timestep': int(min(edge_counts))
        }
        
        return temporal_graphs, nodes, node_map
    
    def create_train_test_split(self):
        """Create training and test data with synthetic anomalies"""
        print("\n" + "="*50)
        print("CREATING TRAIN/TEST SPLIT")
        print("="*50)
        
        # For temporal data, we use the single temporal sequence for training
        # and create synthetic anomalies for testing
        
        # Use original data for training (normal samples)
        self.train_data = [self.temporal_graphs]
        
        # Create test data with synthetic anomalies
        print("Creating synthetic anomalies for testing...")
        
        # Create multiple copies with different levels of anomalies
        test_sequences = []
        test_labels = []
        
        # Normal test samples
        num_normal_test = self.config['num_test_normal']
        for i in range(num_normal_test):
            # Add small random variations to create normal test samples
            normal_seq = []
            for graph in self.temporal_graphs:
                new_graph = graph.clone()
                # Add small noise
                noise = torch.randn_like(new_graph.x) * 0.1
                new_graph.x = new_graph.x + noise
                normal_seq.append(new_graph)
            
            test_sequences.append(normal_seq)
            test_labels.extend([0] * self.num_nodes)
        
        # Anomalous test samples
        num_anomalous_test = self.config['num_test_anomalous']
        for i in range(num_anomalous_test):
            anomalous_seq = []
            for t, graph in enumerate(self.temporal_graphs):
                new_graph = graph.clone()
                
                if t == len(self.temporal_graphs) - 1:  # Only modify final timestep
                    # Create structural anomalies by removing/adding edges randomly
                    edge_index = new_graph.edge_index.clone()
                    num_edges = edge_index.size(1)
                    
                    # Remove some edges
                    remove_mask = torch.rand(num_edges) > 0.8
                    edge_index = edge_index[:, ~remove_mask]
                    
                    # Add some random edges
                    num_new_edges = int(num_edges * 0.1)
                    new_edges = torch.randint(0, self.num_nodes, (2, num_new_edges))
                    edge_index = torch.cat([edge_index, new_edges], dim=1)
                    
                    new_graph.edge_index = edge_index
                    
                    # Also add feature noise
                    noise = torch.randn_like(new_graph.x) * 0.5
                    new_graph.x = new_graph.x + noise
                
                anomalous_seq.append(new_graph)
            
            test_sequences.append(anomalous_seq)
            test_labels.extend([1] * self.num_nodes)
        
        self.test_data = test_sequences
        self.test_labels = np.array(test_labels)
        
        print(f"Created training data: {len(self.train_data)} sequences")
        print(f"Created test data: {len(self.test_data)} sequences")
        print(f"Test labels: {len(test_labels)} total, {sum(test_labels)} anomalous")
        
        return self.train_data, self.test_data, self.test_labels
    
    def initialize_model(self):
        """Initialize the Temporal GRAM model"""
        print("\n" + "="*50)
        print("INITIALIZING MODEL")
        print("="*50)
        
        # Determine input dimension based on feature type
        if self.config['feature_type'] == 'onehot':
            in_dim = self.num_nodes
        elif self.config['feature_type'] == 'random':
            in_dim = 10
        else:  # degree or constant
            in_dim = 1
        
        self.model = TemporalGRAM(
            in_dim=in_dim,
            hid_dim=self.config['hid_dim'],
            latent_size=self.config['latent_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            lstm_hidden_dim=self.config.get('lstm_hidden_dim', None)
        )
        
        self.trainer = TemporalGRAMTrainer(
            model=self.model,
            device=self.device,
            lr=self.config['learning_rate'],
            alpha=self.config['alpha']
        )
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model initialized:")
        print(f"  Input dimension: {in_dim}")
        print(f"  Hidden dimension: {self.config['hid_dim']}")
        print(f"  Latent dimension: {self.config['latent_size']}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return self.model, self.trainer
    
    def train_model(self):
        """Train the Temporal GRAM model"""
        print("\n" + "="*50)
        print("TRAINING MODEL")
        print("="*50)
        
        train_losses = []
        val_losses = []
        
        best_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # Training
            train_loss = self.trainer.train_epoch(self.train_data)
            train_losses.append(float(train_loss))  # Convert to float
            
            # Validation (use training data for now)
            val_results = self.trainer.evaluate(self.train_data)
            val_loss = val_results['loss']
            val_losses.append(float(val_loss))  # Convert to float
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_temporal_gram.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:3d}/{self.config['epochs']} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        
        # Load best model
        self.model.load_state_dict(torch.load('best_temporal_gram.pth'))
        
        # Store results with proper type conversion
        self.results['training'] = {
            'train_losses': train_losses,  # Already converted to float above
            'val_losses': val_losses,  # Already converted to float above
            'best_loss': float(best_loss),
            'total_epochs': int(epoch + 1),
            'training_time': float(total_time),
            'early_stopping_patience': int(self.config['patience']),
            'final_train_loss': float(train_losses[-1]) if train_losses else 0.0,
            'final_val_loss': float(val_losses[-1]) if val_losses else 0.0
        }
        
        print(f"\nTraining completed:")
        print(f"  Total epochs: {epoch + 1}")
        print(f"  Best validation loss: {best_loss:.4f}")
        print(f"  Training time: {total_time:.2f} seconds")
        
        return train_losses, val_losses
    
    def evaluate_model(self):
        """Evaluate the model on test data"""
        print("\n" + "="*50)
        print("EVALUATING MODEL")
        print("="*50)
        
        start_time = time.time()
        
        # Evaluate on test data
        test_results = self.trainer.evaluate(self.test_data, self.test_labels)
        
        eval_time = time.time() - start_time
        
        # Calculate metrics
        scores = np.array(test_results['scores'])
        labels = self.test_labels
        
        # Calculate AUC
        try:
            auc = roc_auc_score(labels, scores)
            fpr, tpr, thresholds = roc_curve(labels, scores)
        except ValueError as e:
            print(f"Warning: Could not calculate AUC: {e}")
            auc = 0.5
            fpr, tpr, thresholds = [0, 1], [0, 1], [0, 1]
        
        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate precision, recall, F1 at optimal threshold
        predictions = (scores > optimal_threshold).astype(int)
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(labels)
        
        # Store results with explicit type conversion to avoid JSON serialization issues
        self.results['evaluation'] = {
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'optimal_threshold': float(optimal_threshold),
            'evaluation_time': float(eval_time),
            'scores': scores.tolist(),  # Convert to list for JSON serialization
            'labels': labels.tolist(),  # Convert to list for JSON serialization
            'confusion_matrix': {
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn)
            }
        }
        
        print(f"Evaluation Results:")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  Evaluation time: {eval_time:.2f} seconds")
        
        return test_results
    
    def visualize_results(self):
        """Create visualizations of the results"""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Create output directory
        os.makedirs('results', exist_ok=True)
        
        try:
            # Set matplotlib backend to avoid display issues
            plt.switch_backend('Agg')
            
            # 1. Training curves
            if 'training' in self.results:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                train_losses = self.results['training']['train_losses']
                val_losses = self.results['training']['val_losses']
                
                plt.plot(train_losses, label='Training Loss', color='blue')
                plt.plot(val_losses, label='Validation Loss', color='red')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Progress')
                plt.legend()
                plt.grid(True)
                
                # 2. Data statistics
                plt.subplot(1, 2, 2)
                edge_counts = self.results['data_stats']['edge_counts']
                timesteps = range(1, len(edge_counts) + 1)
                plt.plot(timesteps, edge_counts, 'b-o', label='Edge Count')
                plt.xlabel('Timestep')
                plt.ylabel('Number of Edges')
                plt.title('Temporal Graph Evolution')
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig('results/training_and_data.png', dpi=300, bbox_inches='tight')
                plt.close()  # Close to free memory
                print("✓ Saved training_and_data.png")
            
            # 3. ROC Curve and Score Distribution
            if 'evaluation' in self.results:
                plt.figure(figsize=(15, 5))
                
                scores = np.array(self.results['evaluation']['scores'])
                labels = np.array(self.results['evaluation']['labels'])
                
                # ROC Curve
                plt.subplot(1, 3, 1)
                try:
                    fpr, tpr, _ = roc_curve(labels, scores)
                    auc = self.results['evaluation']['auc']
                    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', color='blue')
                    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                except Exception as e:
                    plt.text(0.5, 0.5, f'ROC Curve error:\n{str(e)[:50]}...', 
                            ha='center', va='center', fontsize=10)
                    plt.title('ROC Curve (Error)')
                
                # Score distribution
                plt.subplot(1, 3, 2)
                try:
                    normal_scores = scores[labels == 0]
                    anomaly_scores = scores[labels == 1]
                    
                    if len(normal_scores) > 0:
                        plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
                    if len(anomaly_scores) > 0:
                        plt.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red', density=True)
                    
                    plt.xlabel('Anomaly Score')
                    plt.ylabel('Density')
                    plt.title('Score Distribution')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                except Exception as e:
                    plt.text(0.5, 0.5, f'Distribution error:\n{str(e)[:50]}...', 
                            ha='center', va='center', fontsize=10)
                    plt.title('Score Distribution (Error)')
                
                # Metrics bar chart
                plt.subplot(1, 3, 3)
                try:
                    metrics = ['AUC', 'Precision', 'Recall', 'F1', 'Accuracy']
                    values = [
                        self.results['evaluation']['auc'],
                        self.results['evaluation']['precision'],
                        self.results['evaluation']['recall'],
                        self.results['evaluation']['f1'],
                        self.results['evaluation']['accuracy']
                    ]
                    
                    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
                    bars = plt.bar(metrics, values, color=colors)
                    plt.ylabel('Score')
                    plt.title('Evaluation Metrics')
                    plt.ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                    
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                except Exception as e:
                    plt.text(0.5, 0.5, f'Metrics error:\n{str(e)[:50]}...', 
                            ha='center', va='center', fontsize=10)
                    plt.title('Evaluation Metrics (Error)')
                
                plt.tight_layout()
                plt.savefig('results/evaluation_results.png', dpi=300, bbox_inches='tight')
                plt.close()  # Close to free memory
                print("✓ Saved evaluation_results.png")
            
            print("✓ All visualizations saved to 'results/' directory")
            
        except Exception as e:
            print(f"⚠ Warning: Error creating visualizations: {e}")
            print("  Continuing without visualizations...")
            
            # Create a simple text summary instead
            try:
                with open('results/results_summary.txt', 'w') as f:
                    f.write("Temporal GRAM Experiment Results\n")
                    f.write("=" * 40 + "\n\n")
                    
                    if 'data_stats' in self.results:
                        f.write("Data Statistics:\n")
                        f.write(f"  Nodes: {self.results['data_stats']['num_nodes']}\n")
                        f.write(f"  Timesteps: {self.results['data_stats']['num_timesteps']}\n")
                        f.write(f"  Total edges: {self.results['data_stats'].get('total_edges', 'N/A')}\n\n")
                    
                    if 'training' in self.results:
                        f.write("Training Results:\n")
                        f.write(f"  Epochs: {self.results['training']['total_epochs']}\n")
                        f.write(f"  Best loss: {self.results['training']['best_loss']:.4f}\n")
                        f.write(f"  Training time: {self.results['training']['training_time']:.2f}s\n\n")
                    
                    if 'evaluation' in self.results:
                        f.write("Evaluation Results:\n")
                        f.write(f"  AUC-ROC: {self.results['evaluation']['auc']:.4f}\n")
                        f.write(f"  Precision: {self.results['evaluation']['precision']:.4f}\n")
                        f.write(f"  Recall: {self.results['evaluation']['recall']:.4f}\n")
                        f.write(f"  F1-Score: {self.results['evaluation']['f1']:.4f}\n")
                        f.write(f"  Accuracy: {self.results['evaluation']['accuracy']:.4f}\n")
                
                print("✓ Saved text summary to results_summary.txt")
            except Exception as e2:
                print(f"⚠ Could not save text summary: {e2}")
                
        finally:
            # Ensure we don't leave any plots open
            plt.close('all')
    
    def save_results(self):
        """Save all results to file"""
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)
        
        # Convert results to JSON-serializable format
        def convert_to_serializable(obj):
            """Recursively convert numpy/torch types to Python types"""
            import numpy as np
            
            if isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, 'item'):  # For numpy scalars
                return obj.item()
            elif torch.is_tensor(obj):
                return obj.detach().cpu().numpy().tolist()
            else:
                return obj
        
        # Save configuration and results
        serializable_results = convert_to_serializable(self.results)
        serializable_config = convert_to_serializable(self.config)
        
        output = {
            'config': serializable_config,
            'results': serializable_results,
            'model_info': {
                'total_parameters': int(sum(p.numel() for p in self.model.parameters())),
                'model_architecture': str(self.model)
            }
        }
        
        try:
            with open('results/experiment_results.json', 'w') as f:
                json.dump(output, f, indent=2)
            print("✓ Successfully saved experiment_results.json")
        except Exception as e:
            print(f"✗ Error saving JSON: {e}")
            # Save a simplified version without problematic data
            simplified_output = {
                'config': serializable_config,
                'summary': {
                    'auc': float(self.results['evaluation']['auc']) if 'evaluation' in self.results else 0.0,
                    'best_loss': float(self.results['training']['best_loss']) if 'training' in self.results else 0.0,
                    'total_epochs': int(self.results['training']['total_epochs']) if 'training' in self.results else 0,
                    'num_nodes': int(self.results['data_stats']['num_nodes']) if 'data_stats' in self.results else 0
                },
                'model_info': output['model_info']
            }
            
            with open('results/experiment_summary.json', 'w') as f:
                json.dump(simplified_output, f, indent=2)
            print("✓ Saved simplified summary to experiment_summary.json")
        
        # Save model
        try:
            # Clean results for model saving (remove large arrays)
            clean_results = convert_to_serializable(self.results)
            if 'evaluation' in clean_results and 'scores' in clean_results['evaluation']:
                # Keep only summary statistics, not full score arrays
                scores = clean_results['evaluation']['scores']
                clean_results['evaluation']['scores_summary'] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'length': len(scores)
                }
                del clean_results['evaluation']['scores']
                del clean_results['evaluation']['labels']
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': serializable_config,
                'results_summary': clean_results
            }, 'results/temporal_gram_model.pth')
            print("✓ Successfully saved temporal_gram_model.pth")
        except Exception as e:
            print(f"✗ Error saving model: {e}")
            # Save model only
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': serializable_config
            }, 'results/temporal_gram_model_only.pth')
            print("✓ Saved model weights only to temporal_gram_model_only.pth")
        
        print("\nResults saved successfully!")
        print("Files in 'results/' directory:")
        print("  - experiment_results.json: Complete experiment data")
        print("  - temporal_gram_model.pth: Trained model with metadata")
        print("  - *.png: Visualization plots")
    
    def run_full_experiment(self):
        """Run the complete experiment pipeline"""
        print("Starting Temporal GRAM Experiment on CollegeMsg Dataset")
        print("="*60)
        
        # Load data
        self.load_and_preprocess_data()
        
        # Create train/test split
        self.create_train_test_split()
        
        # Initialize model
        self.initialize_model()
        
        # Train model
        self.train_model()
        
        # Evaluate model
        self.evaluate_model()
        
        # Create visualizations
        self.visualize_results()
        
        # Save results
        self.save_results()
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final AUC-ROC: {self.results['evaluation']['auc']:.4f}")
        print(f"Best Validation Loss: {self.results['training']['best_loss']:.4f}")
        print("Check 'results/' directory for detailed outputs")


def main():
    parser = argparse.ArgumentParser(description='Temporal GRAM on CollegeMsg Dataset')
    parser.add_argument('--data_path', type=str, default='CollegeMsg.txt',
                       help='Path to CollegeMsg.txt file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num_timesteps', type=int, default=12,
                       help='Number of temporal windows')
    parser.add_argument('--feature_type', type=str, default='degree',
                       choices=['degree', 'onehot', 'random', 'constant'],
                       help='Type of node features to generate')
    parser.add_argument('--hid_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--latent_size', type=int, default=64,
                       help='Latent space dimension')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Balance between attribute and structure loss')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_path': args.data_path,
        'epochs': args.epochs,
        'device': 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device,
        'num_timesteps': args.num_timesteps,
        'feature_type': args.feature_type,
        'hid_dim': args.hid_dim,
        'latent_size': args.latent_size,
        'num_layers': 6,
        'dropout': 0.1,
        'learning_rate': args.learning_rate,
        'alpha': args.alpha,
        'patience': 20,
        'num_test_normal': 5,
        'num_test_anomalous': 5
    }
    
    # Run experiment
    experiment = CollegeMsgExperiment(config)
    experiment.run_full_experiment()


if __name__ == "__main__":
    main()
