import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from gram_v2_classification import GNNClassificationModel

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class MultiDatasetExperiment:
    def __init__(self):
        self.results = {}
        self.datasets_info = {}

    def load_dataset(self, dataset_name):
        """Load and explore a TU dataset"""
        print(f'\n{"=" * 60}')
        print(f'LOADING {dataset_name.upper()} DATASET')
        print(f'{"=" * 60}')

        # Load dataset
        dataset = TUDataset(root='./dataset', name=dataset_name)
        print(f'Loaded {len(dataset)} graphs.')

        # Dataset statistics
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of node features: {dataset.num_node_features}')
        print(f'Number of classes: {dataset.num_classes}')

        # Graph size statistics
        num_nodes = [data.x.size(0) for data in dataset]
        num_edges = [data.edge_index.size(1) for data in dataset]

        print(f'\nGraph size statistics:')
        print(f'  Nodes - Mean: {np.mean(num_nodes):.1f}, Std: {np.std(num_nodes):.1f}, '
              f'Min: {np.min(num_nodes)}, Max: {np.max(num_nodes)}')
        print(f'  Edges - Mean: {np.mean(num_edges):.1f}, Std: {np.std(num_edges):.1f}, '
              f'Min: {np.min(num_edges)}, Max: {np.max(num_edges)}')

        # Class distribution
        labels = [data.y.item() for data in dataset]
        unique, counts = np.unique(labels, return_counts=True)
        print(f'\nClass distribution:')
        for label, count in zip(unique, counts):
            print(f'  Class {label}: {count} graphs ({count / len(dataset) * 100:.1f}%)')

        # Store dataset info
        self.datasets_info[dataset_name] = {
            'num_graphs': len(dataset),
            'num_features': dataset.num_node_features,
            'num_classes': dataset.num_classes,
            'avg_nodes': np.mean(num_nodes),
            'avg_edges': np.mean(num_edges),
            'class_distribution': dict(zip(unique, counts))
        }

        return dataset

    def create_data_loaders(self, dataset, dataset_name, batch_size=32, test_size=0.2, val_size=0.1):
        """Create train/validation/test data loaders"""
        print(f'\nCreating data loaders for {dataset_name}...')

        # Adjust batch size for large datasets
        if len(dataset) > 1000:
            batch_size = min(batch_size * 2, 128)
        elif len(dataset) < 200:
            batch_size = max(batch_size // 2, 16)

        # Get indices for splitting
        indices = list(range(len(dataset)))

        # First split: train+val vs test
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=42,
            stratify=[dataset[i].y.item() for i in indices]
        )

        # Second split: train vs val
        if val_size > 0:
            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=val_size / (1 - test_size), random_state=42,
                stratify=[dataset[i].y.item() for i in train_val_indices]
            )
        else:
            train_indices = train_val_indices
            val_indices = []

        # Create subsets
        train_dataset = [dataset[i] for i in train_indices]
        val_dataset = [dataset[i] for i in val_indices] if val_indices else None
        test_dataset = [dataset[i] for i in test_indices]

        print(f'Dataset splits:')
        print(f'  Train: {len(train_dataset)} graphs')
        print(f'  Validation: {len(val_dataset) if val_dataset else 0} graphs')
        print(f'  Test: {len(test_dataset)} graphs')
        print(f'  Batch size: {batch_size}')

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def get_dataset_config(self, dataset_name, num_features, num_classes):
        """Get optimal configuration for each dataset"""
        configs = {
            'MUTAG': {
                'hidden_dim': 64,
                'num_layers': 3,
                'dropout': 0.1,
                'lr': 0.01,
                'weight_decay': 5e-4,
                'epochs': 150,
                'pooling': 'mean'
            },
            'COLLAB': {
                'hidden_dim': 128,
                'num_layers': 4,
                'dropout': 0.2,
                'lr': 0.001,
                'weight_decay': 1e-4,
                'epochs': 100,
                'pooling': 'mean'
            }
        }

        # Default config for unknown datasets
        default_config = {
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'lr': 0.01,
            'weight_decay': 5e-4,
            'epochs': 100,
            'pooling': 'mean'
        }

        return configs.get(dataset_name, default_config)

    def train_and_evaluate_model(self, train_loader, val_loader, test_loader,
                                 dataset, dataset_name, config, gnn_type='gatv2'):
        """Train and evaluate the GNN model"""
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        # Initialize model
        model = GNNClassificationModel(
            in_dim=dataset.num_node_features,
            hidden_dim=config['hidden_dim'],
            num_classes=dataset.num_classes,
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            gnn_type=gnn_type,
            device=device,
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            epochs=config['epochs'],
            pooling=config['pooling'],
            task_type='graph'
        )

        # Train model
        start_time = time.time()
        model.fit(train_loader, val_loader, verbose=False)
        training_time = time.time() - start_time

        # Evaluate on test set
        test_metrics = model.get_detailed_metrics(test_loader)
        test_metrics['training_time'] = training_time

        return model, test_metrics

    def compare_gnn_variants(self, train_loader, val_loader, test_loader,
                             dataset, dataset_name):
        """Compare different GNN variants on a dataset"""
        print(f'\n{"=" * 60}')
        print(f'COMPARING GNN VARIANTS ON {dataset_name.upper()}')
        print(f'{"=" * 60}')

        gnn_types = ['gatv2', 'sage', 'transformer']
        results = {}
        config = self.get_dataset_config(dataset_name, dataset.num_node_features, dataset.num_classes)

        for gnn_type in gnn_types:
            print(f'\nTraining {gnn_type.upper()} model...')

            try:
                model, metrics = self.train_and_evaluate_model(
                    train_loader, val_loader, test_loader, dataset, dataset_name, config, gnn_type
                )
                results[gnn_type] = metrics
                print(f'  Accuracy: {metrics["accuracy"]:.4f}, '
                      f'F1-Score: {metrics["f1_score"]:.4f}, '
                      f'Time: {metrics["training_time"]:.1f}s')

            except Exception as e:
                print(f'  Failed: {str(e)}')
                results[gnn_type] = {'error': str(e)}

        # Store results
        self.results[dataset_name] = results

        # Print comparison
        print(f'\n{dataset_name} Results:')
        print(f'{"Model":<12} {"Accuracy":<10} {"Precision":<11} {"Recall":<8} {"F1-Score":<8} {"Time(s)":<8}')
        print(f'{"-" * 70}')

        for gnn_type, metrics in results.items():
            if 'error' not in metrics:
                print(f'{gnn_type.upper():<12} {metrics["accuracy"]:<10.4f} {metrics["precision"]:<11.4f} '
                      f'{metrics["recall"]:<8.4f} {metrics["f1_score"]:<8.4f} {metrics["training_time"]:<8.1f}')
            else:
                print(f'{gnn_type.upper():<12} {"FAILED":<50}')

        return results

    def run_dataset_experiment(self, dataset_name):
        """Run complete experiment on a single dataset"""
        try:
            # Load dataset
            dataset = self.load_dataset(dataset_name)

            # Create data loaders
            train_loader, val_loader, test_loader = self.create_data_loaders(dataset, dataset_name)

            # Compare GNN variants
            results = self.compare_gnn_variants(train_loader, val_loader, test_loader, dataset, dataset_name)

            return True

        except Exception as e:
            print(f'Error with {dataset_name}: {str(e)}')
            self.results[dataset_name] = {'error': str(e)}
            return False

    def generate_comprehensive_report(self):
        """Generate comprehensive report across all datasets"""
        print(f'\n{"=" * 80}')
        print(f'COMPREHENSIVE RESULTS ACROSS ALL DATASETS')
        print(f'{"=" * 80}')

        # Dataset statistics table
        print(f'\nDATASET STATISTICS:')
        print(f'{"Dataset":<10} {"Graphs":<8} {"Features":<10} {"Classes":<8} {"Avg Nodes":<10} {"Avg Edges":<10}')
        print(f'{"-" * 70}')

        for dataset_name, info in self.datasets_info.items():
            print(f'{dataset_name:<10} {info["num_graphs"]:<8} {info["num_features"]:<10} '
                  f'{info["num_classes"]:<8} {info["avg_nodes"]:<10.1f} {info["avg_edges"]:<10.1f}')

        # Performance comparison table
        print(f'\nPERFORMANCE COMPARISON:')

        # Create comprehensive results table
        all_results = []
        for dataset_name, dataset_results in self.results.items():
            if 'error' not in dataset_results:
                for gnn_type, metrics in dataset_results.items():
                    if 'error' not in metrics:
                        all_results.append({
                            'Dataset': dataset_name,
                            'Model': gnn_type.upper(),
                            'Accuracy': metrics['accuracy'],
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'F1-Score': metrics['f1_score'],
                            'Training Time': metrics['training_time']
                        })

        if all_results:
            df = pd.DataFrame(all_results)

            # Print detailed table
            print(
                f'{"Dataset":<10} {"Model":<12} {"Accuracy":<10} {"Precision":<11} {"Recall":<8} {"F1-Score":<9} {"Time(s)":<8}')
            print(f'{"-" * 80}')

            for _, row in df.iterrows():
                print(f'{row["Dataset"]:<10} {row["Model"]:<12} {row["Accuracy"]:<10.4f} '
                      f'{row["Precision"]:<11.4f} {row["Recall"]:<8.4f} {row["F1-Score"]:<9.4f} '
                      f'{row["Training Time"]:<8.1f}')

            # Best model per dataset
            print(f'\nBEST MODEL PER DATASET:')
            print(f'{"Dataset":<10} {"Best Model":<12} {"Accuracy":<10} {"F1-Score":<9}')
            print(f'{"-" * 50}')

            for dataset in df['Dataset'].unique():
                dataset_df = df[df['Dataset'] == dataset]
                best_row = dataset_df.loc[dataset_df['Accuracy'].idxmax()]
                print(f'{best_row["Dataset"]:<10} {best_row["Model"]:<12} '
                      f'{best_row["Accuracy"]:<10.4f} {best_row["F1-Score"]:<9.4f}')

            # Overall statistics
            print(f'\nOVERALL STATISTICS:')
            print(f'  Average Accuracy: {df["Accuracy"].mean():.4f} (±{df["Accuracy"].std():.4f})')
            print(f'  Average F1-Score: {df["F1-Score"].mean():.4f} (±{df["F1-Score"].std():.4f})')
            print(f'  Average Training Time: {df["Training Time"].mean():.1f}s (±{df["Training Time"].std():.1f}s)')

            # Model comparison across datasets
            print(f'\nMODEL COMPARISON ACROSS DATASETS:')
            model_stats = df.groupby('Model').agg({
                'Accuracy': ['mean', 'std'],
                'F1-Score': ['mean', 'std'],
                'Training Time': ['mean', 'std']
            }).round(4)

            print(f'{"Model":<12} {"Avg Acc":<10} {"Std Acc":<10} {"Avg F1":<10} {"Std F1":<10} {"Avg Time":<10}')
            print(f'{"-" * 70}')
            for model in model_stats.index:
                print(f'{model:<12} {model_stats.loc[model, ("Accuracy", "mean")]:<10.4f} '
                      f'{model_stats.loc[model, ("Accuracy", "std")]:<10.4f} '
                      f'{model_stats.loc[model, ("F1-Score", "mean")]:<10.4f} '
                      f'{model_stats.loc[model, ("F1-Score", "std")]:<10.4f} '
                      f'{model_stats.loc[model, ("Training Time", "mean")]:<10.1f}')

            # Create visualization
            self.create_visualization(df)

        else:
            print("No successful results to report.")

    def create_visualization(self, df):
        """Create visualization of results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Accuracy comparison
        sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Model', ax=ax1)
        ax1.set_title('Accuracy Comparison Across Datasets')
        ax1.set_ylim(0, 1)

        # 2. F1-Score comparison
        sns.barplot(data=df, x='Dataset', y='F1-Score', hue='Model', ax=ax2)
        ax2.set_title('F1-Score Comparison Across Datasets')
        ax2.set_ylim(0, 1)

        # 3. Training time comparison
        sns.barplot(data=df, x='Dataset', y='Training Time', hue='Model', ax=ax3)
        ax3.set_title('Training Time Comparison')
        ax3.set_ylabel('Time (seconds)')

        # 4. Model performance heatmap
        pivot_df = df.pivot_table(values='Accuracy', index='Dataset', columns='Model')
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Accuracy Heatmap')

        plt.tight_layout()
        plt.show()


def main():
    """Main execution function"""
    print("Multi-Dataset Graph Classification with GNN")
    print("=" * 80)

    # Initialize experiment
    experiment = MultiDatasetExperiment()

    # List of datasets to test
    datasets_to_test = ['MUTAG', 'COLLAB']

    print(f"Testing datasets: {datasets_to_test}")

    # Run experiments on each dataset
    successful_datasets = []
    for dataset_name in datasets_to_test:
        success = experiment.run_dataset_experiment(dataset_name)
        if success:
            successful_datasets.append(dataset_name)

    # Generate comprehensive report
    if successful_datasets:
        experiment.generate_comprehensive_report()
    else:
        print("No datasets were successfully processed.")

    print(f'\n{"=" * 80}')
    print(f'EXPERIMENT COMPLETED')
    print(f'{"=" * 80}')
    print(f'Successfully processed datasets: {successful_datasets}')
    print(
        f'Total experiments run: {sum(len(results) for results in experiment.results.values() if "error" not in results)}')


if __name__ == "__main__":
    main()