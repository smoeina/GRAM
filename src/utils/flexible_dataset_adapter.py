#!/usr/bin/env python3
"""
Flexible Dataset Adapter for Temporal GRAM

This module provides utilities to adapt various temporal graph datasets 
for use with the Temporal GRAM model. It supports multiple common formats
and provides easy configuration for new datasets.

Supported formats:
- Edge list with timestamps (like CollegeMsg)
- Adjacency matrices with timestamps
- NetworkX graphs with temporal information
- Custom formats via user-defined parsers

Author: Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, remove_self_loops, add_self_loops
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
import json
import os
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict


class TemporalDatasetAdapter(ABC):
    """Abstract base class for temporal dataset adapters"""
    
    def __init__(self, num_timesteps: int = 12, feature_type: str = 'degree'):
        self.num_timesteps = num_timesteps
        self.feature_type = feature_type
        self.node_mapping = {}
        self.reverse_node_mapping = {}
        self.num_nodes = 0
        
    @abstractmethod
    def load_data(self, data_path: str, **kwargs) -> List[Data]:
        """Load and preprocess temporal graph data"""
        pass
    
    def generate_node_features(self, edge_index: torch.Tensor, num_nodes: int, 
                             extra_info: Dict = None) -> torch.Tensor:
        """Generate node features based on specified type"""
        
        if self.feature_type == 'degree':
            # Node degree features
            degrees = torch.zeros(num_nodes, dtype=torch.float)
            for i in range(num_nodes):
                degrees[i] = (edge_index[0] == i).sum().float()
            return degrees.unsqueeze(1)
        
        elif self.feature_type == 'degree_centrality':
            # Degree centrality (normalized degree)
            degrees = torch.zeros(num_nodes, dtype=torch.float)
            for i in range(num_nodes):
                degrees[i] = (edge_index[0] == i).sum().float()
            max_degree = degrees.max()
            if max_degree > 0:
                degrees = degrees / max_degree
            return degrees.unsqueeze(1)
        
        elif self.feature_type == 'onehot':
            # One-hot encoding
            return torch.eye(num_nodes, dtype=torch.float)
        
        elif self.feature_type == 'random':
            # Random features
            return torch.randn(num_nodes, 10)
        
        elif self.feature_type == 'positional':
            # Simple positional encoding
            positions = torch.arange(num_nodes, dtype=torch.float).unsqueeze(1)
            return torch.sin(positions / 10.0)
        
        elif self.feature_type == 'combined':
            # Combine degree and positional
            degrees = torch.zeros(num_nodes, dtype=torch.float)
            for i in range(num_nodes):
                degrees[i] = (edge_index[0] == i).sum().float()
            positions = torch.arange(num_nodes, dtype=torch.float)
            return torch.stack([degrees, positions], dim=1)
        
        elif self.feature_type == 'constant':
            # Constant features
            return torch.ones(num_nodes, 1, dtype=torch.float)
        
        elif self.feature_type == 'custom' and extra_info and 'features' in extra_info:
            # Custom features provided by user
            return extra_info['features']
        
        else:
            warnings.warn(f"Unknown feature type: {self.feature_type}. Using constant features.")
            return torch.ones(num_nodes, 1, dtype=torch.float)
    
    def create_node_mapping(self, all_nodes: List):
        """Create mapping from original node IDs to consecutive integers"""
        unique_nodes = sorted(list(set(all_nodes)))
        self.node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
        self.reverse_node_mapping = {idx: node for node, idx in self.node_mapping.items()}
        self.num_nodes = len(unique_nodes)
        return self.node_mapping


class EdgeListAdapter(TemporalDatasetAdapter):
    """Adapter for edge list format with timestamps (like CollegeMsg)"""
    
    def load_data(self, data_path: str, **kwargs) -> List[Data]:
        """
        Load edge list data with timestamps
        
        Expected format: each line contains "node1 node2 timestamp"
        """
        # Parse parameters
        delimiter = kwargs.get('delimiter', None)  # Auto-detect if None
        has_header = kwargs.get('has_header', False)
        timestamp_col = kwargs.get('timestamp_col', 2)
        node1_col = kwargs.get('node1_col', 0)
        node2_col = kwargs.get('node2_col', 1)
        time_window_strategy = kwargs.get('time_window_strategy', 'equal_intervals')
        
        print(f"Loading edge list from: {data_path}")
        
        # Load data
        try:
            # Try different delimiters if not specified
            if delimiter is None:
                for delim in [' ', '\t', ',', ';']:
                    try:
                        df = pd.read_csv(data_path, delimiter=delim, header=0 if has_header else None)
                        if df.shape[1] >= 3:
                            delimiter = delim
                            break
                    except:
                        continue
                if delimiter is None:
                    raise ValueError("Could not automatically detect delimiter")
            else:
                df = pd.read_csv(data_path, delimiter=delimiter, header=0 if has_header else None)
            
            # Rename columns for consistency
            df.columns = ['node1', 'node2', 'timestamp'] + list(df.columns[3:])
            
        except Exception as e:
            print(f"Error loading CSV, trying manual parsing: {e}")
            # Manual parsing as fallback
            edges = []
            with open(data_path, 'r') as f:
                for line_no, line in enumerate(f):
                    if line_no == 0 and has_header:
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            node1 = int(parts[node1_col])
                            node2 = int(parts[node2_col])
                            timestamp = int(parts[timestamp_col])
                            edges.append((node1, node2, timestamp))
                        except ValueError:
                            # Skip lines that can't be parsed
                            continue
            
            df = pd.DataFrame(edges, columns=['node1', 'node2', 'timestamp'])
        
        if df.empty:
            raise ValueError("No valid data loaded")
        
        print(f"Loaded {len(df)} edges")
        
        # Create node mapping
        all_nodes = df['node1'].tolist() + df['node2'].tolist()
        self.create_node_mapping(all_nodes)
        
        # Map nodes to consecutive integers
        df['node1_mapped'] = df['node1'].map(self.node_mapping)
        df['node2_mapped'] = df['node2'].map(self.node_mapping)
        
        # Create temporal windows
        if time_window_strategy == 'equal_intervals':
            # Divide time into equal intervals
            min_time, max_time = df['timestamp'].min(), df['timestamp'].max()
            time_bins = np.linspace(min_time, max_time, self.num_timesteps + 1)
            
        elif time_window_strategy == 'equal_edges':
            # Divide edges into equal-sized groups
            sorted_times = df['timestamp'].sort_values()
            edges_per_window = len(df) // self.num_timesteps
            time_bins = [df['timestamp'].min()]
            for i in range(1, self.num_timesteps):
                idx = min(i * edges_per_window, len(sorted_times) - 1)
                time_bins.append(sorted_times.iloc[idx])
            time_bins.append(df['timestamp'].max() + 1)
            
        else:
            raise ValueError(f"Unknown time window strategy: {time_window_strategy}")
        
        # Create graphs for each time window
        temporal_graphs = []
        
        for t in range(self.num_timesteps):
            start_time = time_bins[t]
            end_time = time_bins[t + 1]
            
            # Get edges in this time window
            mask = (df['timestamp'] >= start_time) & (df['timestamp'] < end_time)
            window_df = df[mask]
            
            if len(window_df) > 0:
                # Create edge list
                edges = []
                for _, row in window_df.iterrows():
                    src = int(row['node1_mapped'])
                    dst = int(row['node2_mapped'])
                    edges.append([src, dst])
                    edges.append([dst, src])  # Make undirected
                
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_index = remove_self_loops(edge_index)[0]
                edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
            else:
                # No edges in this window - create self-loops only
                edge_index = torch.stack([
                    torch.arange(self.num_nodes), 
                    torch.arange(self.num_nodes)
                ], dim=0)
            
            # Generate node features
            node_features = self.generate_node_features(edge_index, self.num_nodes)
            
            # Create graph
            graph = Data(x=node_features, edge_index=edge_index)
            temporal_graphs.append(graph)
            
            print(f"  Time window {t+1}: {edge_index.size(1)} edges, "
                  f"time range [{start_time}, {end_time})")
        
        return temporal_graphs


class AdjacencyMatrixAdapter(TemporalDatasetAdapter):
    """Adapter for adjacency matrix format with timestamps"""
    
    def load_data(self, data_path: str, **kwargs) -> List[Data]:
        """
        Load adjacency matrices with timestamps
        
        Expected format: Directory with files named like "adj_t0.csv", "adj_t1.csv", etc.
        Or a single file with multiple matrices
        """
        if os.path.isdir(data_path):
            return self._load_from_directory(data_path, **kwargs)
        else:
            return self._load_from_file(data_path, **kwargs)
    
    def _load_from_directory(self, data_dir: str, **kwargs) -> List[Data]:
        """Load from directory of adjacency matrix files"""
        file_pattern = kwargs.get('file_pattern', 'adj_t{}.csv')
        
        temporal_graphs = []
        
        for t in range(self.num_timesteps):
            filename = file_pattern.format(t)
            filepath = os.path.join(data_dir, filename)
            
            if os.path.exists(filepath):
                # Load adjacency matrix
                adj_matrix = pd.read_csv(filepath, header=None).values
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)
                
                # Convert to edge index
                edge_index = adj_matrix.nonzero().t().contiguous()
                
                # Generate node features
                num_nodes = adj_matrix.size(0)
                if t == 0:  # Set num_nodes on first iteration
                    self.num_nodes = num_nodes
                    self.create_node_mapping(list(range(num_nodes)))
                
                node_features = self.generate_node_features(edge_index, num_nodes)
                
                # Create graph
                graph = Data(x=node_features, edge_index=edge_index)
                temporal_graphs.append(graph)
            else:
                print(f"Warning: File {filepath} not found, creating empty graph")
                # Create empty graph
                edge_index = torch.empty((2, 0), dtype=torch.long)
                node_features = self.generate_node_features(edge_index, self.num_nodes)
                graph = Data(x=node_features, edge_index=edge_index)
                temporal_graphs.append(graph)
        
        return temporal_graphs
    
    def _load_from_file(self, data_path: str, **kwargs) -> List[Data]:
        """Load from single file containing multiple matrices"""
        # This would need to be implemented based on specific file format
        raise NotImplementedError("Single file adjacency matrix loading not yet implemented")


class NetworkXAdapter(TemporalDatasetAdapter):
    """Adapter for NetworkX temporal graphs"""
    
    def load_data(self, data_path: str, **kwargs) -> List[Data]:
        """
        Load NetworkX graphs with temporal information
        
        Expected: Pickle file containing list of NetworkX graphs
        """
        import pickle
        
        with open(data_path, 'rb') as f:
            nx_graphs = pickle.load(f)
        
        if not isinstance(nx_graphs, list):
            raise ValueError("Expected list of NetworkX graphs")
        
        # Take only the specified number of timesteps
        nx_graphs = nx_graphs[:self.num_timesteps]
        
        temporal_graphs = []
        
        for t, G in enumerate(nx_graphs):
            # Convert NetworkX to PyTorch Geometric
            if len(G.nodes()) == 0:
                # Empty graph
                edge_index = torch.empty((2, 0), dtype=torch.long)
                num_nodes = self.num_nodes if hasattr(self, 'num_nodes') and self.num_nodes > 0 else 10
                node_features = self.generate_node_features(edge_index, num_nodes)
                graph = Data(x=node_features, edge_index=edge_index)
            else:
                # Convert to PyG format
                graph = from_networkx(G)
                
                # Generate node features if not present
                if not hasattr(graph, 'x') or graph.x is None:
                    num_nodes = graph.num_nodes
                    if t == 0:  # Set num_nodes on first iteration
                        self.num_nodes = num_nodes
                        self.create_node_mapping(list(G.nodes()))
                    
                    node_features = self.generate_node_features(graph.edge_index, num_nodes)
                    graph.x = node_features
            
            temporal_graphs.append(graph)
        
        return temporal_graphs


class CustomFormatAdapter(TemporalDatasetAdapter):
    """Adapter for custom formats using user-defined parsers"""
    
    def __init__(self, num_timesteps: int = 12, feature_type: str = 'degree', 
                 custom_parser: Callable = None):
        super().__init__(num_timesteps, feature_type)
        self.custom_parser = custom_parser
    
    def load_data(self, data_path: str, **kwargs) -> List[Data]:
        """Load data using custom parser"""
        if self.custom_parser is None:
            raise ValueError("Custom parser function must be provided")
        
        return self.custom_parser(data_path, self, **kwargs)


class TemporalDatasetFactory:
    """Factory class for creating appropriate dataset adapters"""
    
    ADAPTERS = {
        'edgelist': EdgeListAdapter,
        'adjacency': AdjacencyMatrixAdapter,
        'networkx': NetworkXAdapter,
        'custom': CustomFormatAdapter
    }
    
    @classmethod
    def create_adapter(cls, format_type: str, **kwargs) -> TemporalDatasetAdapter:
        """Create appropriate adapter based on format type"""
        if format_type not in cls.ADAPTERS:
            raise ValueError(f"Unknown format type: {format_type}. "
                           f"Available: {list(cls.ADAPTERS.keys())}")
        
        return cls.ADAPTERS[format_type](**kwargs)
    
    @classmethod
    def auto_detect_format(cls, data_path: str) -> str:
        """Attempt to auto-detect data format"""
        if os.path.isdir(data_path):
            return 'adjacency'
        
        extension = os.path.splitext(data_path)[1].lower()
        
        if extension == '.pkl' or extension == '.pickle':
            return 'networkx'
        elif extension in ['.txt', '.csv', '.tsv']:
            return 'edgelist'
        else:
            return 'edgelist'  # Default fallback


def load_temporal_dataset(data_path: str, format_type: str = 'auto', **kwargs) -> Tuple[List[Data], Dict]:
    """
    Convenient function to load temporal graph datasets
    
    Args:
        data_path: Path to dataset
        format_type: Type of format ('edgelist', 'adjacency', 'networkx', 'custom', 'auto')
        **kwargs: Additional parameters for the adapter
    
    Returns:
        temporal_graphs: List of PyTorch Geometric Data objects
        metadata: Dictionary with dataset information
    """
    
    # Auto-detect format if requested
    if format_type == 'auto':
        format_type = TemporalDatasetFactory.auto_detect_format(data_path)
        print(f"Auto-detected format: {format_type}")
    
    # Create adapter
    adapter = TemporalDatasetFactory.create_adapter(format_type, **kwargs)
    
    # Load data
    temporal_graphs = adapter.load_data(data_path, **kwargs)
    
    # Create metadata
    metadata = {
        'num_timesteps': len(temporal_graphs),
        'num_nodes': adapter.num_nodes,
        'node_mapping': adapter.node_mapping,
        'reverse_node_mapping': adapter.reverse_node_mapping,
        'feature_type': adapter.feature_type,
        'format_type': format_type
    }
    
    # Add graph statistics
    edge_counts = []
    node_features_dim = None
    
    for graph in temporal_graphs:
        edge_counts.append(graph.edge_index.size(1))
        if node_features_dim is None:
            node_features_dim = graph.x.size(1)
    
    metadata.update({
        'edge_counts': edge_counts,
        'node_features_dim': node_features_dim,
        'total_edges': sum(edge_counts),
        'avg_edges_per_timestep': np.mean(edge_counts)
    })
    
    return temporal_graphs, metadata


# Example custom parser for a specific format
def example_custom_parser(data_path: str, adapter: TemporalDatasetAdapter, **kwargs) -> List[Data]:
    """
    Example custom parser for a specific data format
    
    This is a template that users can modify for their specific needs
    """
    temporal_graphs = []
    
    # Example: Load your custom format here
    # with open(data_path, 'r') as f:
    #     your_custom_loading_logic()
    
    # For demonstration, create random graphs
    num_nodes = kwargs.get('num_nodes', 50)
    adapter.num_nodes = num_nodes
    adapter.create_node_mapping(list(range(num_nodes)))
    
    for t in range(adapter.num_timesteps):
        # Create random edges
        num_edges = np.random.randint(10, 100)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Generate features
        node_features = adapter.generate_node_features(edge_index, num_nodes)
        
        # Create graph
        graph = Data(x=node_features, edge_index=edge_index)
        temporal_graphs.append(graph)
    
    return temporal_graphs


# Configuration templates for different datasets
DATASET_CONFIGS = {
    'collegemsg': {
        'format_type': 'edgelist',
        'num_timesteps': 12,
        'feature_type': 'degree',
        'delimiter': ' ',
        'has_header': False,
        'time_window_strategy': 'equal_intervals'
    },
    
    'email_temporal': {
        'format_type': 'edgelist',
        'num_timesteps': 10,
        'feature_type': 'degree_centrality',
        'delimiter': '\t',
        'has_header': True,
        'time_window_strategy': 'equal_edges'
    },
    
    'social_network': {
        'format_type': 'adjacency',
        'num_timesteps': 15,
        'feature_type': 'combined',
        'file_pattern': 'network_t{}.csv'
    },
    
    'custom_dataset': {
        'format_type': 'custom',
        'num_timesteps': 8,
        'feature_type': 'random',
        'custom_parser': example_custom_parser
    }
}


def get_dataset_config(dataset_name: str) -> Dict:
    """Get configuration for a known dataset"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(DATASET_CONFIGS.keys())}")
    
    return DATASET_CONFIGS[dataset_name].copy()


# Usage examples
if __name__ == "__main__":
    print("Temporal Dataset Adapter Examples")
    print("=" * 40)
    
    # Example 1: Load CollegeMsg dataset
    print("\n1. Loading CollegeMsg-style dataset:")
    try:
        config = get_dataset_config('collegemsg')
        temporal_graphs, metadata = load_temporal_dataset(
            'CollegeMsg.txt', 
            **config
        )
        print(f"Loaded {len(temporal_graphs)} timesteps with {metadata['num_nodes']} nodes")
    except FileNotFoundError:
        print("CollegeMsg.txt not found, skipping this example")
    
    # Example 2: Auto-detect format
    print("\n2. Auto-detecting format:")
    # This would work with any supported file
    
    # Example 3: Custom adapter
    print("\n3. Using custom adapter:")
    adapter = TemporalDatasetFactory.create_adapter(
        'custom',
        num_timesteps=5,
        feature_type='degree',
        custom_parser=example_custom_parser
    )
    
    temporal_graphs = adapter.load_data('dummy_path', num_nodes=30)
    print(f"Created {len(temporal_graphs)} synthetic temporal graphs")
    
    # Example 4: Different feature types
    print("\n4. Different feature types available:")
    feature_types = ['degree', 'degree_centrality', 'onehot', 'random', 'positional', 'combined', 'constant']
    for ft in feature_types:
        adapter = EdgeListAdapter(num_timesteps=3, feature_type=ft)
        # Demo with small synthetic data
        dummy_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        features = adapter.generate_node_features(dummy_edge_index, 3)
        print(f"  {ft}: shape {features.shape}")
    
    print("\nAdapter ready for use with Temporal GRAM!")
