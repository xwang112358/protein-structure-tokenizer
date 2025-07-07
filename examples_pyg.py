#!/usr/bin/env python3
"""
Example usage of the PyTorch Geometric Data Pipeline

This script demonstrates how to use the PyG data pipeline for different scenarios:
1. Processing a single PDB file
2. Converting existing BatchDataVQ3D to PyG format
3. Processing multiple proteins in batch
4. Using the data with PyTorch Geometric models
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyg_data_pipeline import PyGDataPipeline, convert_batch_data_to_pyg
from data_pipeline import DataPipeline


class SimpleProteinGNN(torch.nn.Module):
    """
    A simple GNN model for demonstration purposes.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        return x


def example_single_protein():
    """Example: Process a single PDB file to PyG format."""
    print("=== Example 1: Single Protein Processing ===")
    
    # Initialize pipeline
    config = {
        "num_neighbor": 20,
        "crop_index": 256,
        "include_global_features": True,
    }
    pipeline = PyGDataPipeline(config)
    
    # Check if example PDB exists
    pdb_path = "casp14_pdbs/T1024.pdb"
    if not os.path.exists(pdb_path):
        print(f"PDB file {pdb_path} not found. Skipping this example.")
        return None
    
    # Process the protein
    try:
        pyg_data = pipeline.process_single(
            input_source=pdb_path,
            output_path="example_output_T1024.pt"
        )
        
        print(f"Successfully processed {pdb_path}")
        print(f"Number of nodes: {pyg_data.num_nodes}")
        print(f"Number of edges: {pyg_data.num_edges}")
        print(f"Node features shape: {pyg_data.x.shape}")
        print(f"Edge features shape: {pyg_data.edge_attr.shape}")
        print(f"Chain ID: {pyg_data.chain_id}")
        
        return pyg_data
        
    except Exception as e:
        print(f"Error processing protein: {e}")
        return None


def example_conversion_from_original():
    """Example: Convert existing BatchDataVQ3D format to PyG."""
    print("\n=== Example 2: Convert Original Format to PyG ===")
    
    # First, create data using the original pipeline
    original_pipeline = DataPipeline()
    
    pdb_path = "casp14_pdbs/T1025.pdb"
    if not os.path.exists(pdb_path):
        print(f"PDB file {pdb_path} not found. Skipping this example.")
        return None
    
    try:
        # Process with original pipeline
        original_data = original_pipeline.process_single(
            input_source=pdb_path,
            output_path="temp_original.npy"
        )
        
        # Convert to PyG format
        pyg_data = convert_batch_data_to_pyg(original_data)
        
        print("Successfully converted original format to PyG")
        print(f"Original graph nodes: {original_data.graph.n_node.item()}")
        print(f"PyG graph nodes: {pyg_data.num_nodes}")
        print(f"Original graph edges: {original_data.graph.n_edge.item()}")
        print(f"PyG graph edges: {pyg_data.num_edges}")
        
        # Verify that conversion preserves data
        original_coords = np.array(original_data.graph.nodes_original_coordinates)
        pyg_coords = pyg_data.pos.cpu().numpy()
        print(f"Coordinate preservation check: {np.allclose(original_coords, pyg_coords, atol=1e-5)}")
        
        return pyg_data
        
    except Exception as e:
        print(f"Error in conversion example: {e}")
        return None


def example_batch_processing():
    """Example: Process multiple proteins in batch."""
    print("\n=== Example 3: Batch Processing ===")
    
    # Find available PDB files
    import glob
    pdb_files = glob.glob("casp14_pdbs/*.pdb")[:3]  # Take first 3 files
    
    if len(pdb_files) == 0:
        print("No PDB files found in casp14_pdbs/. Skipping this example.")
        return None
    
    print(f"Processing {len(pdb_files)} proteins in batch")
    
    # Initialize pipeline
    config = {
        "num_neighbor": 15,
        "crop_index": 128,  # Smaller for batch processing
        "include_global_features": False,  # Reduce memory usage
    }
    pipeline = PyGDataPipeline(config)
    
    try:
        # Process batch
        batch_data = pipeline.process_batch(
            input_sources=pdb_files,
            output_dir="batch_outputs",
            input_type="pdb_file"
        )
        
        print("Batch processing completed!")
        print(f"Total proteins in batch: {len(batch_data)}")
        print(f"Total nodes: {batch_data.num_nodes}")
        print(f"Total edges: {batch_data.num_edges}")
        print(f"Batch tensor shape: {batch_data.batch.shape}")
        
        return batch_data
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return None


def example_model_usage():
    """Example: Use PyG data with a simple GNN model."""
    print("\n=== Example 4: Using PyG Data with GNN Model ===")
    
    # Create some example data
    pyg_data = example_single_protein()
    if pyg_data is None:
        print("Cannot demonstrate model usage without data. Skipping.")
        return
    
    # Create a simple dataset
    data_list = [pyg_data]
    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    
    # Initialize model
    input_dim = pyg_data.x.shape[1]  # 3D coordinates
    hidden_dim = 64
    output_dim = 32  # Some embedding dimension
    
    model = SimpleProteinGNN(input_dim, hidden_dim, output_dim)
    model.eval()
    
    print(f"Model initialized with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
    
    # Forward pass
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            print(f"Model output shape: {output.shape}")
            print(f"Model output: {output}")
            break
    
    print("Model successfully processed PyG data!")


def example_feature_analysis():
    """Example: Analyze features in PyG data."""
    print("\n=== Example 5: Feature Analysis ===")
    
    pyg_data = example_single_protein()
    if pyg_data is None:
        print("Cannot analyze features without data. Skipping.")
        return
    
    print("=== Graph Structure Analysis ===")
    print(f"Number of nodes: {pyg_data.num_nodes}")
    print(f"Number of edges: {pyg_data.num_edges}")
    print(f"Average degree: {pyg_data.num_edges / pyg_data.num_nodes:.2f}")
    
    print("\n=== Feature Analysis ===")
    print(f"Node features (x) shape: {pyg_data.x.shape}")
    print(f"Node positions (pos) shape: {pyg_data.pos.shape}")
    print(f"Edge features (edge_attr) shape: {pyg_data.edge_attr.shape}")
    print(f"Edge index shape: {pyg_data.edge_index.shape}")
    
    if hasattr(pyg_data, 'sequence'):
        print(f"Sequence length: {len(pyg_data.sequence)}")
        unique_aa = torch.unique(pyg_data.sequence)
        print(f"Unique amino acid types: {len(unique_aa)}")
    
    if hasattr(pyg_data, 'protein_features'):
        print("\n=== Protein Features ===")
        for key, value in pyg_data.protein_features.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {value}")
    
    print("\n=== Coordinate Statistics ===")
    coords = pyg_data.pos
    print(f"Coordinate range: x[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
          f"y[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}], "
          f"z[{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")
    print(f"Coordinate center: ({coords.mean(dim=0).tolist()})")


def main():
    """Run all examples."""
    print("PyTorch Geometric Data Pipeline Examples")
    print("=" * 50)
    
    # Create output directories
    os.makedirs("batch_outputs", exist_ok=True)
    
    # Run examples
    example_single_protein()
    example_conversion_from_original()
    example_batch_processing()
    example_model_usage()
    example_feature_analysis()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    
    # Cleanup temporary files
    temp_files = ["temp_original.npy", "example_output_T1024.pt"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Cleaned up {temp_file}")


if __name__ == "__main__":
    main()
