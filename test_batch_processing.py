#!/usr/bin/env python3
"""
Simple test script for batch processing of PDB files.

This script demonstrates the basic usage of process_batch method and returns
a list of PyG data objects.
"""

import os
import glob
import torch
import sys
from pyg_data_pipeline import PyGDataPipeline, safe_get_num_edges
import pyg_data_pipeline

def test_process_batch(pdb_folder_path: str):
    """
    Test the process_batch method with PDB files from a specified directory.
    
    Args:
        pdb_folder_path: Path to directory containing PDB files
    
    Returns:
        List of PyTorch Geometric Data objects
    """
    
    # Enable debug mode for detailed output
    pyg_data_pipeline.DEBUG_MODE = True
    
    print(f"Testing process_batch with PDB files from: {pdb_folder_path}")
    print("=" * 60)
    
    # Validate directory exists
    if not os.path.exists(pdb_folder_path):
        raise FileNotFoundError(f"PDB directory not found: {pdb_folder_path}")
    
    # Get all PDB files from the directory
    pdb_files = sorted(glob.glob(os.path.join(pdb_folder_path, "*.pdb")))
    
    if not pdb_files:
        raise ValueError(f"No PDB files found in {pdb_folder_path}")
    
    print(f"Found {len(pdb_files)} PDB files:")
    for i, pdb_file in enumerate(pdb_files[:5]):  # Show first 5
        print(f"  {i+1}. {os.path.basename(pdb_file)}")
    if len(pdb_files) > 5:
        print(f"  ... and {len(pdb_files) - 5} more files")
    
    # Configure pipeline with reasonable settings
    config = {
        "num_neighbor": 20,          # Fewer neighbors for faster processing
        "crop_index": 256,           # Smaller crop for faster processing
        "noise_level": 0.0,          # No noise for testing
        "min_number_valid_residues": 10,
        "max_number_residues": 1000,
        "dtype": torch.float32,
        "device": "cpu",
        "include_global_features": True,
    }
    
    # Initialize pipeline
    pipeline = PyGDataPipeline(config)
    
    # Process the batch
    print("\nStarting batch processing...")
    batch_data = pipeline.process_batch(
        input_sources=pdb_files,
        input_type="pdb_file"
    )
    
    # Convert batch to list of individual data objects
    data_list = batch_data.to_data_list()
    
    print("\nBatch processing completed!")
    print(f"Successfully processed: {len(data_list)} proteins")
    print(f"Total nodes in batch: {batch_data.num_nodes}")
    print(f"Total edges in batch: {safe_get_num_edges(batch_data)}")
    
    # Print details for each processed protein
    print("\nIndividual protein details:")
    for i, data in enumerate(data_list):
        pdb_name = os.path.basename(pdb_files[i]).replace('.pdb', '')
        print(f"  {i+1:2d}. {pdb_name:<8} - Nodes: {data.num_nodes:3d}, Edges: {safe_get_num_edges(data):4d}")
        
        # Verify the data structure
        assert hasattr(data, 'x'), f"Data {i} missing node features"
        assert hasattr(data, 'edge_index'), f"Data {i} missing edge index"
        assert hasattr(data, 'edge_attr'), f"Data {i} missing edge attributes"
        assert data.x.shape[0] == data.num_nodes, f"Data {i} node count mismatch"
        assert data.edge_index.shape[1] == safe_get_num_edges(data), f"Data {i} edge count mismatch"
    
    print("\nAll data objects verified successfully!")
    print("=" * 50)
    
    return data_list


def demonstrate_data_usage(data_list):
    """
    Demonstrate how to use the returned data objects.
    
    Args:
        data_list: List of PyG Data objects
    """
    print("Demonstrating data usage:")
    print("-" * 30)
    
    if not data_list:
        print("No data objects to demonstrate")
        return
    
    # Example 1: Access individual protein data
    sample_data = data_list[0]
    print("Sample protein (first in list):")
    print(f"  Node features shape: {sample_data.x.shape}")
    print(f"  Edge index shape: {sample_data.edge_index.shape}")
    print(f"  Edge features shape: {sample_data.edge_attr.shape}")
    
    # Example 2: Create a new batch from subset
    subset = data_list[:3]  # First 3 proteins
    from torch_geometric.data import Batch
    new_batch = Batch.from_data_list(subset)
    print("\nCreated new batch from 3 proteins:")
    print(f"  Batch nodes: {new_batch.num_nodes}")
    print(f"  Batch edges: {safe_get_num_edges(new_batch)}")
    
    # Example 3: Iterate through all proteins
    print(f"\nQuick summary of all {len(data_list)} proteins:")
    node_counts = [data.num_nodes for data in data_list]
    edge_counts = [safe_get_num_edges(data) for data in data_list]
    print(f"  Min nodes: {min(node_counts)}")
    print(f"  Max nodes: {max(node_counts)}")
    print(f"  Avg nodes: {sum(node_counts) / len(node_counts):.1f}")
    print(f"  Min edges: {min(edge_counts)}")
    print(f"  Max edges: {max(edge_counts)}")
    print(f"  Avg edges: {sum(edge_counts) / len(edge_counts):.1f}")
    
    # Example 4: Check for additional attributes
    sample = data_list[0]
    print("\nAvailable attributes in data objects:")
    attrs = [attr for attr in dir(sample) if not attr.startswith('_')]
    for attr in sorted(attrs):
        if hasattr(sample, attr):
            value = getattr(sample, attr)
            if torch.is_tensor(value):
                print(f"  {attr}: tensor shape {value.shape}")
            else:
                print(f"  {attr}: {type(value).__name__}")


if __name__ == "__main__":
    print("PDB Process Batch Test")
    print("=" * 50)
    
    # Default to CASP14 directory if no argument provided
    pdb_folder = "casp14_pdbs"
    
    # Check if folder path provided as command line argument
    if len(sys.argv) > 1:
        pdb_folder = sys.argv[1]
    
    print(f"Using PDB folder: {pdb_folder}")
    
    try:
        # Test the batch processing
        data_list = test_process_batch(pdb_folder)
        
        # Demonstrate usage
        demonstrate_data_usage(data_list)
        
        print("\n" + "=" * 50)
        print("TEST COMPLETED SUCCESSFULLY!")
        print(f"Returned {len(data_list)} PyG data objects")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
