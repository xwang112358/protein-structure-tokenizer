#!/usr/bin/env python3
"""
Example script demonstrating batch processing of PDB files using PyG Data Pipeline.

This script shows how to:
1. Load all PDB files from a specified directory
2. Process them in batch using the PyGDataPipeline
3. Return a list of individual PyG data objects
4. Optionally save the batch and individual files
"""

import os
import glob
import torch
import argparse
import sys
from typing import List
from torch_geometric.data import Data, Batch

# Import the pipeline
from pyg_data_pipeline import PyGDataPipeline, get_num_edges
import pyg_data_pipeline


def process_pdb_batch(
    pdb_folder_path: str,
    save_individual: bool = False,
    save_batch: bool = False,
    save_data_list: bool = False,
    output_dir: str = "batch_outputs",
    debug: bool = True,
    config_overrides: dict = None
) -> List[Data]:
    """
    Process all PDB files in a specified directory in batch and return a list of PyG data objects.
    
    Args:
        pdb_folder_path: Path to directory containing PDB files
        save_individual: Whether to save individual PyG data files
        save_batch: Whether to save the batch file
        output_dir: Directory to save outputs
        debug: Enable debug printing
        config_overrides: Dictionary of configuration overrides
    
    Returns:
        List of PyTorch Geometric Data objects
    """
    
    # Set debug mode
    pyg_data_pipeline.DEBUG_MODE = debug
    
    print("="*60)
    print("PDB BATCH PROCESSING EXAMPLE")
    print("="*60)
    
    # Validate input directory
    if not os.path.exists(pdb_folder_path):
        raise FileNotFoundError(f"PDB directory not found: {pdb_folder_path}")
    
    # Find all PDB files
    pdb_pattern = os.path.join(pdb_folder_path, "*.pdb")
    pdb_files = glob.glob(pdb_pattern)
    
    if not pdb_files:
        raise ValueError(f"No PDB files found in {pdb_folder_path}")
    
    pdb_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(pdb_files)} PDB files in {pdb_folder_path}")
    print("Files to process:")
    for i, pdb_file in enumerate(pdb_files, 1):
        print(f"  {i:2d}. {os.path.basename(pdb_file)}")
    
    # Create output directory if saving files
    if save_individual or save_batch:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Configure the pipeline
    config = {
        "num_neighbor": 30,
        "crop_index": 512,
        "noise_level": 0.0,
        "min_number_valid_residues": 10,
        "max_number_residues": 1000,
        "output_format": "pt",
        "dtype": torch.float32,
        "device": "cpu",
        "include_global_features": True,
        "save_intermediate": False,
    }
    
    # Apply any configuration overrides
    if config_overrides:
        config.update(config_overrides)
        print(f"Applied config overrides: {list(config_overrides.keys())}")
    
    print(f"Pipeline configuration: {config}")
    print("\n" + "="*60)
    
    # Initialize the pipeline
    pipeline = PyGDataPipeline(config)
    
    # Process the batch
    print("Starting batch processing...")
    try:
        batch_data = pipeline.process_batch(
            input_sources=pdb_files,
            output_dir=output_dir if save_individual else None,
            input_type="pdb_file",
        )
        
        print("Batch processing completed successfully!")
        
        # Extract individual data objects from the batch
        individual_data_list = batch_data.to_data_list()

        if save_data_list:
            data_list_filename = f"data_list_{os.path.basename(pdb_folder_path)}.pt"
            data_list_path = os.path.join(output_dir, data_list_filename)
            torch.save(individual_data_list, data_list_path)    
            print(f"data list saved to: {data_list_path}")
        
        print("\nBatch Summary:")
        print(f"  Total proteins processed: {len(individual_data_list)}")
        print(f"  Total nodes in batch: {batch_data.num_nodes}")
        print(f"  Total edges in batch: {get_num_edges(batch_data)}")
        print(f"  Batch size: {batch_data.batch.max().item() + 1}")
        
        # Save batch if requested
        if save_batch:
            batch_filename = f"batch_{os.path.basename(pdb_folder_path)}.pt"
            batch_path = os.path.join(output_dir, batch_filename)
            torch.save(batch_data, batch_path)
            print(f"  Batch saved to: {batch_path}")
        
        # Print information about individual data objects
        print("\nIndividual Data Objects:")
        for i, data in enumerate(individual_data_list):
            pdb_name = os.path.basename(pdb_files[i]).replace('.pdb', '')
            print(f"  {i+1:2d}. {pdb_name:<10} - Nodes: {data.num_nodes:3d}, Edges: {get_num_edges(data):4d}")
            
            # Print additional info if available
            if hasattr(data, 'chain_id'):
                print(f"      Chain: {data.chain_id}, Resolution: {getattr(data, 'resolution', 'N/A')}")
        
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return individual_data_list
        
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        raise


def analyze_batch_results(data_list: List[Data]):
    """
    Analyze the results of batch processing.
    
    Args:
        data_list: List of PyG Data objects to analyze
    """
    print("\n" + "="*60)
    print("BATCH ANALYSIS")
    print("="*60)
    
    if not data_list:
        print("No data objects to analyze.")
        return
    
    # Basic statistics
    total_nodes = sum(data.num_nodes for data in data_list)
    total_edges = sum(get_num_edges(data) for data in data_list)
    
    node_counts = [data.num_nodes for data in data_list]
    edge_counts = [get_num_edges(data) for data in data_list]
    
    print("Dataset Statistics:")
    print(f"  Number of proteins: {len(data_list)}")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Total edges: {total_edges}")
    print(f"  Average nodes per protein: {total_nodes / len(data_list):.1f}")
    print(f"  Average edges per protein: {total_edges / len(data_list):.1f}")
    
    print("\nNode Count Distribution:")
    print(f"  Min: {min(node_counts)}")
    print(f"  Max: {max(node_counts)}")
    print(f"  Mean: {sum(node_counts) / len(node_counts):.1f}")
    
    print("\nEdge Count Distribution:")
    print(f"  Min: {min(edge_counts)}")
    print(f"  Max: {max(edge_counts)}")
    print(f"  Mean: {sum(edge_counts) / len(edge_counts):.1f}")
    
    # Feature analysis
    if data_list:
        sample_data = data_list[0]
        print("\nFeature Information (from first sample):")
        print(f"  Node features shape: {sample_data.x.shape}")
        print(f"  Edge features shape: {sample_data.edge_attr.shape}")
        print(f"  Has sequence info: {hasattr(sample_data, 'sequence')}")
        print(f"  Has position info: {hasattr(sample_data, 'pos')}")
        print(f"  Has protein features: {hasattr(sample_data, 'protein_features')}")
        
        if hasattr(sample_data, 'protein_features'):
            print(f"  Protein features keys: {list(sample_data.protein_features.keys())}")



def main():
    """Main function demonstrating different usage patterns."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Batch process PDB files using PyG Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process CASP14 files
    python example_batch_processing.py --pdb_folder casp14_pdbs
    
    # Process with saving and custom output directory
    python example_batch_processing.py --pdb_folder casp14_pdbs --save_individual --save_batch --output_dir my_outputs
    
    # Process with custom config
    python example_batch_processing.py --pdb_folder casp14_pdbs --crop_index 256 --num_neighbor 20
    
    # Quiet mode (less verbose)
    python example_batch_processing.py --pdb_folder casp14_pdbs --quiet
        """
    )
    
    parser.add_argument(
        "--pdb_folder", 
        type=str, 
        required=True,
        help="Path to directory containing PDB files"
    )
    parser.add_argument(
        "--save_individual", 
        action="store_true",
        help="Save individual PyG data files"
    )
    parser.add_argument(
        "--save_data_list",
        action="store_true",
        help="Save the list of individual PyG data objects"
    )
    parser.add_argument(
        "--save_batch", 
        action="store_true",
        help="Save the batch file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="batch_outputs",
        help="Directory to save outputs (default: batch_outputs)"
    )
    parser.add_argument(
        "--crop_index", 
        type=int, 
        default=512,
        help="Maximum sequence length after cropping (default: 512)"
    )
    parser.add_argument(
        "--num_neighbor", 
        type=int, 
        default=30,
        help="Number of nearest neighbors (default: 30)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Disable debug output"
    )
    
    args = parser.parse_args()
    
    # Set up configuration overrides
    config_overrides = {
        "crop_index": args.crop_index,
        "num_neighbor": args.num_neighbor,
    }
    
    print("PDB Batch Processing Example")
    print(f"Processing PDB files from: {args.pdb_folder}")
    print(f"Configuration: crop_index={args.crop_index}, num_neighbor={args.num_neighbor}")
    
    # Process the batch
    try:
        data_list = process_pdb_batch(
            pdb_folder_path=args.pdb_folder,
            save_individual=args.save_individual,
            save_data_list=args.save_data_list,
            save_batch=args.save_batch,
            output_dir=args.output_dir,
            debug=not args.quiet,
            config_overrides=config_overrides
        )
        
        # Analyze the results
        analyze_batch_results(data_list)
        
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETED SUCCESSFULLY!")
        print(f"Processed {len(data_list)} proteins from {args.pdb_folder}")
        print("="*60)
        
    except Exception as e:
        print(f"Batch processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0




if __name__ == "__main__":
    main()