#!/usr/bin/env python3
"""
Example usage of the data_pipeline.py script

This script demonstrates different ways to use the DataPipeline class
for preprocessing protein structure data.
"""

import os
import sys
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import DataPipeline
from structure_tokenizer.data.protein_structure_sample import ProteinStructureSample


def example_1_basic_usage():
    """Example 1: Basic usage with a PDB file"""
    print("=" * 60)
    print("Example 1: Basic PDB file processing")
    print("=" * 60)
    
    # Initialize pipeline with default configuration
    pipeline = DataPipeline()
    
    # Check if we have a sample PDB file
    pdb_file = "casp14_pdbs/T1024.pdb"
    if not os.path.exists(pdb_file):
        print(f"Sample PDB file {pdb_file} not found. Skipping this example.")
        return
    
    try:
        # Process the PDB file
        output_path = "output_example1.npy"
        processed_data = pipeline.process_single(
            input_source=pdb_file,
            output_path=output_path
        )
        
        print(f"✓ Successfully processed {pdb_file}")
        print(f"✓ Output saved to {output_path}")
        print(f"  - Number of nodes: {processed_data.graph.n_node.item()}")
        print(f"  - Number of edges: {processed_data.graph.n_edge.item()}")
        
        # Load the sample and get info
        sample = pipeline.load_from_pdb_file(pdb_file)
        info = pipeline.get_sample_info(sample)
        print(f"  - Protein sequence length: {info['total_residues']}")
        print(f"  - Valid residues: {info['valid_residues']}")
        
    except Exception as e:
        print(f"✗ Error processing {pdb_file}: {e}")


def example_2_custom_config():
    """Example 2: Using custom configuration"""
    print("\n" + "=" * 60)
    print("Example 2: Custom configuration")
    print("=" * 60)
    
    # Custom configuration for smaller proteins
    custom_config = {
        'num_neighbor': 20,
        'crop_index': 256,
        'padding_num_residue': 256,
        'min_number_valid_residues': 5,
        'noise_level': 0.1,  # Add some noise for data augmentation
        'save_intermediate': True,
        'output_format': 'npz'
    }
    
    pipeline = DataPipeline(config=custom_config)
    
    pdb_file = "casp14_pdbs/T1025.pdb"
    if not os.path.exists(pdb_file):
        print(f"Sample PDB file {pdb_file} not found. Skipping this example.")
        return
    
    try:
        output_path = "output_example2.npz"
        processed_data = pipeline.process_single(
            input_source=pdb_file,
            output_path=output_path
        )
        
        print(f"✓ Successfully processed with custom config")
        print(f"✓ Output saved to {output_path}")
        print(f"  - Configuration: {custom_config}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_3_pdb_string():
    """Example 3: Processing from PDB string"""
    print("\n" + "=" * 60)
    print("Example 3: Processing from PDB string")
    print("=" * 60)
    
    # Simple PDB string for a small peptide (just a few atoms for demonstration)
    pdb_string = """
ATOM      1  N   ALA A   1      20.154  16.967  27.462  1.00 11.99           N  
ATOM      2  CA  ALA A   1      19.030  16.522  28.289  1.00 11.85           C  
ATOM      3  C   ALA A   1      18.114  15.704  27.409  1.00 11.75           C  
ATOM      4  O   ALA A   1      18.476  14.916  26.539  1.00 11.53           O  
ATOM      5  CB  ALA A   1      18.221  17.699  28.899  1.00 12.15           C  
ATOM      6  N   GLY A   2      16.849  15.831  27.556  1.00 11.81           N  
ATOM      7  CA  GLY A   2      15.877  15.108  26.777  1.00 11.90           C  
ATOM      8  C   GLY A   2      15.320  13.967  27.584  1.00 12.04           C  
ATOM      9  O   GLY A   2      15.031  14.137  28.771  1.00 12.19           O  
END
"""
    
    pipeline = DataPipeline()
    
    try:
        output_path = "output_example3.npy"
        processed_data = pipeline.process_single(
            input_source=pdb_string,
            output_path=output_path,
            input_type='pdb_string'
        )
        
        print(f"✓ Successfully processed PDB string")
        print(f"✓ Output saved to {output_path}")
        
        # Load sample info
        sample = pipeline.load_from_pdb_string(pdb_string)
        info = pipeline.get_sample_info(sample)
        print(f"  - Sequence: {info['sequence']}")
        print(f"  - Length: {info['total_residues']} residues")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_4_batch_processing():
    """Example 4: Batch processing multiple files"""
    print("\n" + "=" * 60)
    print("Example 4: Batch processing")
    print("=" * 60)
    
    pipeline = DataPipeline()
    
    # Find all PDB files in the casp14_pdbs directory
    pdb_dir = "casp14_pdbs"
    if not os.path.exists(pdb_dir):
        print(f"Directory {pdb_dir} not found. Skipping this example.")
        return
    
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')][:3]  # Process first 3 files
    
    if not pdb_files:
        print(f"No PDB files found in {pdb_dir}. Skipping this example.")
        return
    
    print(f"Processing {len(pdb_files)} PDB files...")
    
    successful = 0
    failed = 0
    
    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_dir, pdb_file)
        output_path = f"batch_output_{pdb_file.replace('.pdb', '.npy')}"
        
        try:
            # First check if the sample is valid
            sample = pipeline.load_from_pdb_file(pdb_path)
            if not pipeline.validate_sample(sample):
                print(f"⚠ Skipping {pdb_file}: Failed validation")
                continue
            
            processed_data = pipeline.process_single(
                input_source=pdb_path,
                output_path=output_path
            )
            
            info = pipeline.get_sample_info(sample)
            print(f"✓ {pdb_file}: {info['total_residues']} residues → {output_path}")
            successful += 1
            
        except Exception as e:
            print(f"✗ {pdb_file}: {e}")
            failed += 1
    
    print(f"\nBatch processing complete: {successful} successful, {failed} failed")


def example_5_info_only():
    """Example 5: Getting sample information without processing"""
    print("\n" + "=" * 60)
    print("Example 5: Sample information analysis")
    print("=" * 60)
    
    pipeline = DataPipeline()
    
    pdb_file = "casp14_pdbs/T1024.pdb"
    if not os.path.exists(pdb_file):
        print(f"Sample PDB file {pdb_file} not found. Skipping this example.")
        return
    
    try:
        # Load sample and get detailed information
        sample = pipeline.load_from_pdb_file(pdb_file)
        info = pipeline.get_sample_info(sample)
        
        print(f"Protein Information for {pdb_file}:")
        print("-" * 40)
        for key, value in info.items():
            print(f"{key:20}: {value}")
        
        # Check validation
        is_valid = pipeline.validate_sample(sample)
        print(f"{'Validation Status':20}: {'PASS' if is_valid else 'FAIL'}")
        
        # Additional analysis
        missing_coords_mask = sample.get_missing_backbone_coords_mask()
        print(f"{'Missing residues':20}: {np.sum(missing_coords_mask)}")
        print(f"{'Complete residues':20}: {np.sum(~missing_coords_mask)}")
        print(f"{'Completeness':20}: {100 * np.sum(~missing_coords_mask) / len(missing_coords_mask):.1f}%")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def cleanup_example_outputs():
    """Clean up example output files"""
    output_files = [
        "output_example1.npy", "output_example1_raw.npy",
        "output_example2.npz", "output_example2_raw.npy", 
        "output_example3.npy", "output_example3_raw.npy"
    ]
    
    # Also clean up batch outputs
    for f in os.listdir('.'):
        if f.startswith('batch_output_') and f.endswith('.npy'):
            output_files.append(f)
    
    cleaned = 0
    for output_file in output_files:
        if os.path.exists(output_file):
            os.remove(output_file)
            cleaned += 1
    
    if cleaned > 0:
        print(f"\n🧹 Cleaned up {cleaned} example output files")


def main():
    """Run all examples"""
    print("Protein Structure Data Pipeline Examples")
    print("=" * 60)
    
    try:
        example_1_basic_usage()
        example_2_custom_config()
        example_3_pdb_string()
        example_4_batch_processing()
        example_5_info_only()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
        # Ask user if they want to clean up output files
        response = input("\nClean up example output files? (y/n): ").strip().lower()
        if response in ('y', 'yes'):
            cleanup_example_outputs()
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")


if __name__ == "__main__":
    main()
