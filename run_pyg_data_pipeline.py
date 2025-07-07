#!/usr/bin/env python3
"""
Run PyTorch Geometric Data Pipeline Script for Protein Structure Preprocessing

This script provides a command-line interface for the PyG data preprocessing pipeline.

Usage:
    python run_pyg_data_pipeline.py --input_path path/to/protein.pdb --output_path path/to/output.pt
    python run_pyg_data_pipeline.py --pdb_string "ATOM ..." --output_path path/to/output.pt
    python run_pyg_data_pipeline.py --input_path path/to/protein.pdb --config config.yaml
"""

import argparse
import os
import sys
import traceback
import glob

# PyTorch imports
import torch

# Import the pipeline classes and functions
from pyg_data_pipeline import (
    PyGDataPipeline,
    load_config_from_file,
    debug_print
)

# Make DEBUG_MODE and OUTPUT_FILE available globally
import pyg_data_pipeline


def main():
    """Main function for command-line interface."""
    
    parser = argparse.ArgumentParser(
        description="Protein Structure PyTorch Geometric Data Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a PDB file to PyG format
    python run_pyg_data_pipeline.py --input_path protein.pdb --output_path processed.pt
    
    # Process with custom config
    python run_pyg_data_pipeline.py --input_path protein.pdb --config config.yaml --output_path processed.pt
    
    # Process a specific chain
    python run_pyg_data_pipeline.py --input_path protein.pdb --chain_id A --output_path processed.pt
    
    # Get sample information only
    python run_pyg_data_pipeline.py --input_path protein.pdb --info_only
    
    # Process multiple files in a directory
    python run_pyg_data_pipeline.py --input_dir path/to/pdbs --output_dir path/to/outputs
    
    # Debug mode with output to file
    python run_pyg_data_pipeline.py --input_path protein.pdb --output_path processed.pt --debug --log_file debug.log
        """,
    )

    # Input arguments
    parser.add_argument(
        "--input_path", type=str, help="Path to input PDB file or .npy file"
    )
    parser.add_argument(
        "--pdb_string", type=str, help="PDB format string (alternative to input_path)"
    )
    parser.add_argument(
        "--input_dir", type=str, help="Directory containing PDB files to process"
    )
    parser.add_argument("--output_path", type=str, help="Path to save processed output")
    parser.add_argument("--output_dir", type=str, help="Directory to save processed outputs")

    # Configuration arguments
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--chain_id", type=str, help="Specific chain ID to process")
    parser.add_argument(
        "--num_neighbor", type=int, default=30, help="Number of nearest neighbors"
    )
    parser.add_argument(
        "--crop_index",
        type=int,
        default=512,
        help="Maximum sequence length after cropping",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.0,
        help="Noise level for data augmentation",
    )

    # PyG-specific arguments
    parser.add_argument(
        "--output_format", choices=["pt", "pkl"], default="pt", help="Output format"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for tensor creation"
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float64"], default="float32", help="Data type"
    )
    parser.add_argument(
        "--save_intermediate", action="store_true", help="Save intermediate raw sample"
    )
    parser.add_argument(
        "--info_only", action="store_true", help="Only display sample information"
    )
    
    # Debug arguments
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with verbose output"
    )
    parser.add_argument(
        "--log_file", type=str, help="File to save debug output (in addition to terminal)"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Disable debug output (opposite of --debug)"
    )

    args = parser.parse_args()

    # Set debug mode in both local and module scope
    if args.quiet:
        pyg_data_pipeline.DEBUG_MODE = False
    elif args.debug:
        pyg_data_pipeline.DEBUG_MODE = True
    
    # Set output file for logging
    if args.log_file:
        pyg_data_pipeline.OUTPUT_FILE = args.log_file
        # Clear the log file
        with open(args.log_file, 'w') as f:
            f.write("=== PyG Data Pipeline Debug Log ===\n\n")
        debug_print(f"Debug output will be saved to: {args.log_file}", "MAIN")

    # Validate input arguments
    if not args.input_path and not args.pdb_string and not args.input_dir:
        parser.error("Either --input_path, --pdb_string, or --input_dir must be provided")

    if not args.info_only and not args.output_path and not args.output_dir:
        parser.error("Output path/directory is required unless --info_only is specified")

    debug_print("=== Starting PyG Data Pipeline ===", "MAIN")
    debug_print(f"Arguments: {vars(args)}", "MAIN", 1)

    try:
        # Load configuration
        config = {}
        if args.config:
            debug_print(f"Loading config from: {args.config}", "MAIN", 1)
            config = load_config_from_file(args.config)

        # Convert dtype string to torch dtype
        dtype_map = {"float32": torch.float32, "float64": torch.float64}
        dtype = dtype_map[args.dtype]

        # Override config with command line arguments
        config.update(
            {
                "chain_id": args.chain_id,
                "num_neighbor": args.num_neighbor,
                "crop_index": args.crop_index,
                "noise_level": args.noise_level,
                "output_format": args.output_format,
                "save_intermediate": args.save_intermediate,
                "dtype": dtype,
                "device": args.device,
            }
        )

        # Initialize pipeline
        debug_print("Initializing pipeline...", "MAIN", 1)
        pipeline = PyGDataPipeline(config)

        if args.input_dir:
            # Process directory of PDB files
            debug_print(f"Processing directory: {args.input_dir}", "MAIN", 1)
            pdb_files = glob.glob(os.path.join(args.input_dir, "*.pdb"))
            if not pdb_files:
                error_msg = f"No PDB files found in {args.input_dir}"
                debug_print(f"ERROR: {error_msg}", "MAIN")
                print(f"ERROR: {error_msg}")
                sys.exit(1)
            
            debug_print(f"Found {len(pdb_files)} PDB files to process", "MAIN", 1)
            print(f"\nFound {len(pdb_files)} PDB files to process")
            
            # Process batch
            batch_data = pipeline.process_batch(
                input_sources=pdb_files,
                output_dir=args.output_dir,
                input_type="pdb_file",
            )
            
            # Save batch if output directory specified
            if args.output_dir:
                batch_path = os.path.join(args.output_dir, "batch.pt")
                debug_print(f"Saving batch to: {batch_path}", "MAIN", 1)
                torch.save(batch_data, batch_path)
                debug_print("Batch saved ✓", "MAIN", 1)
            
            print("\n" + "="*50)
            print("BATCH PROCESSING COMPLETED!")
            print("="*50)
            print(f"Successfully processed: {len(batch_data)} proteins")
            print(f"Total nodes: {batch_data.num_nodes}")
            print(f"Total edges: {batch_data.num_edges}")
            if args.output_dir:
                print(f"Batch saved to: {batch_path}")
            
        else:
            # Process single protein
            # Determine input source
            input_source = args.input_path if args.input_path else args.pdb_string
            input_type = "pdb_string" if args.pdb_string else "auto"
            
            debug_print(f"Processing single protein: {input_source[:50]}...", "MAIN", 1)

            if args.info_only:
                debug_print("Info-only mode enabled", "MAIN", 1)
                # Load sample and display info
                if input_type == "pdb_string":
                    sample = pipeline.load_from_pdb_string(input_source, args.chain_id)
                elif args.input_path.endswith(".npy"):
                    sample = pipeline.load_from_npy_file(args.input_path)
                else:
                    sample = pipeline.load_from_pdb_file(args.input_path, args.chain_id)

                info = pipeline.get_sample_info(sample)
                print("\n" + "="*50)
                print("PROTEIN SAMPLE INFORMATION")
                print("="*50)
                for key, value in info.items():
                    print(f"{key:20}: {value}")
                print("="*50)

                # Check validation
                is_valid = pipeline.validate_sample(sample)
                print(f"Validation Status: {'PASS ✓' if is_valid else 'FAIL ✗'}")

            else:
                # Process the input
                pyg_data = pipeline.process_single(
                    input_source=input_source,
                    output_path=args.output_path,
                    input_type=input_type,
                )

                print("\n" + "="*50)
                print("PYG PROCESSING COMPLETED!")
                print("="*50)
                print(f"Output saved to: {args.output_path}")
                print(f"Graph nodes: {pyg_data.num_nodes}")
                print(f"Graph edges: {pyg_data.num_edges}")
                print(f"Node features shape: {pyg_data.x.shape}")
                print(f"Edge features shape: {pyg_data.edge_attr.shape}")
                print("="*50)

        debug_print("=== Pipeline completed successfully ===", "MAIN")

    except Exception as e:
        error_msg = f"PyG pipeline failed: {str(e)}"
        debug_print(f"FATAL ERROR: {error_msg}", "MAIN")
        print(f"\n{'='*50}")
        print("PIPELINE FAILED!")
        print("="*50)
        print(f"Error: {error_msg}")
        if pyg_data_pipeline.DEBUG_MODE:
            print("\nFull traceback:")
            traceback.print_exc()
        print("="*50)
        sys.exit(1)


if __name__ == "__main__":
    main()
