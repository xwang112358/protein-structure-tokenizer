#!/usr/bin/env python3
"""
ESM Protein Structure Preprocessing Script

This script preprocesses protein structure PDB files into ESM3-compatible format.
It uses the ESM data pipeline to extract backbone coordinates and other features
needed for structure-based models.

Usage:
    python esn_preprocess.py --input path/to/protein.pdb --output path/to/output.pt
    python esn_preprocess.py --input path/to/pdb_directory --output path/to/output_dir --batch
"""

import os
import argparse
import torch
from pathlib import Path
from typing import Optional

# Import the ESM data pipeline
# Note: The file is without extension in the workspace
from esm_data_pipeline import DataPipeline, load_config_from_file, debug_print


def preprocess_single_pdb(
    input_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    chain_id: Optional[str] = None,
) -> None:
    """
    Preprocess a single PDB file using the ESM data pipeline.

    Args:
        input_path: Path to the input PDB file
        output_path: Path to save the processed output
        config_path: Path to a YAML configuration file (optional)
        chain_id: Specific chain to process (optional)
    """
    # Load configuration if provided
    config = None
    if config_path:
        config = load_config_from_file(config_path)

    # Create data pipeline
    pipeline = DataPipeline(config=config)

    try:
        # Load and validate the protein sample
        sample = pipeline.load_from_pdb_file(input_path, chain_id=chain_id)

        # Get ESM3 data tensors
        esm_data = pipeline.get_esm3_data_for_protein(sample, as_tensors=True)

        # Get additional sample information
        sample_info = pipeline.get_sample_info(sample)

        # Combine data and info into one dictionary
        output_data = {**esm_data, "info": sample_info}

        # Save the processed data
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(output_data, output_path)

        print(f"Successfully processed {input_path} and saved to {output_path}")
        print(
            f"Processed {sample_info['valid_residues']} residues from chain {sample_info['chain_id'] or 'ALL'}"
        )

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        raise

    return output_data


def preprocess_batch(
    input_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
    file_ext: str = ".pdb",
) -> None:
    """
    Batch process multiple PDB files from a directory.

    Args:
        input_dir: Directory containing PDB files
        output_dir: Directory to save processed outputs
        config_path: Path to a YAML configuration file (optional)
        file_ext: File extension to filter input files (default: .pdb)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of PDB files
    input_path = Path(input_dir)
    pdb_files = list(input_path.glob(f"*{file_ext}"))

    if not pdb_files:
        print(f"No {file_ext} files found in {input_dir}")
        return

    print(f"Found {len(pdb_files)} {file_ext} files to process")

    # Process each file
    for i, pdb_file in enumerate(pdb_files):
        # Create output filename
        output_file = Path(output_dir) / f"{pdb_file.stem}.pt"

        print(f"[{i + 1}/{len(pdb_files)}] Processing {pdb_file.name}...")

        try:
            preprocess_single_pdb(
                str(pdb_file), str(output_file), config_path=config_path
            )
        except Exception as e:
            print(f"  Failed: {str(e)}")
            continue


def main():
    """Parse arguments and run preprocessing."""
    parser = argparse.ArgumentParser(description="ESM Protein Structure Preprocessing")

    parser.add_argument(
        "--input", "-i", required=True, help="Input PDB file or directory"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output file (.pt) or directory",
        default='esm_output.pt'
    )
    parser.add_argument("--config", "-c", help="Path to YAML configuration file")
    parser.add_argument("--chain", help="Specific chain ID to process")
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Process all PDB files in the input directory",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Configure debug mode
    if args.debug:
        debug_print("Debug mode enabled", "MAIN")

    # Process input(s)
    if args.batch:
        if not os.path.isdir(args.input):
            parser.error("--batch requires input to be a directory")

        preprocess_batch(args.input, args.output, config_path=args.config)
    else:
        single_data = preprocess_single_pdb(
            args.input, args.output, config_path=args.config, chain_id=args.chain
        )

        print(single_data)

    print("Processing complete")


if __name__ == "__main__":
    main()
