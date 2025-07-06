#!/usr/bin/env python3
"""
Data Pipeline Script for Protein Structure Preprocessing

This script provides a comprehensive pipeline for preprocessing raw protein data
(PDB files or PDB strings) into the format required by the VQ-VAE structure tokenizer.

Usage:
    python data_pipeline.py --input_path path/to/protein.pdb --output_path path/to/output
    python data_pipeline.py --pdb_string "ATOM ..." --output_path path/to/output
    python data_pipeline.py --input_path path/to/protein.pdb --config config.yaml
"""

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np
import yaml

# Add the current directory to the path to import the structure_tokenizer modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from structure_tokenizer.data.preprocessing import preprocess_sample, filter_out_sample
from structure_tokenizer.data.protein_structure_sample import (
    ProteinStructureSample,
    protein_structure_from_pdb_string,
)
from structure_tokenizer.utils.log import get_logger
from structure_tokenizer.types import BatchDataVQ3D

logger = get_logger(__name__)


class DataPipeline:
    """
    A comprehensive data pipeline for protein structure preprocessing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data pipeline with configuration parameters.

        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        # Default configuration
        self.config = {
            "num_neighbor": 30,
            "downsampling_ratio": 4,
            "residue_loc_is_alphac": True,
            "padding_num_residue": 512,
            "crop_index": 512,
            "noise_level": 0.0,
            "min_number_valid_residues": 10,
            "max_number_residues": 1000,
            "chain_id": None,  # If None, processes all chains
            "save_intermediate": False,
            "output_format": "npy",  # 'npy' or 'npz'
        }

        if config:
            self.config.update(config)

        logger.info(f"Initialized DataPipeline with config: {self.config}")

    def load_from_pdb_file(
        self, pdb_file_path: str, chain_id: Optional[str] = None
    ) -> ProteinStructureSample:
        """
        Load protein structure from a PDB file.

        Args:
            pdb_file_path: Path to the PDB file
            chain_id: Specific chain ID to extract (if None, processes all chains)

        Returns:
            ProteinStructureSample object
        """
        logger.info(f"Loading protein structure from {pdb_file_path}")

        if not os.path.exists(pdb_file_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")

        try:
            with open(pdb_file_path, "r") as f:
                pdb_string = f.read()

            sample = protein_structure_from_pdb_string(
                pdb_str=pdb_string, chain_id=chain_id or self.config["chain_id"]
            )

            logger.info(
                f"Successfully loaded protein with {sample.nb_residues} residues"
            )
            return sample

        except Exception as e:
            logger.error(f"Error loading PDB file {pdb_file_path}: {str(e)}")
            raise

    def load_from_pdb_string(
        self, pdb_string: str, chain_id: Optional[str] = None
    ) -> ProteinStructureSample:
        """
        Load protein structure from a PDB string.

        Args:
            pdb_string: PDB format string
            chain_id: Specific chain ID to extract (if None, processes all chains)

        Returns:
            ProteinStructureSample object
        """
        logger.info("Loading protein structure from PDB string")

        try:
            sample = protein_structure_from_pdb_string(
                pdb_str=pdb_string, chain_id=chain_id or self.config["chain_id"]
            )

            logger.info(
                f"Successfully loaded protein with {sample.nb_residues} residues"
            )
            return sample

        except Exception as e:
            logger.error(f"Error parsing PDB string: {str(e)}")
            raise

    def load_from_npy_file(self, npy_file_path: str) -> ProteinStructureSample:
        """
        Load protein structure from a saved numpy file.

        Args:
            npy_file_path: Path to the .npy file containing ProteinStructureSample data

        Returns:
            ProteinStructureSample object
        """
        logger.info(f"Loading protein structure from {npy_file_path}")

        try:
            sample = ProteinStructureSample.from_file(npy_file_path)
            logger.info(
                f"Successfully loaded protein with {sample.nb_residues} residues"
            )
            return sample

        except Exception as e:
            logger.error(f"Error loading .npy file {npy_file_path}: {str(e)}")
            raise

    def validate_sample(self, sample: ProteinStructureSample) -> bool:
        """
        Validate if the protein sample meets the filtering criteria.

        Args:
            sample: ProteinStructureSample to validate

        Returns:
            True if sample is valid, False otherwise
        """
        should_filter = filter_out_sample(
            sample=sample,
            min_number_valid_residues=self.config["min_number_valid_residues"],
            max_number_residues=self.config["max_number_residues"],
        )

        if should_filter:
            missing_coords_mask = sample.get_missing_backbone_coords_mask()
            num_valid_residues = np.sum(~missing_coords_mask)
            logger.warning(
                f"Sample filtered out: {num_valid_residues} valid residues "
                f"(min: {self.config['min_number_valid_residues']}, "
                f"max: {self.config['max_number_residues']}), "
                f"total residues: {sample.nb_residues}"
            )
            return False

        logger.info("Sample passed validation")
        return True

    def preprocess(self, sample: ProteinStructureSample) -> BatchDataVQ3D:
        """
        Preprocess the protein sample for model input.

        Args:
            sample: ProteinStructureSample to preprocess

        Returns:
            BatchDataVQ3D object ready for model input
        """
        logger.info("Starting preprocessing...")

        try:
            preprocessed_data = preprocess_sample(
                sample=sample,
                num_neighbor=self.config["num_neighbor"],
                downsampling_ratio=self.config["downsampling_ratio"],
                residue_loc_is_alphac=self.config["residue_loc_is_alphac"],
                padding_num_residue=self.config["padding_num_residue"],
                crop_index=self.config["crop_index"],
                noise_level=self.config["noise_level"],
            )

            logger.info("Preprocessing completed successfully")
            return preprocessed_data

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise

    def save_output(
        self,
        data: BatchDataVQ3D,
        output_path: str,
        save_raw_sample: bool = False,
        raw_sample: Optional[ProteinStructureSample] = None,
    ):
        """
        Save the preprocessed data to file.

        Args:
            data: Preprocessed BatchDataVQ3D object
            output_path: Path to save the output
            save_raw_sample: Whether to also save the raw ProteinStructureSample
            raw_sample: Raw ProteinStructureSample (required if save_raw_sample=True)
        """
        logger.info(f"Saving output to {output_path}")

        # Ensure output directory exists
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        try:
            if self.config["output_format"] == "npz":
                # Save as compressed numpy archive
                save_dict = {
                    "graph_n_node": data.graph.n_node,
                    "graph_n_edge": data.graph.n_edge,
                    "graph_nodes_mask": data.graph.nodes_mask,
                    "graph_nodes_original_coordinates": data.graph.nodes_original_coordinates,
                    "graph_node_features": data.graph.node_features,
                    "graph_edge_features": data.graph.edge_features,
                    "graph_tokens_mask": data.graph.tokens_mask,
                    "graph_senders": data.graph.senders,
                    "graph_receivers": data.graph.receivers,
                    "features": data.features,
                }
                np.savez_compressed(output_path, **save_dict)

            else:  # Default to .npy format
                np.save(output_path, data)

            logger.info(f"Successfully saved preprocessed data to {output_path}")

            # Save raw sample if requested
            if save_raw_sample and raw_sample is not None:
                raw_output_path = output_path.replace(".npy", "_raw.npy").replace(
                    ".npz", "_raw.npy"
                )
                raw_sample.to_file(raw_output_path)
                logger.info(f"Successfully saved raw sample to {raw_output_path}")

        except Exception as e:
            logger.error(f"Error saving output: {str(e)}")
            raise

    def process_single(
        self,
        input_source: Union[str, ProteinStructureSample],
        output_path: str,
        input_type: str = "auto",
    ) -> BatchDataVQ3D:
        """
        Process a single protein input through the complete pipeline.

        Args:
            input_source: Path to PDB file, PDB string, path to .npy file, or ProteinStructureSample
            output_path: Path to save the processed output
            input_type: Type of input ('pdb_file', 'pdb_string', 'npy_file', 'sample', or 'auto')

        Returns:
            Preprocessed BatchDataVQ3D object
        """
        logger.info("Starting single protein processing pipeline")

        # Load the sample based on input type
        if isinstance(input_source, ProteinStructureSample):
            sample = input_source
        elif input_type == "auto":
            if os.path.exists(str(input_source)):
                if str(input_source).endswith(".npy"):
                    sample = self.load_from_npy_file(str(input_source))
                else:
                    sample = self.load_from_pdb_file(str(input_source))
            else:
                # Assume it's a PDB string
                sample = self.load_from_pdb_string(str(input_source))
        elif input_type == "pdb_file":
            sample = self.load_from_pdb_file(str(input_source))
        elif input_type == "pdb_string":
            sample = self.load_from_pdb_string(str(input_source))
        elif input_type == "npy_file":
            sample = self.load_from_npy_file(str(input_source))
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        # Validate the sample
        if not self.validate_sample(sample):
            raise ValueError(
                "Sample failed validation - does not meet filtering criteria"
            )

        # Preprocess the sample
        preprocessed_data = self.preprocess(sample)

        # Save the output
        self.save_output(
            preprocessed_data,
            output_path,
            save_raw_sample=self.config.get("save_intermediate", False),
            raw_sample=sample,
        )

        logger.info("Pipeline processing completed successfully")
        return preprocessed_data

    def get_sample_info(self, sample: ProteinStructureSample) -> Dict[str, Any]:
        """
        Get information about the protein sample.

        Args:
            sample: ProteinStructureSample to analyze

        Returns:
            Dictionary containing sample information
        """
        missing_coords_mask = sample.get_missing_backbone_coords_mask()
        num_valid_residues = np.sum(~missing_coords_mask)

        return {
            "chain_id": sample.chain_id,
            "total_residues": sample.nb_residues,
            "valid_residues": int(num_valid_residues),
            "missing_residues": int(np.sum(missing_coords_mask)),
            "resolution": sample.resolution,
            "pdb_cluster_size": sample.pdb_cluster_size,
            "sequence": "".join(
                ["ACDEFGHIKLMNPQRSTVWYX"[np.argmax(aa)] for aa in sample.aatype]
            ),
        }


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Protein Structure Data Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a PDB file
    python data_pipeline.py --input_path protein.pdb --output_path processed.npy
    
    # Process with custom config
    python data_pipeline.py --input_path protein.pdb --config config.yaml --output_path processed.npy
    
    # Process a specific chain
    python data_pipeline.py --input_path protein.pdb --chain_id A --output_path processed.npy
    
    # Get sample information only
    python data_pipeline.py --input_path protein.pdb --info_only
        """,
    )

    # Input arguments
    parser.add_argument(
        "--input_path", type=str, help="Path to input PDB file or .npy file"
    )
    parser.add_argument(
        "--pdb_string", type=str, help="PDB format string (alternative to input_path)"
    )
    parser.add_argument("--output_path", type=str, help="Path to save processed output")

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
        "--padding_num_residue", type=int, default=512, help="Padding size for residues"
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.0,
        help="Noise level for data augmentation",
    )

    # Output arguments
    parser.add_argument(
        "--output_format", choices=["npy", "npz"], default="npy", help="Output format"
    )
    parser.add_argument(
        "--save_intermediate", action="store_true", help="Save intermediate raw sample"
    )
    parser.add_argument(
        "--info_only", action="store_true", help="Only display sample information"
    )

    args = parser.parse_args()

    # Validate input arguments
    if not args.input_path and not args.pdb_string:
        parser.error("Either --input_path or --pdb_string must be provided")

    if not args.info_only and not args.output_path:
        parser.error("--output_path is required unless --info_only is specified")

    try:
        # Load configuration
        config = {}
        if args.config:
            config = load_config_from_file(args.config)

        # Override config with command line arguments
        config.update(
            {
                "chain_id": args.chain_id,
                "num_neighbor": args.num_neighbor,
                "crop_index": args.crop_index,
                "padding_num_residue": args.padding_num_residue,
                "noise_level": args.noise_level,
                "output_format": args.output_format,
                "save_intermediate": args.save_intermediate,
            }
        )

        # Initialize pipeline
        pipeline = DataPipeline(config)

        # Determine input source
        input_source = args.input_path if args.input_path else args.pdb_string
        input_type = "pdb_string" if args.pdb_string else "auto"

        if args.info_only:
            # Load sample and display info
            if input_type == "pdb_string":
                sample = pipeline.load_from_pdb_string(input_source, args.chain_id)
            elif args.input_path.endswith(".npy"):
                sample = pipeline.load_from_npy_file(args.input_path)
            else:
                sample = pipeline.load_from_pdb_file(args.input_path, args.chain_id)

            info = pipeline.get_sample_info(sample)
            print("\nProtein Sample Information:")
            print("=" * 40)
            for key, value in info.items():
                print(f"{key:20}: {value}")
            print("=" * 40)

            # Check validation
            is_valid = pipeline.validate_sample(sample)
            print(f"Validation Status: {'PASS' if is_valid else 'FAIL'}")

        else:
            # Process the input
            processed_data = pipeline.process_single(
                input_source=input_source,
                output_path=args.output_path,
                input_type=input_type,
            )

            print(f"\nProcessing completed successfully!")
            print(f"Output saved to: {args.output_path}")
            print(f"Graph nodes: {processed_data.graph.n_node.item()}")
            print(f"Graph edges: {processed_data.graph.n_edge.item()}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        if logger.level <= 10:  # DEBUG level
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
