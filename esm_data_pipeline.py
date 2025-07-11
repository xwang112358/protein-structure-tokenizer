#!/usr/bin/env python3
"""
PyTorch Geometric Data Pipeline Script for Protein Structure Preprocessing

This script provides a comprehensive pipeline for preprocessing raw protein data
(PDB files or PDB strings) into PyTorch Geometric format for graph neural networks.

Usage:
    python pyg_data_pipeline.py --input_path path/to/protein.pdb --output_path path/to/output.pt
    python pyg_data_pipeline.py --pdb_string "ATOM ..." --output_path path/to/output.pt
    python pyg_data_pipeline.py --input_path path/to/protein.pdb --config config.yaml
"""

import os
import sys
from typing import Optional, Dict, Any, Union
import numpy as np
import yaml
import torch

# Add the current directory to the path to import the structure_tokenizer modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from structure_tokenizer.data.preprocessing import filter_out_sample
from structure_tokenizer.data.protein_structure_sample import (
    ProteinStructureSample,
    protein_structure_from_pdb_string,
)
from structure_tokenizer.data import residue_constants

# logger = get_logger(__name__)  # DISABLED

# Global debug flag
DEBUG_MODE = True
OUTPUT_FILE = None


def debug_print(message: str, stage: str = "DEBUG", indent: int = 0):
    """Print debug message with stage info and optional indentation."""
    if DEBUG_MODE:
        indent_str = "  " * indent
        formatted_msg = f"[{stage}] {indent_str}{message}"
        print(formatted_msg)
        if OUTPUT_FILE:
            with open(OUTPUT_FILE, 'a') as f:
                f.write(formatted_msg + '\n')


def progress_indicator(current: int, total: int, stage: str = "Processing"):
    """Simple progress indicator."""
    if DEBUG_MODE and total > 1:
        percentage = (current / total) * 100
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f'\r{stage}: |{bar}| {percentage:.1f}% ({current}/{total})', end='')
        if current == total:
            print()  # New line when complete


class DataPipeline:
    """
    A comprehensive data pipeline for protein structure preprocessing with PyTorch Geometric output.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the PyG data pipeline with configuration parameters.

        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        debug_print("Initializing PyGDataPipeline", "INIT")
        
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
            "output_format": "pt",  # 'pt' (torch) or 'pkl' (pickle)
            "include_global_features": True,
            "dtype": torch.float32,
            "device": "cpu",
        }

        if config:
            self.config.update(config)
            debug_print(f"Updated config with user settings: {list(config.keys())}", "INIT", 1)

        debug_print(f"Pipeline initialized with config keys: {list(self.config.keys())}", "INIT", 1)

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
        debug_print(f"Loading protein from PDB file: {pdb_file_path}", "LOAD")

        if not os.path.exists(pdb_file_path):
            error_msg = f"PDB file not found: {pdb_file_path}"
            raise FileNotFoundError(error_msg)

        try:
            with open(pdb_file_path, "r") as f:
                pdb_string = f.read()

            # debug_print(f"PDB content length: {len(pdb_string)} characters", "LOAD", 1)
            # debug_print(f"Processing chain_id: {chain_id or self.config['chain_id'] or 'ALL'}", "LOAD", 1)
            
            sample = protein_structure_from_pdb_string(
                pdb_str=pdb_string, chain_id=chain_id or self.config["chain_id"]
            )

            # debug_print(f"Successfully loaded protein with {sample.nb_residues} residues", "LOAD", 1)
            return sample

        except Exception as e:
            error_msg = f"Failed to load PDB file {pdb_file_path}: {str(e)}"
            debug_print(f"ERROR: {error_msg}", "LOAD")
            debug_print(f"Exception type: {type(e).__name__}", "LOAD", 1)
            raise Exception(error_msg) from e


    def validate_sample(self, sample: ProteinStructureSample) -> bool:
        """
        Validate if the protein sample meets the filtering criteria.

        Args:
            sample: ProteinStructureSample to validate

        Returns:
            True if sample is valid, False otherwise
        """
        debug_print("Validating protein sample", "VALIDATE")
        
        # Helper function to safely convert to numpy int/float
        def to_scalar(value):
            """Convert JAX arrays or other scalar-like objects to Python scalar."""
            if hasattr(value, "item"):
                return value.item()
            if hasattr(value, "to_py") and callable(getattr(value, "to_py")):
                return value.to_py()
            return value
        
        should_filter = filter_out_sample(
            sample=sample,
            min_number_valid_residues=self.config["min_number_valid_residues"],
            max_number_residues=self.config["max_number_residues"],
        )

        if should_filter:
            missing_coords_mask = sample.get_missing_backbone_coords_mask()
            # Handle JAX arrays safely
            if hasattr(missing_coords_mask, "to_py") and callable(getattr(missing_coords_mask, "to_py")):
                missing_coords_mask = missing_coords_mask.to_py()
            
            num_valid_residues = np.sum(~missing_coords_mask)
            debug_print("VALIDATION FAILED:", "VALIDATE", 1)
            debug_print(f"  Valid residues: {to_scalar(num_valid_residues)}", "VALIDATE", 2)
            debug_print(f"  Missing residues: {to_scalar(np.sum(missing_coords_mask))}", "VALIDATE", 2)
            debug_print(f"  Total residues: {to_scalar(sample.nb_residues)}", "VALIDATE", 2)
            return False

        return True

    def _get_esm3_data(self,
                       sample: ProteinStructureSample,
                       add_batch_dim: bool = True) -> Dict[str, np.ndarray]:
        """
        Convert ProteinStructureSample to ESM3-compatible format.
        
        Args:
            sample: Protein structure sample
            add_batch_dim: Whether to add batch dimension to outputs (default: True)
            
        Returns:
            Dictionary with ESM3-compatible tensors:
                - coords: (B, L, 3, 3) 3D coordinates of N, CA, C atoms
                - attention_mask: (B, L) Valid residue mask
                - sequence_id: (B, L) Sequence identifiers
                - residue_index: (B, L) Residue position indices
        """
        debug_print("Extracting ESM3 data from protein sample", "ESM3")
        
        # Helper function to safely convert to numpy array
        def to_numpy(arr):
            """Convert JAX arrays or other array-like objects to numpy arrays."""
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                return arr
            # Handle JAX arrays
            if hasattr(arr, "to_py") and callable(getattr(arr, "to_py")):
                return arr.to_py()
            # Try normal numpy conversion for everything else
            return np.array(arr)
        
        # Get mask for residues with missing backbone atoms
        missing_coords_residue_mask = to_numpy(sample.get_missing_backbone_coords_mask())  # Shape: (L,)
        num_residues_with_coords = np.sum(~missing_coords_residue_mask)
        
        debug_print(f"Initial residues with coordinates: {num_residues_with_coords}", "ESM3", 1)

        # Get indices of backbone atoms
        (n_index, ca_index, c_index) = [
            residue_constants.atom_order[a] for a in ("N", "CA", "C")
        ]

        # Extract backbone atom coordinates
        atom_positions = to_numpy(sample.atom37_positions)
        n_xyz = atom_positions[:, n_index, :]   # Shape: (L, 3)
        ca_xyz = atom_positions[:, ca_index, :]  # Shape: (L, 3)
        c_xyz = atom_positions[:, c_index, :]   # Shape: (L, 3)

        # Stack coordinates to get (L, 3, 3) - for each residue, coords of N, CA, C atoms
        coords = np.stack([n_xyz, ca_xyz, c_xyz], axis=1)  # Shape: (L, 3, 3)
        
        # Create attention mask (1 = valid residue, 0 = invalid/missing residue)
        attention_mask = ~missing_coords_residue_mask  # Shape: (L,)
        
        # Create sequence IDs (using amino acid types)
        if hasattr(sample, 'aatype') and sample.aatype is not None:
            aatype = to_numpy(sample.aatype)
            if len(aatype.shape) == 2:
                # One-hot encoded amino acids - convert to indices
                sequence_id = np.argmax(aatype, axis=-1)  # Shape: (L,)
            else:
                # Already indices
                sequence_id = aatype  # Shape: (L,)
        else:
            # Fallback: use placeholder values
            sequence_id = np.zeros(len(missing_coords_residue_mask), dtype=np.int32)
        
        # Create residue indices (sequential positions)
        residue_index = np.arange(len(missing_coords_residue_mask), dtype=np.int32)  # Shape: (L,)
        
        # Add batch dimension if requested (B=1 for single protein)
        if add_batch_dim:
            coords = coords[np.newaxis, ...]           # Shape: (1, L, 3, 3)
            attention_mask = attention_mask[np.newaxis, ...]  # Shape: (1, L)
            sequence_id = sequence_id[np.newaxis, ...]      # Shape: (1, L)
            residue_index = residue_index[np.newaxis, ...]   # Shape: (1, L)
        
        # Package results
        debug_print(f"ESM3 data extracted - coords shape: {coords.shape}", "ESM3", 1)
        return {
            "coords": coords,                  # Shape: (B, L, 3, 3) or (L, 3, 3)
            "attention_mask": attention_mask,  # Shape: (B, L) or (L,)
            "sequence_id": sequence_id,        # Shape: (B, L) or (L,)
            "residue_index": residue_index     # Shape: (B, L) or (L,)
        }
        

    def get_esm3_data_for_protein(self, 
                                  input_source: Union[str, ProteinStructureSample],
                                  as_tensors: bool = True) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Convenience method to get ESM3-compatible data for a protein.
        
        Args:
            input_source: Path to PDB file or ProteinStructureSample object
            as_tensors: Whether to return PyTorch tensors (True) or numpy arrays (False)
            
        Returns:
            Dictionary with ESM3-compatible data
        """
        # Load the protein if needed
        if isinstance(input_source, ProteinStructureSample):
            sample = input_source
        else:
            sample = self.load_from_pdb_file(str(input_source))
            
        # Validate the sample
        if not self.validate_sample(sample):
            raise ValueError("Sample failed validation - does not meet filtering criteria")
        
        # Get ESM3 data
        esm3_data = self._get_esm3_data(sample, add_batch_dim=True)
        
        # Convert to tensors if requested
        if as_tensors:
            return self.convert_esm3_data_to_tensors(esm3_data)
        
        return esm3_data

    def get_sample_info(self, sample: ProteinStructureSample) -> Dict[str, Any]:
        """
        Get information about the protein sample.

        Args:
            sample: ProteinStructureSample to analyze

        Returns:
            Dictionary containing sample information
        """
        debug_print("Extracting sample information", "INFO")
        
        # Helper function to safely convert to numpy array
        def to_numpy(arr):
            """Convert JAX arrays or other array-like objects to numpy arrays."""
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                return arr
            # Handle JAX arrays
            if hasattr(arr, "to_py") and callable(getattr(arr, "to_py")):
                return arr.to_py()
            # Try normal numpy conversion for everything else
            return np.array(arr)
        
        # Helper function to safely convert to Python scalar
        def to_scalar(value):
            """Convert JAX scalar or numpy scalar to Python scalar."""
            if hasattr(value, "item"):
                return value.item()
            if hasattr(value, "to_py") and callable(getattr(value, "to_py")):
                return value.to_py()
            return value
        
        missing_coords_mask = to_numpy(sample.get_missing_backbone_coords_mask())
        num_valid_residues = np.sum(~missing_coords_mask)
        
        # Convert aatype to numpy array if needed
        aatype = to_numpy(sample.aatype)

        info = {
            "chain_id": sample.chain_id,
            "total_residues": to_scalar(sample.nb_residues),
            "valid_residues": int(to_scalar(num_valid_residues)),
            "missing_residues": int(to_scalar(np.sum(missing_coords_mask))),
            "resolution": to_scalar(sample.resolution),
            "pdb_cluster_size": to_scalar(sample.pdb_cluster_size),
            "sequence": "".join(
                ["ACDEFGHIKLMNPQRSTVWYX"[np.argmax(aa)] for aa in aatype]
            ),
        }
        
        debug_print(f"Sample info extracted: {len(info)} fields", "INFO", 1)
        return info

    def convert_esm3_data_to_tensors(self, esm3_data: Dict[str, np.ndarray], device: str = None) -> Dict[str, torch.Tensor]:
        """
        Convert ESM3 format numpy arrays to PyTorch tensors.
        
        Args:
            esm3_data: Dictionary with ESM3 data from _get_esm3_data
            device: PyTorch device to place tensors on (default: self.config["device"])
            
        Returns:
            Dictionary with the same keys but values converted to PyTorch tensors
        """
        if device is None:
            device = self.config["device"]
        
        result = {}
        for key, value in esm3_data.items():
            try:
                # Convert to numpy array first to handle JAX arrays
                if hasattr(value, "to_py") and callable(getattr(value, "to_py")):
                    # This is a JAX array, convert to numpy
                    value = value.to_py()
                elif not isinstance(value, np.ndarray):
                    value = np.array(value)
                
                # Then convert to PyTorch tensor
                if key == "sequence_id":
                    # Sequence IDs are typically integer indices
                    result[key] = torch.tensor(value, dtype=torch.long, device=device)
                elif key == "attention_mask":
                    # Attention mask is boolean
                    result[key] = torch.tensor(value, dtype=torch.bool, device=device)
                else:
                    # All other tensors (coords, residue_index) as float
                    result[key] = torch.tensor(value, dtype=self.config["dtype"], device=device)
            except Exception as e:
                debug_print(f"Error converting {key} to tensor: {str(e)}", "ERROR")
                debug_print(f"Value type: {type(value)}, shape: {getattr(value, 'shape', 'unknown')}", "ERROR", 1)
                raise
        
        return result
    

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
