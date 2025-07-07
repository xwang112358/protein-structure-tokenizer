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

import argparse
import os
import sys
import traceback
import pickle
from typing import Optional, Dict, Any, Union, List
import numpy as np
import yaml

# PyTorch and PyG imports
import torch
from torch_geometric.data import Data, Batch

# Add the current directory to the path to import the structure_tokenizer modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from structure_tokenizer.data.preprocessing import filter_out_sample
from structure_tokenizer.data.protein_structure_sample import (
    ProteinStructureSample,
    protein_structure_from_pdb_string,
)
# from structure_tokenizer.utils.log import get_logger  # DISABLED
from structure_tokenizer.types import BatchDataVQ3D

# Import preprocessing utilities
from structure_tokenizer.data import residue_constants
from structure_tokenizer.model import quat_affine
from structure_tokenizer.utils.protein_utils import (
    compute_nearest_neighbors_graph,
    protein_align_unbound_and_bound,
)
import jax.numpy as jnp
import jax

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


class PyGDataPipeline:
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
            debug_print(f"ERROR: {error_msg}", "LOAD")
            raise FileNotFoundError(error_msg)

        try:
            debug_print("Reading PDB file content...", "LOAD", 1)
            with open(pdb_file_path, "r") as f:
                pdb_string = f.read()

            debug_print(f"PDB content length: {len(pdb_string)} characters", "LOAD", 1)
            debug_print(f"Processing chain_id: {chain_id or self.config['chain_id'] or 'ALL'}", "LOAD", 1)
            
            sample = protein_structure_from_pdb_string(
                pdb_str=pdb_string, chain_id=chain_id or self.config["chain_id"]
            )

            debug_print(f"Successfully loaded protein with {sample.nb_residues} residues", "LOAD", 1)
            return sample

        except Exception as e:
            error_msg = f"Failed to load PDB file {pdb_file_path}: {str(e)}"
            debug_print(f"ERROR: {error_msg}", "LOAD")
            debug_print(f"Exception type: {type(e).__name__}", "LOAD", 1)
            raise Exception(error_msg) from e

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
        debug_print("Loading protein from PDB string", "LOAD")
        debug_print(f"PDB string length: {len(pdb_string)} characters", "LOAD", 1)

        try:
            debug_print(f"Processing chain_id: {chain_id or self.config['chain_id'] or 'ALL'}", "LOAD", 1)
            
            sample = protein_structure_from_pdb_string(
                pdb_str=pdb_string, chain_id=chain_id or self.config["chain_id"]
            )

            debug_print(f"Successfully loaded protein with {sample.nb_residues} residues", "LOAD", 1)
            return sample

        except Exception as e:
            error_msg = f"Failed to parse PDB string: {str(e)}"
            debug_print(f"ERROR: {error_msg}", "LOAD")
            debug_print(f"Exception type: {type(e).__name__}", "LOAD", 1)
            raise Exception(error_msg) from e

    def load_from_npy_file(self, npy_file_path: str) -> ProteinStructureSample:
        """
        Load protein structure from a saved numpy file.

        Args:
            npy_file_path: Path to the .npy file containing ProteinStructureSample data

        Returns:
            ProteinStructureSample object
        """
        debug_print(f"Loading protein from NPY file: {npy_file_path}", "LOAD")

        try:
            sample = ProteinStructureSample.from_file(npy_file_path)
            debug_print(f"Successfully loaded protein with {sample.nb_residues} residues", "LOAD", 1)
            return sample

        except Exception as e:
            error_msg = f"Failed to load .npy file {npy_file_path}: {str(e)}"
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
        
        debug_print(f"Total residues: {sample.nb_residues}", "VALIDATE", 1)
        debug_print(f"Min required: {self.config['min_number_valid_residues']}", "VALIDATE", 1)
        debug_print(f"Max allowed: {self.config['max_number_residues']}", "VALIDATE", 1)
        
        should_filter = filter_out_sample(
            sample=sample,
            min_number_valid_residues=self.config["min_number_valid_residues"],
            max_number_residues=self.config["max_number_residues"],
        )

        if should_filter:
            missing_coords_mask = sample.get_missing_backbone_coords_mask()
            num_valid_residues = np.sum(~missing_coords_mask)
            debug_print(f"VALIDATION FAILED:", "VALIDATE", 1)
            debug_print(f"  Valid residues: {num_valid_residues}", "VALIDATE", 2)
            debug_print(f"  Missing residues: {np.sum(missing_coords_mask)}", "VALIDATE", 2)
            debug_print(f"  Total residues: {sample.nb_residues}", "VALIDATE", 2)
            return False

        debug_print("Sample passed validation ✓", "VALIDATE", 1)
        return True

    def _compute_pyg_graph(
        self,
        sample: ProteinStructureSample,
        num_neighbor: int,
        downsampling_ratio: int,
        residue_loc_is_alphac: bool,
        crop_index: int,
        noise_level: float,
    ) -> Dict[str, Any]:
        """
        Compute graph representation compatible with PyTorch Geometric.
        
        This follows the same preprocessing logic as the original pipeline
        but outputs PyG-compatible tensors.
        """
        debug_print("Computing PyG graph representation", "GRAPH")
        
        # Building protein graph
        debug_print("Extracting atom coordinates and masks", "GRAPH", 1)
        atom37_coords = sample.atom37_positions
        atom37_mask = sample.atom37_gt_exists & sample.atom37_atom_exists
        missing_coords_residue_mask = sample.get_missing_backbone_coords_mask()
        num_residues_with_coords = np.sum(~missing_coords_residue_mask)
        
        debug_print(f"Initial residues with coordinates: {num_residues_with_coords}", "GRAPH", 1)

        # Local frames - BB atoms index
        debug_print("Computing local frames", "GRAPH", 1)
        (n_index, ca_index, c_index) = [
            residue_constants.atom_order[a] for a in ("N", "CA", "C")
        ]
        
        # Frame rotation
        (rot, _) = quat_affine.make_transform_from_reference(
            n_xyz=sample.atom37_positions[:, n_index, :],
            ca_xyz=sample.atom37_positions[:, ca_index, :],
            c_xyz=sample.atom37_positions[:, c_index, :],
        )
        
        # Get frame axis
        [u_i_feat, v_i_feat, n_i_feat] = np.split(rot, 3, axis=-1)
        u_i_feat = u_i_feat[..., 0]
        v_i_feat = v_i_feat[..., 0]
        n_i_feat = n_i_feat[..., 0]

        # Remove unobserved coordinates
        debug_print("Filtering out missing coordinates", "GRAPH", 1)
        (n_i_feat, u_i_feat, v_i_feat, atom37_coords, atom37_mask, aatype) = jax.tree_util.tree_map(
            lambda x: x[~missing_coords_residue_mask],
            (n_i_feat, u_i_feat, v_i_feat, atom37_coords, atom37_mask, sample.aatype),
        )

        # Crop to maximum length
        debug_print(f"Cropping to max length {crop_index}", "GRAPH", 1)
        crop_start_idx = (
            0
            if num_residues_with_coords <= crop_index
            else np.random.randint(0, num_residues_with_coords - crop_index)
        )
        
        debug_print(f"Crop start index: {crop_start_idx}", "GRAPH", 2)

        (n_i_feat, u_i_feat, v_i_feat, atom37_coords, atom37_mask, aatype) = jax.tree_util.tree_map(
            lambda x: x[crop_start_idx : crop_start_idx + crop_index],
            (n_i_feat, u_i_feat, v_i_feat, atom37_coords, atom37_mask, aatype),
        )

        # Residue representative locations
        debug_print("Computing residue representative locations", "GRAPH", 1)
        res_representatives_loc_feat = (
            atom37_coords[:, ca_index]
            if residue_loc_is_alphac
            else np.mean(atom37_coords, axis=1, where=atom37_mask)
        )

        # Align structures if needed
        if not residue_loc_is_alphac:
            debug_print("Aligning structures", "GRAPH", 1)
            (
                res_representatives_loc_feat,
                n_i_feat,
                u_i_feat,
                v_i_feat,
            ) = protein_align_unbound_and_bound(
                stacked_residue_representatives_coordinates=res_representatives_loc_feat,
                protein_n_i_feat=n_i_feat,
                protein_u_i_feat=u_i_feat,
                protein_v_i_feat=v_i_feat,
                alphac_atom_coordinates=atom37_coords[:, ca_index],
            )

        # Build the k-NN graph
        num_residues_with_coords = np.minimum(num_residues_with_coords, crop_index)
        n_neighbor = num_residues_with_coords if num_neighbor == -1 else num_neighbor
        
        debug_print(f"Building k-NN graph with {n_neighbor} neighbors", "GRAPH", 1)
        debug_print(f"Final number of residues: {num_residues_with_coords}", "GRAPH", 2)
        
        list_atom_coordinates = [
            atom37_coords[i, atom37_mask[i]] for i in range(num_residues_with_coords)
        ]

        (
            n_node,
            n_edge,
            nodes_x,
            edges_features,
            senders,
            receivers,
        ) = compute_nearest_neighbors_graph(
            protein_num_residues=num_residues_with_coords,
            list_atom_coordinates=list_atom_coordinates,
            stacked_residue_coordinates=res_representatives_loc_feat,
            protein_n_i_feat=n_i_feat,
            protein_u_i_feat=u_i_feat,
            protein_v_i_feat=v_i_feat,
            num_neighbor=n_neighbor,
            noise_level=noise_level,
        )
        
        debug_print(f"Graph computed: {n_node} nodes, {n_edge} edges", "GRAPH", 1)

        return {
            "n_node": n_node,
            "n_edge": n_edge,
            "nodes_x": nodes_x,
            "edges_features": edges_features,
            "senders": senders,
            "receivers": receivers,
            "aatype": aatype,
            "atom37_coords": atom37_coords,
            "atom37_mask": atom37_mask,
            "crop_start_idx": crop_start_idx,
            "downsampling_ratio": downsampling_ratio,
        }

    def preprocess_to_pyg(self, sample: ProteinStructureSample) -> Data:
        """
        Preprocess the protein sample for PyTorch Geometric format.

        Args:
            sample: ProteinStructureSample to preprocess

        Returns:
            torch_geometric.data.Data object ready for GNN models
        """
        debug_print("Starting PyG preprocessing", "PREPROCESS")

        try:
            # Get graph data
            debug_print("Computing graph data...", "PREPROCESS", 1)
            graph_data = self._compute_pyg_graph(
                sample=sample,
                num_neighbor=self.config["num_neighbor"],
                downsampling_ratio=self.config["downsampling_ratio"],
                residue_loc_is_alphac=self.config["residue_loc_is_alphac"],
                crop_index=self.config["crop_index"],
                noise_level=self.config["noise_level"],
            )

            # Extract data
            debug_print("Converting to PyTorch tensors...", "PREPROCESS", 1)
            n_node = graph_data["n_node"]
            n_edge = graph_data["n_edge"]
            nodes_x = graph_data["nodes_x"]
            edges_features = graph_data["edges_features"]
            senders = graph_data["senders"]
            receivers = graph_data["receivers"]
            aatype = graph_data["aatype"]

            # Convert to PyTorch tensors
            dtype = self.config["dtype"]
            device = self.config["device"]
            
            debug_print(f"Target dtype: {dtype}, device: {device}", "PREPROCESS", 2)
            
            # Debug JAX array shapes and types before conversion
            debug_print(f"JAX array shapes - aatype: {aatype.shape if hasattr(aatype, 'shape') else 'no shape'}, type: {type(aatype)}", "PREPROCESS", 2)
            debug_print(f"nodes_x: {nodes_x.shape if hasattr(nodes_x, 'shape') else 'no shape'}, type: {type(nodes_x)}", "PREPROCESS", 2)

            # Node features and positions
            x = torch.tensor(np.array(nodes_x), dtype=dtype, device=device)  # [num_nodes, 3]
            pos = torch.tensor(np.array(nodes_x), dtype=dtype, device=device)  # [num_nodes, 3]
            
            # Edge connectivity in PyG format [2, num_edges]
            edge_index = torch.stack([
                torch.tensor(np.array(senders), dtype=torch.long, device=device),
                torch.tensor(np.array(receivers), dtype=torch.long, device=device)
            ], dim=0)

            # Edge features [num_edges, edge_feature_dim]
            edge_attr = torch.tensor(np.array(edges_features), dtype=dtype, device=device)

            # Sequence information - ensure aatype is properly converted to numpy first
            aatype_np = np.array(aatype)
            debug_print(f"aatype_np shape: {aatype_np.shape}, dtype: {aatype_np.dtype}", "PREPROCESS", 2)
            
            # Ensure aatype has proper dimensions
            if aatype_np.ndim == 0:
                # If scalar, expand to match number of nodes
                debug_print("aatype is scalar, expanding to match node count", "PREPROCESS", 2)
                aatype_np = np.full(n_node, aatype_np.item(), dtype=np.int32)
            elif aatype_np.ndim == 2:
                # If 2D (one-hot encoded), convert to indices
                debug_print("aatype is 2D (one-hot), converting to indices", "PREPROCESS", 2)
                aatype_np = np.argmax(aatype_np, axis=-1)
            
            sequence = torch.tensor(aatype_np, dtype=torch.long, device=device)
            
            # Node mask (indicating real vs padded nodes)
            node_mask = torch.ones(n_node, dtype=torch.bool, device=device)
            
            # Token mask for downsampling
            token_num = int(n_node / graph_data["downsampling_ratio"])
            tokens_mask = torch.ones(token_num, dtype=torch.bool, device=device)

            debug_print(f"Tensor shapes - x: {x.shape}, edge_index: {edge_index.shape}, edge_attr: {edge_attr.shape}", "PREPROCESS", 2)

            # Create PyG Data object
            debug_print("Creating PyG Data object...", "PREPROCESS", 1)
            pyg_data = Data(
                # Core graph structure
                x=x,  # Node features (coordinates)
                edge_index=edge_index,  # Edge connectivity [2, num_edges]
                edge_attr=edge_attr,  # Edge features
                pos=pos,  # Node positions (same as x for now)
                
                # Sequence and mask information
                sequence=sequence,  # Amino acid types [num_nodes]
                node_mask=node_mask,  # Valid node indicator [num_nodes]
                tokens_mask=tokens_mask,  # Token mask for downsampling
                
                # Graph metadata
                num_nodes=n_node,
                num_edges=n_edge,
            )

            # Add global/protein-level features if requested
            if self.config["include_global_features"]:
                debug_print("Adding global protein features...", "PREPROCESS", 1)
                # Build protein features for structure module
                features = sample.make_protein_features()
                
                # Remove unobserved coordinates
                missing_mask = sample.get_missing_backbone_coords_mask()
                features = jax.tree_util.tree_map(
                    lambda x: x[~missing_mask], features
                )
                
                # Crop features
                crop_start_idx = graph_data["crop_start_idx"]
                crop_index = self.config["crop_index"]
                features = jax.tree_util.tree_map(
                    lambda x: x[crop_start_idx : crop_start_idx + crop_index], features
                )
                
                # Convert features to tensors and add to data
                protein_features = {}
                for key, value in features.items():
                    if isinstance(value, (np.ndarray, jnp.ndarray)):
                        protein_features[key] = torch.tensor(
                            np.array(value), dtype=dtype, device=device
                        )
                    else:
                        protein_features[key] = value
                
                protein_features["nb_residues"] = n_node
                pyg_data.protein_features = protein_features
                debug_print(f"Added {len(protein_features)} protein features", "PREPROCESS", 2)

            # Add metadata
            pyg_data.chain_id = sample.chain_id
            pyg_data.resolution = sample.resolution
            pyg_data.pdb_cluster_size = sample.pdb_cluster_size

            debug_print("PyG preprocessing completed successfully ✓", "PREPROCESS", 1)
            return pyg_data

        except Exception as e:
            error_msg = f"PyG preprocessing failed: {str(e)}"
            debug_print(f"ERROR: {error_msg}", "PREPROCESS")
            debug_print(f"Exception type: {type(e).__name__}", "PREPROCESS", 1)
            raise Exception(error_msg) from e

    def save_pyg_data(
        self,
        data: Data,
        output_path: str,
        save_raw_sample: bool = False,
        raw_sample: Optional[ProteinStructureSample] = None,
    ):
        """
        Save the PyG data to file.

        Args:
            data: PyTorch Geometric Data object
            output_path: Path to save the output
            save_raw_sample: Whether to also save the raw ProteinStructureSample
            raw_sample: Raw ProteinStructureSample (required if save_raw_sample=True)
        """
        debug_print(f"Saving PyG data to: {output_path}", "SAVE")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
        if not os.path.exists(output_dir):
            debug_print(f"Creating output directory: {output_dir}", "SAVE", 1)
            os.makedirs(output_dir, exist_ok=True)

        try:
            debug_print(f"Output format: {self.config['output_format']}", "SAVE", 1)
            
            if self.config["output_format"] == "pkl":
                # Save as pickle file
                with open(output_path, "wb") as f:
                    pickle.dump(data, f)
            else:  # Default to .pt format
                torch.save(data, output_path)

            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            debug_print(f"Successfully saved PyG data ({file_size:.2f} MB) ✓", "SAVE", 1)

            # Save raw sample if requested
            if save_raw_sample and raw_sample is not None:
                raw_output_path = output_path.replace(".pt", "_raw.npy").replace(
                    ".pkl", "_raw.npy"
                )
                debug_print(f"Saving raw sample to: {raw_output_path}", "SAVE", 1)
                raw_sample.to_file(raw_output_path)
                debug_print("Raw sample saved ✓", "SAVE", 2)

        except Exception as e:
            error_msg = f"Failed to save PyG data to {output_path}: {str(e)}"
            debug_print(f"ERROR: {error_msg}", "SAVE")
            debug_print(f"Exception type: {type(e).__name__}", "SAVE", 1)
            raise Exception(error_msg) from e

    def process_single(
        self,
        input_source: Union[str, ProteinStructureSample],
        output_path: Optional[str] = None,
        input_type: str = "auto",
    ) -> Data:
        """
        Process a single protein input through the complete PyG pipeline.

        Args:
            input_source: Path to PDB file, PDB string, path to .npy file, or ProteinStructureSample
            output_path: Path to save the processed output (optional)
            input_type: Type of input ('pdb_file', 'pdb_string', 'npy_file', 'sample', or 'auto')

        Returns:
            PyTorch Geometric Data object
        """
        debug_print("=== Starting PyG pipeline for single protein ===", "PIPELINE")

        # Load the sample based on input type
        debug_print(f"Input type: {input_type}", "PIPELINE", 1)
        
        if isinstance(input_source, ProteinStructureSample):
            debug_print("Using provided ProteinStructureSample", "PIPELINE", 1)
            sample = input_source
        elif input_type == "auto":
            debug_print("Auto-detecting input type...", "PIPELINE", 1)
            if os.path.exists(str(input_source)):
                if str(input_source).endswith(".npy"):
                    debug_print("Detected: NPY file", "PIPELINE", 2)
                    sample = self.load_from_npy_file(str(input_source))
                else:
                    debug_print("Detected: PDB file", "PIPELINE", 2)
                    sample = self.load_from_pdb_file(str(input_source))
            else:
                debug_print("Detected: PDB string", "PIPELINE", 2)
                sample = self.load_from_pdb_string(str(input_source))
        elif input_type == "pdb_file":
            sample = self.load_from_pdb_file(str(input_source))
        elif input_type == "pdb_string":
            sample = self.load_from_pdb_string(str(input_source))
        elif input_type == "npy_file":
            sample = self.load_from_npy_file(str(input_source))
        else:
            error_msg = f"Unknown input_type: {input_type}"
            debug_print(f"ERROR: {error_msg}", "PIPELINE")
            raise ValueError(error_msg)

        # Validate the sample
        debug_print("Validating sample...", "PIPELINE", 1)
        if not self.validate_sample(sample):
            error_msg = "Sample failed validation - does not meet filtering criteria"
            debug_print(f"ERROR: {error_msg}", "PIPELINE")
            raise ValueError(error_msg)

        # Preprocess to PyG format
        debug_print("Starting preprocessing...", "PIPELINE", 1)
        pyg_data = self.preprocess_to_pyg(sample)

        # Save the output if path is provided
        if output_path:
            debug_print("Saving output...", "PIPELINE", 1)
            self.save_pyg_data(
                pyg_data,
                output_path,
                save_raw_sample=self.config.get("save_intermediate", False),
                raw_sample=sample,
            )

        debug_print("=== PyG pipeline completed successfully ===", "PIPELINE")
        return pyg_data

    def process_batch(
        self,
        input_sources: List[Union[str, ProteinStructureSample]],
        output_dir: Optional[str] = None,
        input_type: str = "auto",
    ) -> Batch:
        """
        Process multiple proteins and create a PyG batch.

        Args:
            input_sources: List of input sources
            output_dir: Directory to save individual outputs (optional)
            input_type: Type of inputs

        Returns:
            PyTorch Geometric Batch object
        """
        debug_print(f"=== Starting batch processing of {len(input_sources)} proteins ===", "BATCH")

        pyg_data_list = []
        failed_count = 0
        
        for i, input_source in enumerate(input_sources):
            try:
                progress_indicator(i + 1, len(input_sources), "Batch Processing")
                
                debug_print(f"Processing protein {i+1}/{len(input_sources)}: {input_source}", "BATCH", 1)
                
                output_path = None
                if output_dir:
                    filename = f"protein_{i:04d}.pt"
                    output_path = os.path.join(output_dir, filename)

                pyg_data = self.process_single(
                    input_source=input_source,
                    output_path=output_path,
                    input_type=input_type,
                )
                pyg_data_list.append(pyg_data)
                debug_print(f"Protein {i+1} completed ✓", "BATCH", 2)

            except Exception as e:
                failed_count += 1
                debug_print(f"WARNING: Failed to process protein {i+1}: {str(e)}", "BATCH", 1)
                debug_print(f"Continuing with remaining proteins...", "BATCH", 2)
                continue

        if not pyg_data_list:
            error_msg = "No proteins were successfully processed"
            debug_print(f"ERROR: {error_msg}", "BATCH")
            raise ValueError(error_msg)

        # Create batch
        debug_print("Creating PyG batch...", "BATCH", 1)
        batch = Batch.from_data_list(pyg_data_list)
        
        debug_print(f"=== Batch processing completed ===", "BATCH")
        debug_print(f"  Successfully processed: {len(pyg_data_list)}", "BATCH", 1)
        debug_print(f"  Failed: {failed_count}", "BATCH", 1)
        debug_print(f"  Total nodes in batch: {batch.num_nodes}", "BATCH", 1)
        debug_print(f"  Total edges in batch: {batch.num_edges}", "BATCH", 1)

        return batch

    def get_sample_info(self, sample: ProteinStructureSample) -> Dict[str, Any]:
        """
        Get information about the protein sample.

        Args:
            sample: ProteinStructureSample to analyze

        Returns:
            Dictionary containing sample information
        """
        debug_print("Extracting sample information", "INFO")
        
        missing_coords_mask = sample.get_missing_backbone_coords_mask()
        num_valid_residues = np.sum(~missing_coords_mask)

        info = {
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
        
        debug_print(f"Sample info extracted: {len(info)} fields", "INFO", 1)
        return info


def convert_batch_data_to_pyg(
    batch_data_vq3d: BatchDataVQ3D, 
    config: Optional[Dict[str, Any]] = None
) -> Data:
    """
    Convert existing BatchDataVQ3D format to PyTorch Geometric Data.
    
    This function provides backward compatibility for existing preprocessed data.
    
    Args:
        batch_data_vq3d: Existing BatchDataVQ3D object
        config: Optional configuration for conversion
        
    Returns:
        PyTorch Geometric Data object
    """
    if config is None:
        config = {"dtype": torch.float32, "device": "cpu"}
    
    dtype = config.get("dtype", torch.float32)
    device = config.get("device", "cpu")
    
    graph = batch_data_vq3d.graph
    features = batch_data_vq3d.features
    
    # Convert core graph data
    x = torch.tensor(np.array(graph.node_features), dtype=dtype, device=device)
    pos = torch.tensor(np.array(graph.nodes_original_coordinates), dtype=dtype, device=device)
    
    # Edge connectivity in PyG format
    edge_index = torch.stack([
        torch.tensor(np.array(graph.senders), dtype=torch.long, device=device),
        torch.tensor(np.array(graph.receivers), dtype=torch.long, device=device)
    ], dim=0)
    
    edge_attr = torch.tensor(np.array(graph.edge_features), dtype=dtype, device=device)
    
    # Masks
    node_mask = torch.tensor(
        np.array(graph.nodes_mask).squeeze(), dtype=torch.bool, device=device
    )
    tokens_mask = torch.tensor(
        np.array(graph.tokens_mask).squeeze(), dtype=torch.bool, device=device
    )
    
    # Create Data object
    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        node_mask=node_mask,
        tokens_mask=tokens_mask,
        num_nodes=int(graph.n_node.item()),
        num_edges=int(graph.n_edge.item()),
    )
    
    # Add protein features
    protein_features = {}
    for key, value in features.items():
        if isinstance(value, (np.ndarray, jnp.ndarray)):
            protein_features[key] = torch.tensor(
                np.array(value), dtype=dtype, device=device
            )
        else:
            protein_features[key] = value
    
    pyg_data.protein_features = protein_features
    
    return pyg_data


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main function for command-line interface."""
    global DEBUG_MODE, OUTPUT_FILE
    
    parser = argparse.ArgumentParser(
        description="Protein Structure PyTorch Geometric Data Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a PDB file to PyG format
    python pyg_data_pipeline.py --input_path protein.pdb --output_path processed.pt
    
    # Process with custom config
    python pyg_data_pipeline.py --input_path protein.pdb --config config.yaml --output_path processed.pt
    
    # Process a specific chain
    python pyg_data_pipeline.py --input_path protein.pdb --chain_id A --output_path processed.pt
    
    # Get sample information only
    python pyg_data_pipeline.py --input_path protein.pdb --info_only
    
    # Process multiple files in a directory
    python pyg_data_pipeline.py --input_dir path/to/pdbs --output_dir path/to/outputs
    
    # Debug mode with output to file
    python pyg_data_pipeline.py --input_path protein.pdb --output_path processed.pt --debug --log_file debug.log
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

    # Set debug mode
    if args.quiet:
        DEBUG_MODE = False
    elif args.debug:
        DEBUG_MODE = True
    
    # Set output file for logging
    if args.log_file:
        OUTPUT_FILE = args.log_file
        # Clear the log file
        with open(OUTPUT_FILE, 'w') as f:
            f.write("=== PyG Data Pipeline Debug Log ===\n\n")
        debug_print(f"Debug output will be saved to: {OUTPUT_FILE}", "MAIN")

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
            import glob
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
        if DEBUG_MODE:
            print("\nFull traceback:")
            traceback.print_exc()
        print("="*50)
        sys.exit(1)


if __name__ == "__main__":
    main()
