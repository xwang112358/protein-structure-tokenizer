#!/usr/bin/env python3
"""
Test script for PyTorch Geometric Data Pipeline

This script runs comprehensive tests to validate the PyG pipeline functionality
and ensure compatibility with the original pipeline.
"""

import os
import sys
import tempfile
import numpy as np
import torch

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyg_data_pipeline import PyGDataPipeline, convert_batch_data_to_pyg
from data_pipeline import DataPipeline


def test_pipeline_initialization():
    """Test pipeline initialization with various configurations."""
    print("Testing pipeline initialization...")
    
    # Test default initialization
    pipeline = PyGDataPipeline()
    assert pipeline.config["num_neighbor"] == 30
    assert pipeline.config["output_format"] == "pt"
    print("✓ Default initialization")
    
    # Test custom configuration
    custom_config = {
        "num_neighbor": 15,
        "crop_index": 256,
        "dtype": torch.float64,
        "device": "cpu"
    }
    pipeline = PyGDataPipeline(custom_config)
    assert pipeline.config["num_neighbor"] == 15
    assert pipeline.config["crop_index"] == 256
    print("✓ Custom configuration")


def test_pdb_loading():
    """Test loading PDB files and strings."""
    print("Testing PDB loading...")
    
    pipeline = PyGDataPipeline()
    
    # Test PDB file loading
    test_pdb_files = ["casp14_pdbs/T1024.pdb", "casp14_pdbs/T1025.pdb"]
    for pdb_file in test_pdb_files:
        if os.path.exists(pdb_file):
            try:
                sample = pipeline.load_from_pdb_file(pdb_file)
                assert sample.nb_residues > 0
                print(f"✓ Loaded {pdb_file} with {sample.nb_residues} residues")
                break
            except Exception as e:
                print(f"✗ Failed to load {pdb_file}: {e}")
        else:
            print(f"! PDB file {pdb_file} not found")
    
    # Test PDB string loading (using a minimal example)
    minimal_pdb = """ATOM      1  N   ALA A   1      -8.901   4.127  -0.555  1.00 11.99           N  
ATOM      2  CA  ALA A   1      -8.608   3.135  -1.618  1.00 11.99           C  
ATOM      3  C   ALA A   1      -7.221   2.458  -1.497  1.00 11.99           C  
ATOM      4  O   ALA A   1      -6.632   2.415  -0.421  1.00 11.99           O  
ATOM      5  CB  ALA A   1      -9.722   2.083  -1.710  1.00 11.99           C  
END"""
    
    try:
        sample = pipeline.load_from_pdb_string(minimal_pdb)
        assert sample.nb_residues > 0
        print(f"✓ Loaded PDB string with {sample.nb_residues} residues")
    except Exception as e:
        print(f"✗ Failed to load PDB string: {e}")


def test_pyg_preprocessing():
    """Test PyG-specific preprocessing."""
    print("Testing PyG preprocessing...")
    
    pipeline = PyGDataPipeline({
        "num_neighbor": 10,
        "crop_index": 64,
        "include_global_features": True
    })
    
    # Find a test PDB file
    test_pdb = None
    for pdb_file in ["casp14_pdbs/T1024.pdb", "casp14_pdbs/T1025.pdb"]:
        if os.path.exists(pdb_file):
            test_pdb = pdb_file
            break
    
    if test_pdb is None:
        print("! No test PDB files found, skipping PyG preprocessing test")
        return
    
    try:
        # Load and process
        sample = pipeline.load_from_pdb_file(test_pdb)
        pyg_data = pipeline.preprocess_to_pyg(sample)
        
        # Validate PyG data structure
        assert hasattr(pyg_data, 'x'), "Missing node features (x)"
        assert hasattr(pyg_data, 'edge_index'), "Missing edge_index"
        assert hasattr(pyg_data, 'edge_attr'), "Missing edge attributes"
        assert hasattr(pyg_data, 'pos'), "Missing node positions"
        assert hasattr(pyg_data, 'num_nodes'), "Missing num_nodes"
        assert hasattr(pyg_data, 'num_edges'), "Missing num_edges"
        
        # Validate tensor shapes
        assert pyg_data.x.shape[0] == pyg_data.num_nodes, "Node features shape mismatch"
        assert pyg_data.pos.shape == (pyg_data.num_nodes, 3), "Position shape mismatch"
        assert pyg_data.edge_index.shape[0] == 2, "Edge index should be [2, num_edges]"
        assert pyg_data.edge_index.shape[1] == pyg_data.num_edges, "Edge index shape mismatch"
        assert pyg_data.edge_attr.shape[0] == pyg_data.num_edges, "Edge attributes shape mismatch"
        
        # Validate data types
        assert pyg_data.x.dtype == torch.float32, "Node features should be float32"
        assert pyg_data.edge_index.dtype == torch.long, "Edge index should be long"
        
        print("✓ PyG preprocessing successful")
        print(f"  Nodes: {pyg_data.num_nodes}, Edges: {pyg_data.num_edges}")
        print(f"  Node features shape: {pyg_data.x.shape}")
        print(f"  Edge features shape: {pyg_data.edge_attr.shape}")
        
    except Exception as e:
        print(f"✗ PyG preprocessing failed: {e}")
        raise


def test_format_conversion():
    """Test conversion between original and PyG formats."""
    print("Testing format conversion...")
    
    # Find a test PDB file
    test_pdb = None
    for pdb_file in ["casp14_pdbs/T1024.pdb", "casp14_pdbs/T1025.pdb"]:
        if os.path.exists(pdb_file):
            test_pdb = pdb_file
            break
    
    if test_pdb is None:
        print("! No test PDB files found, skipping conversion test")
        return
    
    try:
        # Process with original pipeline
        original_pipeline = DataPipeline({
            "num_neighbor": 15,
            "crop_index": 128
        })
        original_data = original_pipeline.process_single(test_pdb)
        
        # Convert to PyG format
        pyg_data = convert_batch_data_to_pyg(original_data)
        
        # Validate conversion
        original_nodes = int(original_data.graph.n_node.item())
        original_edges = int(original_data.graph.n_edge.item())
        
        assert pyg_data.num_nodes == original_nodes, "Node count mismatch in conversion"
        assert pyg_data.num_edges == original_edges, "Edge count mismatch in conversion"
        
        # Check coordinate preservation
        original_coords = np.array(original_data.graph.nodes_original_coordinates)
        pyg_coords = pyg_data.pos.cpu().numpy()
        
        # Only compare valid nodes (excluding padding)
        nodes_mask = np.array(original_data.graph.nodes_mask).squeeze()
        valid_nodes = np.sum(nodes_mask[:original_nodes])
        
        coord_match = np.allclose(
            original_coords[:valid_nodes], 
            pyg_coords[:valid_nodes], 
            atol=1e-5
        )
        assert coord_match, "Coordinates not preserved in conversion"
        
        print("✓ Format conversion successful")
        print(f"  Original nodes: {original_nodes}, PyG nodes: {pyg_data.num_nodes}")
        print(f"  Original edges: {original_edges}, PyG edges: {pyg_data.num_edges}")
        print(f"  Coordinate preservation: {coord_match}")
        
    except Exception as e:
        print(f"✗ Format conversion failed: {e}")
        raise


def test_file_operations():
    """Test saving and loading PyG data."""
    print("Testing file operations...")
    
    pipeline = PyGDataPipeline({
        "output_format": "pt",
        "crop_index": 64
    })
    
    # Find a test PDB file
    test_pdb = None
    for pdb_file in ["casp14_pdbs/T1024.pdb", "casp14_pdbs/T1025.pdb"]:
        if os.path.exists(pdb_file):
            test_pdb = pdb_file
            break
    
    if test_pdb is None:
        print("! No test PDB files found, skipping file operations test")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            output_path = os.path.join(temp_dir, "test_output.pt")
            
            # Process and save
            pyg_data = pipeline.process_single(
                input_source=test_pdb,
                output_path=output_path
            )
            
            # Verify file was created
            assert os.path.exists(output_path), "Output file was not created"
            
            # Load and verify
            loaded_data = torch.load(output_path)
            
            assert loaded_data.num_nodes == pyg_data.num_nodes, "Loaded data node count mismatch"
            assert loaded_data.num_edges == pyg_data.num_edges, "Loaded data edge count mismatch"
            assert torch.allclose(loaded_data.x, pyg_data.x), "Loaded data features mismatch"
            
            print("✓ File operations successful")
            print(f"  Saved to: {output_path}")
            print(f"  File size: {os.path.getsize(output_path)} bytes")
            
        except Exception as e:
            print(f"✗ File operations failed: {e}")
            raise


def test_batch_processing():
    """Test batch processing functionality."""
    print("Testing batch processing...")
    
    # Find multiple test PDB files
    import glob
    test_pdbs = glob.glob("casp14_pdbs/*.pdb")[:2]  # Take first 2
    
    if len(test_pdbs) < 2:
        print("! Need at least 2 PDB files for batch testing, skipping")
        return
    
    pipeline = PyGDataPipeline({
        "num_neighbor": 10,
        "crop_index": 64,
        "include_global_features": False
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Process batch
            batch_data = pipeline.process_batch(
                input_sources=test_pdbs,
                output_dir=temp_dir,
                input_type="pdb_file"
            )
            
            # Validate batch
            assert hasattr(batch_data, 'batch'), "Batch missing batch attribute"
            assert len(batch_data) == len(test_pdbs), "Batch size mismatch"
            
            # Check individual files were created
            for i in range(len(test_pdbs)):
                file_path = os.path.join(temp_dir, f"protein_{i:04d}.pt")
                assert os.path.exists(file_path), f"Individual file {file_path} not created"
            
            print("✓ Batch processing successful")
            print(f"  Processed {len(test_pdbs)} proteins")
            print(f"  Total nodes: {batch_data.num_nodes}")
            print(f"  Total edges: {batch_data.num_edges}")
            
        except Exception as e:
            print(f"✗ Batch processing failed: {e}")
            raise


def test_validation():
    """Test sample validation functionality."""
    print("Testing sample validation...")
    
    pipeline = PyGDataPipeline({
        "min_number_valid_residues": 5,
        "max_number_residues": 1000
    })
    
    # Test with minimal PDB that should pass validation
    minimal_pdb = """ATOM      1  N   ALA A   1      -8.901   4.127  -0.555  1.00 11.99           N  
ATOM      2  CA  ALA A   1      -8.608   3.135  -1.618  1.00 11.99           C  
ATOM      3  C   ALA A   1      -7.221   2.458  -1.497  1.00 11.99           C  
ATOM      4  O   ALA A   1      -6.632   2.415  -0.421  1.00 11.99           O  
ATOM      5  CB  ALA A   1      -9.722   2.083  -1.710  1.00 11.99           C  
ATOM      6  N   VAL A   2      -6.823   1.854  -2.618  1.00 11.99           N  
ATOM      7  CA  VAL A   2      -5.487   1.159  -2.697  1.00 11.99           C  
ATOM      8  C   VAL A   2      -5.581  -0.337  -2.378  1.00 11.99           C  
ATOM      9  O   VAL A   2      -6.449  -0.793  -1.644  1.00 11.99           O  
ATOM     10  CB  VAL A   2      -4.837   1.373  -4.075  1.00 11.99           C  
END"""
    
    try:
        sample = pipeline.load_from_pdb_string(minimal_pdb)
        is_valid = pipeline.validate_sample(sample)
        assert is_valid, "Sample should pass validation"
        print(f"✓ Validation passed for sample with {sample.nb_residues} residues")
        
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        raise


def test_device_compatibility():
    """Test device compatibility (CPU/GPU)."""
    print("Testing device compatibility...")
    
    # Test CPU
    cpu_pipeline = PyGDataPipeline({"device": "cpu", "crop_index": 32})
    
    # Test with minimal data
    minimal_pdb = """ATOM      1  N   ALA A   1      -8.901   4.127  -0.555  1.00 11.99           N  
ATOM      2  CA  ALA A   1      -8.608   3.135  -1.618  1.00 11.99           C  
ATOM      3  C   ALA A   1      -7.221   2.458  -1.497  1.00 11.99           C  
ATOM      4  O   ALA A   1      -6.632   2.415  -0.421  1.00 11.99           O  
ATOM      5  CB  ALA A   1      -9.722   2.083  -1.710  1.00 11.99           C  
END"""
    
    try:
        sample = cpu_pipeline.load_from_pdb_string(minimal_pdb)
        pyg_data = cpu_pipeline.preprocess_to_pyg(sample)
        
        assert pyg_data.x.device.type == "cpu", "Data should be on CPU"
        print("✓ CPU device compatibility")
        
        # Test GPU if available
        if torch.cuda.is_available():
            gpu_pipeline = PyGDataPipeline({"device": "cuda", "crop_index": 32})
            gpu_data = gpu_pipeline.preprocess_to_pyg(sample)
            assert gpu_data.x.device.type == "cuda", "Data should be on GPU"
            print("✓ GPU device compatibility")
        else:
            print("! GPU not available, skipping GPU test")
            
    except Exception as e:
        print(f"✗ Device compatibility test failed: {e}")
        raise


def run_all_tests():
    """Run all tests and report results."""
    print("Running PyTorch Geometric Data Pipeline Tests")
    print("=" * 60)
    
    tests = [
        test_pipeline_initialization,
        test_pdb_loading,
        test_validation,
        test_pyg_preprocessing,
        test_format_conversion,
        test_file_operations,
        test_batch_processing,
        test_device_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n[{test.__name__}]")
            test()
            passed += 1
            print(f"✓ {test.__name__} PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
