# PDB Batch Processing Examples

This directory contains example scripts that demonstrate how to use the PyG Data Pipeline to process multiple PDB files in batch from any directory and return a list of PyTorch Geometric data objects.

## Files

### 1. `test_batch_processing.py`
A simple, focused test script that demonstrates the basic usage of the `process_batch` method.

**Features:**
- Processes all PDB files from a specified directory (defaults to `casp14_pdbs`)
- Returns a list of PyG Data objects
- Includes verification of data structure integrity
- Shows how to use the returned data objects

**Usage:**
```bash
# Use default casp14_pdbs directory
python test_batch_processing.py

# Specify a custom PDB directory
python test_batch_processing.py /path/to/your/pdb/folder
```

### 2. `pdb_batch_processing.py`
A comprehensive example with command-line interface and multiple usage patterns.

**Features:**
- Command-line interface with flexible options
- Multiple examples showing different configuration options
- Detailed analysis of batch processing s
- Options for saving individual files and batch files
- Demonstrates subset processing
- Statistical analysis of the processed data

**Usage:**
```bash
# Basic usage with CASP14 files
python pdb_batch_processing.py --pdb_folder casp14_pdbs

# With saving and custom output directory
python pdb_batch_processing.py --pdb_folder casp14_pdbs --save_individual --save_batch --output_dir my_outputs

# With custom configuration
python pdb_batch_processing.py --pdb_folder casp14_pdbs --crop_index 256 --num_neighbor 20

# Quiet mode (less verbose)
python pdb_batch_processing.py --pdb_folder casp14_pdbs --quiet

# Any PDB directory
python pdb_batch_processing.py --pdb_folder /path/to/your/pdb/files
```

## Key Features Demonstrated

### Flexible Directory Input
Both scripts now accept any directory containing PDB files:
1. Specify the directory path as an argument
2. Automatically find all `.pdb` files in the directory
3. Process them in batch using `process_batch()`
4. Extract individual Data objects using `to_data_list()`

### Basic Usage Pattern
```python
from pyg_data_pipeline import PyGDataPipeline

# Initialize pipeline
pipeline = PyGDataPipeline(config)

# Process all PDB files in a directory
pdb_files = glob.glob(os.path.join(pdb_folder_path, "*.pdb"))
batch_data = pipeline.process_batch(input_sources=pdb_files, input_type="pdb_file")

# Get list of individual PyG data objects
data_list = batch_data.to_data_list()  # This is what you want!
```

### Data Structure Verification
The scripts verify that each returned Data object has:
- Node features (`x`)
- Edge connectivity (`edge_index`)
- Edge features (`edge_attr`)
- Correct node and edge counts
- Optional additional attributes (sequence, protein_features, etc.)

### Usage Patterns
Examples of how to:
- Access individual protein data
- Create new batches from subsets
- Analyze dataset statistics
- Iterate through all proteins
- Check available attributes

## Expected Output

When you run the scripts, you should see:
1. List of PDB files being processed from the specified directory
2. Progress information during batch processing
3. Summary statistics (total nodes, edges, etc.)
4. Individual protein details (nodes, edges per protein)
5. Data structure verification results
6. Usage demonstrations

## Configuration Options

The scripts demonstrate various configuration options:
- `num_neighbor`: Number of nearest neighbors for graph construction
- `crop_index`: Maximum sequence length after cropping
- `noise_level`: Noise level for data augmentation
- `min_number_valid_residues`: Minimum valid residues required
- `max_number_residues`: Maximum residues allowed
- `dtype`: Data type for tensors (float32/float64)
- `device`: Device for tensor creation (cpu/cuda)
- `include_global_features`: Whether to include protein-level features

## Return Value

Both scripts return a **list of PyTorch Geometric Data objects**, where each Data object represents one protein and contains:
- `x`: Node features (coordinates)
- `edge_index`: Edge connectivity [2, num_edges]
- `edge_attr`: Edge features
- `pos`: Node positions
- `sequence`: Amino acid sequence
- `node_mask`: Valid node indicators
- `tokens_mask`: Token mask for downsampling
- `protein_features`: Dictionary of protein-level features
- `num_nodes`: Number of nodes in the graph
- `num_edges`: Number of edges in the graph

## Requirements

Make sure you have:
- PyTorch installed
- PyTorch Geometric installed
- All dependencies from `requirements_pyg.txt`
- A directory with PDB files (e.g., `casp14_pdbs` or any other folder)
- The `pyg_data_pipeline.py` module in the same directory

## Performance Notes

- Processing time depends on the number and size of PDB files
- The scripts use configurable crop sizes and neighbor counts for performance tuning
- Debug mode can be disabled for quieter output
- Consider processing subsets for initial testing

## Examples

### Process CASP14 files:
```bash
python pdb_batch_processing.py --pdb_folder casp14_pdbs
```

### Process your own PDB directory:
```bash
python pdb_batch_processing.py --pdb_folder /path/to/your/pdbs
```

### Quick test with default settings:
```bash
python test_batch_processing.py /path/to/your/pdbs
```
