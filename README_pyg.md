# PyTorch Geometric Data Pipeline

This directory contains a comprehensive PyTorch Geometric (PyG) implementation of the protein structure preprocessing pipeline. The PyG pipeline provides the same functionality as the original pipeline but outputs data in PyG-compatible format for seamless integration with graph neural networks.

## Overview

The PyG data pipeline converts protein structure data (PDB files) into PyTorch Geometric `Data` objects that can be directly used with PyG models. It maintains all the preprocessing capabilities of the original pipeline while providing additional features for GNN workflows.

## Key Features

- **Native PyG Compatibility**: Outputs `torch_geometric.data.Data` objects
- **Efficient Graph Representation**: Uses PyG's optimized edge_index format [2, num_edges]
- **Flexible Device Support**: CPU/GPU tensor creation
- **Batch Processing**: Built-in support for multi-protein batches
- **Backward Compatibility**: Convert existing `BatchDataVQ3D` to PyG format
- **Memory Optimization**: Configurable precision and device placement

## Installation

1. Install the base requirements:
```bash
pip install -r requirements_preprocessing.txt
```

2. Install PyG-specific dependencies:
```bash
pip install -r requirements_pyg.txt
```

Alternatively, install PyTorch Geometric manually:
```bash
pip install torch torch-geometric
```

## Quick Start

### Basic Usage

```python
from pyg_data_pipeline import PyGDataPipeline

# Initialize pipeline
pipeline = PyGDataPipeline()

# Process a single PDB file
pyg_data = pipeline.process_single(
    input_source="path/to/protein.pdb",
    output_path="protein_pyg.pt"
)

# Use with PyG models
from torch_geometric.nn import GCNConv
model = GCNConv(3, 64)  # 3D coords -> 64 features
output = model(pyg_data.x, pyg_data.edge_index)
```

### Command Line Interface

```bash
# Process a single PDB file
python pyg_data_pipeline.py --input_path protein.pdb --output_path processed.pt

# Process with custom configuration
python pyg_data_pipeline.py --input_path protein.pdb --config config_pyg.yaml --output_path processed.pt

# Process a directory of PDB files
python pyg_data_pipeline.py --input_dir pdbs/ --output_dir outputs/

# Get protein information only
python pyg_data_pipeline.py --input_path protein.pdb --info_only
```

## Configuration

The pipeline uses YAML configuration files. See `config_pyg.yaml` for a complete example:

```yaml
# Basic parameters
num_neighbor: 30
crop_index: 512
noise_level: 0.0

# PyG-specific settings
output_format: "pt"
include_global_features: true
dtype: "float32"
device: "cpu"
```

## Data Format

The PyG pipeline outputs `torch_geometric.data.Data` objects with the following structure:

```python
Data(
    # Core graph structure
    x=[num_nodes, 3],           # Node features (3D coordinates)
    edge_index=[2, num_edges],  # Edge connectivity
    edge_attr=[num_edges, edge_dim],  # Edge features
    pos=[num_nodes, 3],         # Node positions
    
    # Sequence and masks
    sequence=[num_nodes],       # Amino acid types
    node_mask=[num_nodes],      # Valid node indicator
    tokens_mask=[num_tokens],   # Token mask for downsampling
    
    # Metadata
    num_nodes=int,              # Number of nodes
    num_edges=int,              # Number of edges
    chain_id=str,               # Chain identifier
    resolution=float,           # Crystal resolution
    
    # Optional protein features
    protein_features=dict,      # Structure module features
)
```

## API Reference

### PyGDataPipeline Class

Main class for PyG data preprocessing:

```python
pipeline = PyGDataPipeline(config=None)

# Core methods
pipeline.process_single(input_source, output_path=None, input_type="auto")
pipeline.process_batch(input_sources, output_dir=None, input_type="auto")
pipeline.preprocess_to_pyg(sample)
pipeline.validate_sample(sample)

# I/O methods
pipeline.load_from_pdb_file(pdb_file_path, chain_id=None)
pipeline.load_from_pdb_string(pdb_string, chain_id=None)
pipeline.save_pyg_data(data, output_path)
```

### Utility Functions

```python
# Convert existing format to PyG
convert_batch_data_to_pyg(batch_data_vq3d, config=None)

# Load configuration
load_config_from_file(config_path)
```

## Examples

### Example 1: Single Protein Processing

```python
from pyg_data_pipeline import PyGDataPipeline

pipeline = PyGDataPipeline({
    "num_neighbor": 20,
    "crop_index": 256,
    "include_global_features": True
})

pyg_data = pipeline.process_single("protein.pdb")
print(f"Nodes: {pyg_data.num_nodes}, Edges: {pyg_data.num_edges}")
```

### Example 2: Batch Processing

```python
import glob
from torch_geometric.data import DataLoader

# Process multiple proteins
pdb_files = glob.glob("pdbs/*.pdb")
batch = pipeline.process_batch(pdb_files)

# Create DataLoader for training
loader = DataLoader([batch], batch_size=1)
```

### Example 3: Using with GNN Models

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ProteinGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)

# Use with PyG data
model = ProteinGNN(3, 64, 32)
output = model(pyg_data)
```

### Example 4: Converting Existing Data

```python
from data_pipeline import DataPipeline
from pyg_data_pipeline import convert_batch_data_to_pyg

# Process with original pipeline
original_pipeline = DataPipeline()
batch_data = original_pipeline.process_single("protein.pdb")

# Convert to PyG format
pyg_data = convert_batch_data_to_pyg(batch_data)
```

## Advanced Features

### Memory Optimization

```python
config = {
    "dtype": "float32",        # Use half precision
    "device": "cuda",          # GPU processing
    "padding_num_residue": 256 # Reduce padding
}
pipeline = PyGDataPipeline(config)
```

### Edge Pruning

```python
config = {
    "edge_pruning": {
        "enabled": True,
        "max_distance": 20.0   # Remove edges > 20Å
    }
}
```

### Custom Feature Processing

```python
# Access and modify features after processing
pyg_data = pipeline.process_single("protein.pdb")

# Add custom node features
pyg_data.custom_features = compute_custom_features(pyg_data.pos)

# Modify edge attributes
pyg_data.edge_attr = torch.cat([pyg_data.edge_attr, custom_edge_features], dim=1)
```

## Performance Considerations

1. **Memory Usage**: Use `float32` instead of `float64` for large proteins
2. **GPU Processing**: Set `device="cuda"` for GPU acceleration
3. **Batch Size**: Adjust based on available memory
4. **Edge Pruning**: Enable for very large proteins to reduce memory

## Comparison with Original Pipeline

| Feature | Original Pipeline | PyG Pipeline |
|---------|------------------|--------------|
| Output Format | `BatchDataVQ3D` | `torch_geometric.data.Data` |
| Edge Representation | Separate senders/receivers | `edge_index [2, num_edges]` |
| Framework | JAX/NumPy | PyTorch |
| GPU Support | JAX devices | PyTorch devices |
| Batch Processing | Manual | Built-in PyG batching |
| Model Integration | Custom GNNs | Native PyG models |

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PyTorch Geometric is properly installed
2. **Memory Issues**: Reduce `crop_index` or use `float32` precision
3. **CUDA Errors**: Check GPU compatibility and driver versions
4. **File Not Found**: Verify PDB file paths and permissions

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = PyGDataPipeline(config)
```

## Contributing

To extend the PyG pipeline:

1. Add new features to the `PyGDataPipeline` class
2. Update configuration schema in `config_pyg.yaml`
3. Add tests in `test_pyg_pipeline.py`
4. Update documentation

## Related Files

- `pyg_data_pipeline.py`: Main PyG pipeline implementation
- `config_pyg.yaml`: Default configuration file
- `examples_pyg.py`: Usage examples and demonstrations
- `requirements_pyg.txt`: PyG-specific dependencies
- `data_pipeline.py`: Original pipeline (for comparison)

## License

Same as the main project - Apache License 2.0.
