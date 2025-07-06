# Protein Structure Data Pipeline

This directory contains a comprehensive data preprocessing pipeline for protein structure data, designed to work with the VQ-VAE protein structure tokenizer.

## Files

- **`data_pipeline.py`** - Main pipeline script with both programmatic API and CLI
- **`config_example.yaml`** - Example configuration file with all available parameters
- **`examples.py`** - Comprehensive examples showing different usage patterns
- **`README_pipeline.md`** - This documentation file

## Quick Start

### 1. Basic Usage - Command Line

```bash
# Process a PDB file with default settings
python data_pipeline.py --input_path casp14_pdbs/T1024.pdb --output_path processed.npy

# Process with custom configuration
python data_pipeline.py --input_path protein.pdb --config config_example.yaml --output_path processed.npy

# Get information about a protein without processing
python data_pipeline.py --input_path protein.pdb --info_only

# Process a specific chain
python data_pipeline.py --input_path protein.pdb --chain_id A --output_path processed.npy
```

### 2. Basic Usage - Python API

```python
from data_pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline()

# Process a PDB file
processed_data = pipeline.process_single(
    input_source="protein.pdb",
    output_path="processed.npy"
)

# Get sample information
sample = pipeline.load_from_pdb_file("protein.pdb")
info = pipeline.get_sample_info(sample)
print(f"Protein has {info['total_residues']} residues")
```

### 3. Run Examples

```bash
python examples.py
```

## Input Formats

The pipeline supports multiple input formats:

### PDB Files
```bash
python data_pipeline.py --input_path protein.pdb --output_path processed.npy
```

### PDB Strings
```bash
python data_pipeline.py --pdb_string "ATOM ..." --output_path processed.npy
```

### Numpy Files (previously saved ProteinStructureSample)
```bash
python data_pipeline.py --input_path protein_sample.npy --output_path processed.npy
```

## Configuration

### Default Parameters
```yaml
num_neighbor: 30                    # k-NN graph neighbors
downsampling_ratio: 4               # Structure to sequence token ratio
residue_loc_is_alphac: true         # Use alpha-carbon for residue location
padding_num_residue: 512            # Max residues after padding
crop_index: 512                     # Max sequence length after cropping
noise_level: 0.0                    # Data augmentation noise
min_number_valid_residues: 10       # Minimum valid residues
max_number_residues: 1000           # Maximum residues allowed
```

### Custom Configuration

Create a YAML file (see `config_example.yaml`):

```yaml
# Custom config for small proteins
num_neighbor: 20
crop_index: 256
padding_num_residue: 256
min_number_valid_residues: 5
```

Use it with:
```bash
python data_pipeline.py --input_path protein.pdb --config my_config.yaml --output_path processed.npy
```

## Command Line Arguments

### Required Arguments
- `--input_path` OR `--pdb_string`: Input source
- `--output_path`: Where to save processed data (unless using `--info_only`)

### Optional Arguments
- `--config`: Path to YAML configuration file
- `--chain_id`: Process specific chain (e.g., "A")
- `--num_neighbor`: Number of k-NN neighbors (default: 30)
- `--crop_index`: Max sequence length (default: 512)
- `--padding_num_residue`: Padding size (default: 512)
- `--noise_level`: Data augmentation noise (default: 0.0)
- `--output_format`: "npy" or "npz" (default: "npy")
- `--save_intermediate`: Save raw ProteinStructureSample
- `--info_only`: Just display protein information

## Output Format

The pipeline outputs a `BatchDataVQ3D` object containing:

### Graph Structure
- **nodes**: Residue coordinates and features
- **edges**: k-NN graph connectivity with distance and orientation features
- **masks**: Valid node/edge indicators

### Features
- **aatype**: One-hot encoded amino acid types
- **atom37_gt_positions**: Ground truth atomic coordinates
- **backbone_affine_tensor**: Local reference frames
- **rigid group features**: Protein structural features

## Preprocessing Steps

1. **PDB Parsing**: Convert PDB to internal representation
2. **Quality Filtering**: Remove low-quality samples
3. **Coordinate Processing**: Handle missing atoms and coordinates
4. **Local Frame Computation**: Calculate backbone reference frames
5. **Graph Construction**: Build k-nearest neighbor graphs
6. **Feature Engineering**: Create node and edge features
7. **Sequence Processing**: Apply cropping and padding
8. **Batching Preparation**: Format for model input

## Error Handling

The pipeline includes comprehensive error handling:

- **File not found**: Clear error messages for missing files
- **Invalid PDB**: Detailed parsing error information
- **Quality filtering**: Warnings for samples that don't meet criteria
- **Memory issues**: Suggestions for reducing parameters

## Performance Tips

### For Large Proteins (>500 residues)
```yaml
crop_index: 1024
padding_num_residue: 1024
num_neighbor: 50
```

### For Small Proteins (<100 residues)
```yaml
crop_index: 128
padding_num_residue: 128
num_neighbor: 20
```

### For Training (with data augmentation)
```yaml
noise_level: 0.1
save_intermediate: true
```

### For Inference (fast processing)
```yaml
num_neighbor: 20
noise_level: 0.0
save_intermediate: false
```

## Batch Processing

```python
from data_pipeline import DataPipeline
import os

pipeline = DataPipeline()

for pdb_file in os.listdir("pdb_directory"):
    if pdb_file.endswith('.pdb'):
        try:
            output_path = f"processed_{pdb_file.replace('.pdb', '.npy')}"
            pipeline.process_single(
                input_source=os.path.join("pdb_directory", pdb_file),
                output_path=output_path
            )
            print(f"✓ Processed {pdb_file}")
        except Exception as e:
            print(f"✗ Failed {pdb_file}: {e}")
```

## Troubleshooting

### Common Issues

1. **"Sample filtered out"**: Protein doesn't meet quality criteria
   - Solution: Adjust `min_number_valid_residues` or `max_number_residues`

2. **"PDB contains insertion code"**: PDB has non-standard residue numbering
   - Solution: Clean PDB file or use different structure

3. **Memory errors**: Protein too large for current settings
   - Solution: Reduce `padding_num_residue` and `crop_index`

4. **"No known atom positions"**: All atoms missing for some residues
   - Solution: Check PDB quality, may need different structure

### Debug Mode

For detailed error information:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Model

The processed output can be directly used with the VQ-VAE model:

```python
# Load processed data
import numpy as np
processed_data = np.load("processed.npy", allow_pickle=True).item()

# Use with model
tokens = model.encode(processed_data)
reconstructed = model.decode(tokens)
```

## Dependencies

- numpy
- jax
- biopandas
- Bio.PDB (biopython)
- scipy
- pyyaml

Install with:
```bash
pip install numpy jax biopandas biopython scipy pyyaml
```
