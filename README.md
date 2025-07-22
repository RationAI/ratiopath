# HistoPath

A Python package for histopathology image processing with advanced tiling functionality.

## Features

- **Image Tiling**: Efficiently tile large histopathology images into smaller, manageable patches
- **Flexible Configuration**: Support for custom tile sizes, overlap, and padding modes
- **Memory Efficient**: Iterator-based processing for handling very large images
- **Type Safe**: Full type annotations with mypy support
- **Well Documented**: Comprehensive docstrings and examples

## Installation

```bash
pip install histopath
```

## Quick Start

```python
import numpy as np
from histopath import tile_image, TileConfig

# Create or load an image
image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)

# Configure tiling
config = TileConfig(
    tile_size=256,      # 256x256 pixel tiles
    overlap=32,         # 32 pixel overlap between tiles
    include_partial=True # Include edge tiles that may be smaller
)

# Generate tiles
tiles = tile_image(image, config)
print(f"Generated {len(tiles)} tiles")

# Get tiles with their coordinates
tiles_with_coords = tile_image(image, config, return_coordinates=True)
for tile, (x1, y1, x2, y2) in tiles_with_coords:
    print(f"Tile shape: {tile.shape}, coordinates: ({x1}, {y1}, {x2}, {y2})")
```

## Advanced Usage

### Memory-Efficient Processing

For very large images, use the iterator interface to process tiles one at a time:

```python
from histopath import get_tile_iterator, TileConfig

config = TileConfig(tile_size=512, overlap=64)

# Process tiles one at a time (memory efficient)
for tile in get_tile_iterator("large_image.tiff", config):
    # Process each tile
    processed_tile = some_processing_function(tile)
    save_processed_tile(processed_tile)
```

### Tile Count Estimation

Calculate the number of tiles before processing:

```python
from histopath import calculate_tile_count, TileConfig

config = TileConfig(tile_size=256, overlap=32)
count = calculate_tile_count((2048, 2048), config)
print(f"Will generate {count} tiles")
```

### Configuration Options

```python
from histopath import TileConfig

config = TileConfig(
    tile_size=(256, 256),        # (width, height) or int for square
    overlap=(32, 32),            # (x_overlap, y_overlap) or int for uniform
    padding_mode="reflect",      # "reflect", "constant", "edge"
    padding_value=0,             # Value for constant padding
    include_partial=True         # Include partial tiles at edges
)
```

## API Reference

### TileConfig

Configuration class for tiling operations.

**Parameters:**
- `tile_size`: Size of each tile in pixels. Can be int (square) or (width, height) tuple
- `overlap`: Overlap between adjacent tiles. Can be int (uniform) or (x, y) tuple  
- `padding_mode`: How to handle image borders ("reflect", "constant", "edge")
- `padding_value`: Value for constant padding (default: 0)
- `include_partial`: Whether to include partial tiles at edges (default: True)

### tile_image()

Tile an image into smaller patches.

**Parameters:**
- `image`: Input image (numpy array, PIL Image, or file path)
- `config`: TileConfig object
- `return_coordinates`: If True, returns (tile, coordinates) tuples

**Returns:**
- List of tile arrays or (tile, coordinates) tuples

### get_tile_iterator()

Memory-efficient iterator for processing large images.

**Parameters:**
- `image`: Input image (numpy array, PIL Image, or file path)  
- `config`: TileConfig object
- `return_coordinates`: If True, yields (tile, coordinates) tuples

**Yields:**
- Tile arrays or (tile, coordinates) tuples

### calculate_tile_count()

Calculate the number of tiles for given image dimensions.

**Parameters:**
- `image_shape`: Image dimensions as (height, width)
- `config`: TileConfig object

**Returns:**
- Number of tiles that would be generated

## Requirements

- Python >= 3.11
- numpy >= 2.2.2
- Pillow >= 10.0.0
- dataclasses-json >= 0.6.0

## Development

This project uses modern Python tooling:

- **uv** for dependency management
- **ruff** for linting and formatting
- **mypy** for type checking

To set up the development environment:

```bash
git clone https://github.com/RationAI/histopath.git
cd histopath
uv sync --dev
```

Run linting and type checking:

```bash
uv run ruff check
uv run ruff format
uv run mypy .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.