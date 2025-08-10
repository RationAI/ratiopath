from typing import Any

import numpy as np
import tifffile
from PIL import Image


def tifffile_tile_reader(row: dict[str, Any]) -> Any:
    """Read a tile from an OME-TIFF file using tifffile.
    
    Args:
        row: Dictionary containing tile information with keys:
            - path: Path to the OME-TIFF file
            - tile_x: X coordinate of the tile
            - tile_y: Y coordinate of the tile
            - level: Pyramid level
            - tile_extent_x: Width of the tile
            - tile_extent_y: Height of the tile
    
    Returns:
        The input row with an added 'tile' key containing the tile as a numpy array.
    """
    with tifffile.TiffFile(row["path"]) as tif:
        series = tif.series[0]  # Main image series
        
        # Get the page for the specified level
        level = row["level"]
        if level < len(series.pages):
            page = series.pages[level]
        else:
            # Fallback to the highest available level
            page = series.pages[-1]
        
        # Calculate region coordinates
        x = row["tile_x"]
        y = row["tile_y"]
        width = row["tile_extent_x"]
        height = row["tile_extent_y"]
        
        # Read the region from the TIFF
        # Note: tifffile uses (y, x) indexing for slicing
        tile_data = page.asarray()[y:y+height, x:x+width]
        
        # Convert to RGB if necessary
        if len(tile_data.shape) == 2:
            # Grayscale - convert to RGB
            tile_data = np.stack([tile_data] * 3, axis=-1)
        elif tile_data.shape[2] == 4:
            # RGBA - convert to RGB by compositing with white background
            rgba_image = Image.fromarray(tile_data, 'RGBA')
            background = Image.new("RGB", rgba_image.size, (255, 255, 255))
            rgb_image = Image.alpha_composite(
                background.convert('RGBA'), rgba_image
            ).convert('RGB')
            tile_data = np.asarray(rgb_image)
        elif tile_data.shape[2] > 4:
            # Multi-channel - take first 3 channels
            tile_data = tile_data[:, :, :3]
        
        # Ensure we have the right data type
        if tile_data.dtype != np.uint8:
            # Normalize and convert to uint8 if needed
            if tile_data.max() > 255:
                tile_data = (tile_data / tile_data.max() * 255).astype(np.uint8)
            else:
                tile_data = tile_data.astype(np.uint8)
        
        row["tile"] = tile_data

    return row