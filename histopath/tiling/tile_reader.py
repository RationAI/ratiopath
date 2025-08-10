from typing import Any

from .openslide_tile_reader import openslide_tile_reader
from .tifffile_tile_reader import tifffile_tile_reader


def tile_reader(row: dict[str, Any]) -> Any:
    """Unified tile reader that chooses the appropriate implementation based on file extension.
    
    This function automatically selects between openslide_tile_reader and tifffile_tile_reader
    based on the file extension in the row["path"] field.
    
    Args:
        row: Dictionary containing tile information with keys:
            - path: Path to the image file
            - tile_x: X coordinate of the tile
            - tile_y: Y coordinate of the tile
            - level: Pyramid level
            - tile_extent_x: Width of the tile
            - tile_extent_y: Height of the tile
    
    Returns:
        The input row with an added 'tile' key containing the tile as a numpy array.
    """
    path = row["path"]
    
    # Check if it's an OME-TIFF file
    if path.lower().endswith(('.ome.tiff', '.ome.tif')):
        return tifffile_tile_reader(row)
    else:
        return openslide_tile_reader(row)