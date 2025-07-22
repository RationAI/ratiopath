"""
HistoPath: A Python package for histopathology image processing.

This package provides utilities for processing histopathology images,
including tiling functionality for handling large medical images.
"""

from .tiling import tile_image, TileConfig

__version__ = "0.1.0"
__all__ = ["tile_image", "TileConfig"]
