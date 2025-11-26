from typing import Any

import pandas as pd
from pandas import DataFrame

from ratiopath.openslide import OpenSlide
from ratiopath.tifffile import TiffFile
from ratiopath.tiling.utils import _read_openslide_tiles, _read_tifffile_tiles


def read_openslide_tiles(path: str, df: DataFrame) -> pd.Series:
    """Read batch of tiles from a whole-slide image using OpenSlide."""
    with OpenSlide(path) as slide:
        return _read_openslide_tiles(slide, df)


def read_tifffile_tiles(path: str, df: DataFrame) -> pd.Series:
    """Read batch of tiles from an OME-TIFF file using tifffile."""
    with TiffFile(path) as slide:
        return _read_tifffile_tiles(slide, df)


def read_slide_tiles(batch: dict[str, Any]) -> dict[str, Any]:
    """Reads a batch of tiles from a whole-slide image using either OpenSlide or tifffile.

    Args:
        batch:
            - tile_x: X coordinates of tiles relative to the level
            - tile_y: Y coordinates of tiles relative to the level
            - level: Pyramid levels
            - tile_extent_x: Widths of the tiles
            - tile_extent_y: Heights of the tiles

    Returns:
        The input batch with an added `tile` key containing the list of numpy array tiles.
    """
    # Check if it's an OME-TIFF file
    df = pd.DataFrame(batch)
    for path, group in df.groupby("path"):
        assert isinstance(path, str)
        if path.lower().endswith((".ome.tiff", ".ome.tif")):
            df.loc[group.index, "tile"] = read_tifffile_tiles(path, group)
        else:
            df.loc[group.index, "tile"] = read_openslide_tiles(path, group)

    batch["tile"] = df["tile"].tolist()
    return batch
