import numpy as np
import pyarrow as pa
from ray.data.datatype import DataType
from ray.data.expressions import udf

from ratiopath.openslide import OpenSlide
from ratiopath.tifffile import TiffFile
from ratiopath.tiling.utils import (
    _pyarrow_group,
    _read_openslide_tiles,
    _read_tifffile_tiles,
)


def read_openslide_tiles(path: str, **kwargs) -> np.ndarray:
    """Read batch of tiles from a whole-slide image using OpenSlide."""
    with OpenSlide(path) as slide:
        return _read_openslide_tiles(slide, *kwargs)


def read_tifffile_tiles(path: str, **kwargs) -> np.ndarray:
    """Read batch of tiles from an OME-TIFF file using tifffile."""
    with TiffFile(path) as slide:
        return _read_tifffile_tiles(slide, *kwargs)


@udf(return_dtype=DataType(np.ndarray))
def read_slide_tiles(
    path: pa.Array,
    tile_x: pa.Array,
    tile_y: pa.Array,
    tile_extent_x: pa.Array,
    tile_extent_y: pa.Array,
    level: pa.Array,
) -> pa.Array:
    """Reads a batch of tiles from a whole-slide image using either OpenSlide or tifffile.

    Args:
        path: A pyarrow array of whole-slide image paths.
        tile_x: A pyarrow array of tile x-coordinates.
        tile_y: A pyarrow array of tile y-coordinates.
        tile_extent_x: A pyarrow array of tile extents in the x-dimension.
        tile_extent_y: A pyarrow array of tile extents in the y-dimension.
        level: A pyarrow array of slide levels to read the tiles from.

    Returns:
        A pyarrow array of numpy arrays containing the read tiles.
    """
    import pyarrow.compute as pc

    tiles = np.empty(len(tile_x), dtype=object)

    for p, group in _pyarrow_group(path).items():
        assert isinstance(p, str)

        kwargs = {
            "tile_x": pc.take(tile_x, group),
            "tile_y": pc.take(tile_y, group),
            "tile_extent_x": pc.take(tile_extent_x, group),
            "tile_extent_y": pc.take(tile_extent_y, group),
            "level": pc.take(level, group),
        }

        # Check if it's an OME-TIFF file
        if p.lower().endswith((".ome.tiff", ".ome.tif")):
            tiles[group] = read_tifffile_tiles(p, **kwargs)
        else:
            tiles[group] = read_openslide_tiles(p, **kwargs)

    return pa.array(tiles, type=np.ndarray)
