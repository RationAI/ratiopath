import hashlib
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from ratiopath.openslide import OpenSlide
from ratiopath.tifffile import TiffFile


def row_hash(
    row: dict[str, Any],
    column: str = "id",
    algorithm: Callable[[bytes], hashlib._hashlib.HASH] = hashlib.sha256,  # type: ignore[name-defined]
) -> dict[str, Any]:
    """Hashes a row (dictionary) using SHA256 and adds the hash as a new column.

    Args:
        row: The dictionary (row) to hash.
        column: The name of the column to store the hash. Defaults to "id".
        algorithm: The hashing algorithm to use. Defaults to hashlib.sha256.

    Returns:
        The modified row (dictionary) with the SHA256 hash added.
    """
    row[column] = algorithm(str(row).encode()).hexdigest()
    return row


def _read_openslide_tiles(slide: OpenSlide, df: DataFrame) -> pd.Series:
    """Read batch of tiles from a whole-slide image using OpenSlide."""
    from PIL import Image

    def get_tile(row: pd.Series) -> np.ndarray:
        rgba_region = slide.read_region_relative(
            (row["tile_x"], row["tile_y"]),
            row["level"],
            (row["tile_extent_x"], row["tile_extent_y"]),
        )
        rgb_region = Image.alpha_composite(
            Image.new("RGBA", rgba_region.size, (255, 255, 255)), rgba_region
        ).convert("RGB")
        return np.asarray(rgb_region)

    return df.apply(get_tile, axis=1)


def _read_tifffile_tiles(slide: TiffFile, df: DataFrame) -> pd.Series:
    """Read batch of tiles from an OME-TIFF file using tifffile."""
    import tifffile
    import zarr
    from zarr.core.buffer import NDArrayLike

    def get_tile(row: pd.Series, z: zarr.Array) -> np.ndarray:
        arr = np.full(
            (row["tile_extent_y"], row["tile_extent_x"], 3), 255, dtype=np.uint8
        )
        tile_slice = z[
            row["tile_y"] : row["tile_y"] + row["tile_extent_y"],
            row["tile_x"] : row["tile_x"] + row["tile_extent_x"],
        ]
        assert isinstance(tile_slice, NDArrayLike)
        arr[: tile_slice.shape[0], : tile_slice.shape[1]] = tile_slice[..., :3]  # type: ignore[index]
        return arr

    tiles = pd.Series(index=df.index, dtype=object)
    for level, group in df.groupby("level"):
        assert isinstance(level, int)
        page = slide.series[0].pages[level]
        assert isinstance(page, tifffile.TiffPage)

        z = zarr.open(page.aszarr(), mode="r")
        assert isinstance(z, zarr.Array)

        tiles.loc[group.index] = group.apply(partial(get_tile, z=z), axis=1)

    return tiles
