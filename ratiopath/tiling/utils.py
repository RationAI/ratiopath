import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

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


def _read_openslide_tiles(
    slide: OpenSlide,
    tile_x: pa.Array,
    tile_y: pa.Array,
    tile_extent_x: pa.Array,
    tile_extent_y: pa.Array,
    level: pa.Array,
) -> np.ndarray:
    """Read batch of tiles from a whole-slide image using OpenSlide."""
    from PIL import Image

    def get_tile(
        x: int, y: int, extent_x: int, extent_y: int, level: int
    ) -> np.ndarray:
        rgba_region = slide.read_region_relative(
            (x, y),
            level,
            (extent_x, extent_y),
        )
        rgb_region = Image.alpha_composite(
            Image.new("RGBA", rgba_region.size, (255, 255, 255)), rgba_region
        ).convert("RGB")
        return np.asarray(rgb_region)

    return np.array(
        [
            get_tile(*args)
            for args in zip(
                tile_x.to_numpy(),
                tile_y.to_numpy(),
                tile_extent_x.to_numpy(),
                tile_extent_y.to_numpy(),
                level.to_numpy(),
                strict=True,
            )
        ]
    )


def _read_tifffile_tiles(
    slide: TiffFile,
    tile_x: pa.Array,
    tile_y: pa.Array,
    tile_extent_x: pa.Array,
    tile_extent_y: pa.Array,
    level: pa.Array,
) -> np.ndarray:
    """Read batch of tiles from an OME-TIFF file using tifffile."""
    import tifffile
    import zarr
    from zarr.core.buffer import NDArrayLike

    def get_tile(
        z: zarr.Array, x: int, y: int, extent_x: int, extent_y: int
    ) -> np.ndarray:
        arr = np.full((extent_y, extent_x, 3), 255, dtype=np.uint8)
        tile_slice = z[
            y : y + extent_y,
            x : x + extent_x,
        ]
        assert isinstance(tile_slice, NDArrayLike)
        arr[: tile_slice.shape[0], : tile_slice.shape[1]] = tile_slice[..., :3]  # type: ignore[index]
        return arr

    # All the lists must have the same length
    tiles = np.empty(len(tile_x), dtype=object)

    for page, group in _pyarrow_group_indices(level).items():
        assert isinstance(page, int)
        page = slide.series[0].pages[page]
        assert isinstance(page, tifffile.TiffPage)

        z = zarr.open(page.aszarr(), mode="r")
        assert isinstance(z, zarr.Array)

        tiles[group] = [
            get_tile(z, *args)
            for args in zip(
                pc.take(tile_x, group).to_numpy(),
                pc.take(tile_y, group).to_numpy(),
                pc.take(tile_extent_x, group).to_numpy(),
                pc.take(tile_extent_y, group).to_numpy(),
                strict=True,
            )
        ]

    return tiles


def _pyarrow_group_indices(x: pa.Array) -> dict[Any, pa.Array]:
    """Group indices of a PyArrow array by unique values.

    Args:
        x: A PyArrow array to group.

    Returns:
        A dictionary mapping unique values to PyArrow arrays of integer indices where those values occur.
    """
    unique_values = pc.unique(x)  # type: ignore []
    full_indices = pa.arange(0, len(x))

    groups = {}

    for value in unique_values:
        mask = pc.equal(x, value)  # type: ignore []
        groups[value.as_py()] = pc.filter(full_indices, mask)  # type: ignore []

    return groups
