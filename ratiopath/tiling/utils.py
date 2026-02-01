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
    tile_x: pa.IntegerArray,
    tile_y: pa.IntegerArray,
    tile_extent_x: pa.IntegerArray,
    tile_extent_y: pa.IntegerArray,
    level: pa.IntegerArray,
    background: int = 255,
) -> np.ndarray:
    """Read batch of tiles from a whole-slide image using OpenSlide.

    Args:
        slide: The OpenSlide object to read tiles from.
        tile_x: Array of tile x-coordinates (in pixels) at level 0.
        tile_y: Array of tile y-coordinates (in pixels) at level 0.
        tile_extent_x: Array of tile widths in pixels.
        tile_extent_y: Array of tile heights in pixels.
        level: Array of OpenSlide level indices for each tile.
        background: The RGB value (0-255) to use for transparent areas.
            Defaults to 255 (white).

    Returns:
        A NumPy array of RGB tiles read from the slide.
    """
    from PIL import Image

    def get_tile(
        x: pa.Scalar,
        y: pa.Scalar,
        extent_x: pa.Scalar,
        extent_y: pa.Scalar,
        level: pa.Scalar,
    ) -> np.ndarray:
        rgba_region = slide.read_region_relative(
            (x.as_py(), y.as_py()),
            level.as_py(),
            (extent_x.as_py(), extent_y.as_py()),
        )
        rgb_region = Image.alpha_composite(
            Image.new("RGBA", rgba_region.size, (background, background, background)),
            rgba_region,
        ).convert("RGB")
        return np.asarray(rgb_region)

    return np.array(
        [
            get_tile(*args)
            for args in zip(
                tile_x,
                tile_y,
                tile_extent_x,
                tile_extent_y,
                level,
                strict=True,
            )
        ]
    )


def _read_tifffile_tiles(
    slide: TiffFile,
    tile_x: pa.IntegerArray,
    tile_y: pa.IntegerArray,
    tile_extent_x: pa.IntegerArray,
    tile_extent_y: pa.IntegerArray,
    level: pa.IntegerArray,
    background: int = 255,
) -> np.ndarray:
    """Read batch of tiles from an OME-TIFF file using tifffile.

    Args:
        slide: The OpenSlide object to read tiles from.
        tile_x: Array of tile x-coordinates (in pixels) at level 0.
        tile_y: Array of tile y-coordinates (in pixels) at level 0.
        tile_extent_x: Array of tile widths in pixels.
        tile_extent_y: Array of tile heights in pixels.
        level: Array of OpenSlide level indices for each tile.
        background: The RGB value (0-255) to use for transparent areas.
            Defaults to 255 (white).

    Returns:
        A NumPy array of RGB tiles read from the slide.
    """
    import tifffile
    import zarr
    from zarr.core.buffer import NDArrayLike

    def get_tile(
        z: zarr.Array,
        x: pa.Scalar,
        y: pa.Scalar,
        extent_x: pa.Scalar,
        extent_y: pa.Scalar,
    ) -> np.ndarray:
        x_py = x.as_py()
        y_py = y.as_py()
        extent_x_py: int = extent_x.as_py()
        extent_y_py: int = extent_y.as_py()

        arr = np.full((extent_y_py, extent_x_py, 3), background, dtype=np.uint8)
        tile_slice = z[
            y_py : y_py + extent_y_py,
            x_py : x_py + extent_x_py,
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
                pc.take(tile_x, group),
                pc.take(tile_y, group),
                pc.take(tile_extent_x, group),
                pc.take(tile_extent_y, group),
                strict=True,
            )
        ]

    return tiles


def _pyarrow_group_indices(x: pa.Array) -> dict[Any, pa.Int32Array]:
    """Group indices of a PyArrow array by unique values.

    Args:
        x: A PyArrow array to group.

    Returns:
        A dictionary mapping unique values to PyArrow arrays of integer indices where those values occur.
    """
    unique_values = pc.unique(x)
    full_indices = pa.array(range(len(x)), type=pa.int32())

    groups = {}

    for value in unique_values:
        mask = pc.equal(x, value)
        groups[value.as_py()] = pc.filter(full_indices, mask)

    return groups
