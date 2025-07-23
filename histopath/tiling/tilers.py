import itertools
import warnings
from collections.abc import Iterator
from typing import Literal, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

Dims = TypeVarTuple("Dims")


def grid_tiles(
    slide_extent: tuple[int, *Dims],
    tile_extent: tuple[int, *Dims],
    stride: tuple[int, *Dims],
    last: Literal["shift", "drop", "keep"] = "drop",
) -> Iterator[NDArray[np.int64]]:
    """Generates tiles for the given slide based on its size, tile size, and stride.

    Args:
        slide_extent: The dimensions of the slide in pixels.
        tile_extent: The dimensions of the tile in pixels.
        stride: The stride between tiles in pixels.
        last: The strategy to handle the last tile when it does not fit the stride.
            - "shift": Shift the last tile to the left and up to fit the stride.
            - "drop": Drop the last tile if it does not fit the stride.
            - "keep": Keep the last tile even if it does not fit the slide.

    Returns:
        An iterator of numpy arrays containing the tile coordinates.
    """
    slide_extent_array = np.asarray(slide_extent)
    tile_extent_array = np.asarray(tile_extent)
    stride_array = np.asarray(stride)

    if any(tile_extent_array > slide_extent_array):
        warnings.warn(
            f"TilingModule: tile size {tile_extent_array} is greater than slide dimensions {slide_extent_array}",
            UserWarning,
            stacklevel=2,
        )

    dim_max = (slide_extent_array - tile_extent_array) / stride_array
    if last == "drop":
        dim_max = np.floor(dim_max)
    else:
        dim_max = np.ceil(dim_max)
    dim_max = dim_max.astype(int)

    # Generate tile coordinates
    if last == "drop" or last == "keep":
        for i in itertools.product(*map(range, dim_max + 1)):
            yield np.array(i) * stride_array

    elif last == "shift":
        for i in itertools.product(*map(range, dim_max + 1)):
            base_coord = np.array(i) * stride_array
            yield np.minimum(base_coord, slide_extent_array - tile_extent_array)
