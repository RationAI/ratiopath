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

    The function yields tile coordinates in row-major order, iterating first over the
    x-axis (e.g. (0,0,...), (1,0,...), (2,0,...)) before incrementing the y-axis
    (0,1,...), (1,1,...), etc.

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
    dim_max = np.floor(dim_max) if last == "drop" else np.ceil(dim_max)
    dim_max = dim_max.astype(int)

    # Reverse the dimension max array to iterate over 'x' coordinates first
    dim_max = dim_max[::-1]

    # Generate tile coordinates
    if last == "drop" or last == "keep":
        for i in itertools.product(*map(range, dim_max + 1)):
            yield np.array(i[::-1]) * stride_array

    elif last == "shift":
        for i in itertools.product(*map(range, dim_max + 1)):
            base_coord = np.array(i[::-1]) * stride_array
            yield np.minimum(base_coord, slide_extent_array - tile_extent_array)
