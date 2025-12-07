from typing import TypedDict

import pyarrow as pa


class ReadTilesArguments(TypedDict):
    """TypedDict for read_tiles arguments."""

    tile_x: pa.IntegerArray
    tile_y: pa.IntegerArray
    tile_extent_x: pa.IntegerArray
    tile_extent_y: pa.IntegerArray
    level: pa.IntegerArray


class ReadOverlaysArguments(TypedDict):
    """TypedDict for read_overlays arguments."""

    tile_x: pa.IntegerArray
    tile_y: pa.IntegerArray
    tile_extent_x: pa.IntegerArray
    tile_extent_y: pa.IntegerArray
    mpp_x: pa.FloatArray
    mpp_y: pa.FloatArray
