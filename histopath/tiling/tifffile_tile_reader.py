from typing import Any

import tifffile
import zarr


def tifffile_tile_reader(row: dict[str, Any]) -> Any:
    """Read a tile from an OME-TIFF file using tifffile.

    Args:
        row: Dictionary containing tile information with keys:
            - path: Path to the OME-TIFF file
            - tile_x: X coordinate of the tile
            - tile_y: Y coordinate of the tile
            - level: Pyramid level
            - tile_extent_x: Width of the tile
            - tile_extent_y: Height of the tile

    Returns:
        The input row with an added 'tile' key containing the tile as a numpy array.
    """
    with tifffile.TiffFile(row["path"]) as tif:
        level = int(row["level"])
        tile_x, tile_extent_x = int(row["tile_x"]), int(row["tile_extent_x"])
        tile_y, tile_extent_y = int(row["tile_y"]), int(row["tile_extent_y"])

        page = tif.series[0].pages[level]
        assert isinstance(page, tifffile.TiffPage)

        z = zarr.open(page.aszarr(), mode="r")
        assert isinstance(z, zarr.Array)

        roi = z[
            tile_y : tile_y + tile_extent_y,
            tile_x : tile_x + tile_extent_x,
        ]

        row["tile"] = roi

    return row
