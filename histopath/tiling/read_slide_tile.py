from typing import Any


def _read_openslide_tile(row: dict[str, Any]) -> dict[str, Any]:
    """Read a tile from a whole-slide image using OpenSlide."""
    import numpy as np
    from PIL import Image

    from histopath.openslide import OpenSlide

    with OpenSlide(row["path"]) as slide:
        rgba_region = slide.read_region(
            (row["tile_x"], row["tile_y"]),
            row["level"],
            (row["tile_extent_x"], row["tile_extent_y"]),
        )
        rgb_region = Image.alpha_composite(
            Image.new("RGBA", rgba_region.size, (255, 255, 255)), rgba_region
        ).convert("RGB")
        row["tile"] = np.asarray(rgb_region)

    return row


def _read_tifffile_tile(row: dict[str, Any]) -> dict[str, Any]:
    """Read a tile from an OME-TIFF file using tifffile."""
    import tifffile
    import zarr

    with tifffile.TiffFile(row["path"]) as tif:
        level = int(row["level"])
        tile_x, tile_extent_x = int(row["tile_x"]), int(row["tile_extent_x"])
        tile_y, tile_extent_y = int(row["tile_y"]), int(row["tile_extent_y"])

        page = tif.series[0].pages[level]
        assert isinstance(page, tifffile.TiffPage)

        z = zarr.open(page.aszarr(), mode="r")
        assert isinstance(z, zarr.Array)

        row["tile"] = z[
            tile_y : tile_y + tile_extent_y,
            tile_x : tile_x + tile_extent_x,
        ]

    return row


def read_slide_tile(row: dict[str, Any]) -> dict[str, Any]:
    """Read a tile from a whole-slide image using either OpenSlide or tifffile.

    Args:
        row:
            - path: Path to the image file
            - tile_x: X coordinate of the tile
            - tile_y: Y coordinate of the tile
            - level: Pyramid level
            - tile_extent_x: Width of the tile
            - tile_extent_y: Height of the tile

    Returns:
        The input row with an added `tile` key containing the tile as a numpy array.
    """
    path = str(row["path"])

    # Check if it's an OME-TIFF file
    if path.lower().endswith((".ome.tiff", ".ome.tif")):
        return _read_tifffile_tile(row)
    return _read_openslide_tile(row)
