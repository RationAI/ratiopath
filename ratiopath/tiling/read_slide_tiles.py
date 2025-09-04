from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


def _read_openslide_tiles(path: str, df: DataFrame) -> pd.Series:
    """Read batch of tiles from a whole-slide image using OpenSlide."""
    from PIL import Image

    from ratiopath.openslide import OpenSlide

    with OpenSlide(path) as slide:

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


def _read_tifffile_tiles(path: str, df: DataFrame) -> pd.Series:
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
    with tifffile.TiffFile(path) as tif:
        for level, group in df.groupby("level"):
            assert isinstance(level, int)
            page = tif.series[0].pages[level]
            assert isinstance(page, tifffile.TiffPage)

            z = zarr.open(page.aszarr(), mode="r")
            assert isinstance(z, zarr.Array)

            tiles.loc[group.index] = group.apply(partial(get_tile, z=z), axis=1)

    return tiles


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
            df.loc[group.index, "tile"] = _read_tifffile_tiles(path, group)
        else:
            df.loc[group.index, "tile"] = _read_openslide_tiles(path, group)

    batch["tile"] = df["tile"].tolist()
    return batch
