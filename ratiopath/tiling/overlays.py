from typing import Any

import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from rasterio.transform import Affine
from shapely.geometry.base import BaseGeometry

from ratiopath.openslide import OpenSlide
from ratiopath.tifffile import TiffFile
from ratiopath.tiling.utils import _read_openslide_tiles, _read_tifffile_tiles


def _scale_overlay_args(df: pd.DataFrame, slide: OpenSlide | TiffFile) -> pd.DataFrame:
    """Scale overlay tile arguments based on slide and overlay resolutions.

    Args:
        df: DataFrame containing tile arguments
        slide: The whole-slide image (OpenSlide or TiffFile).

    Returns:
        - pd.DataFrame: Scaled overlay tile arguments

    Raises:
        - ValueError: If neither 'mpp_x' and 'mpp_y' nor 'level' are present in the DataFrame.
    """
    # Determine resolution of the underlying tile
    if "mpp_x" not in df.columns or "mpp_y" not in df.columns:
        if "level" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'mpp_x' and 'mpp_y' columns or 'level' column."
            )

        mpp = df.apply(lambda row: slide.slide_resolution(row["level"]))
    else:
        mpp = df.apply(lambda row: (row["mpp_x"], row["mpp_y"]))

    # Determine closest level and overlay resolution
    level = mpp.apply(lambda res: slide.closest_level(res))
    overlay_mpp = df.apply(lambda row: slide.slide_resolution(level[row.name]))

    # Compute scaling factors
    scaling_factor_x: pd.Series[float] = mpp.apply(
        lambda mpp: mpp[0]
    ) / overlay_mpp.apply(lambda mpp: mpp[0])  # type: ignore[return-value]
    scaling_factor_y: pd.Series[float] = mpp.apply(
        lambda mpp: mpp[1]
    ) / overlay_mpp.apply(lambda mpp: mpp[1])  # type: ignore[return-value]

    def scale(df: pd.Series, scale: pd.Series) -> pd.Series:
        return (df * scale).round(0).astype(int)

    # Scale tile coordinates and extents
    df["tile_x"] = scale(df["tile_x"], scaling_factor_x)
    df["tile_y"] = scale(df["tile_y"], scaling_factor_y)
    df["tile_extent_x"] = scale(df["tile_extent_x"], scaling_factor_x)
    df["tile_extent_y"] = scale(df["tile_extent_y"], scaling_factor_y)
    df["level"] = level

    return df


def _read_openslide_overlay(path: str, df: pd.DataFrame) -> pd.Series:
    """Read batch of overlays from a whole-slide image using OpenSlide."""
    with OpenSlide(path) as slide:
        return _read_openslide_tiles(slide, _scale_overlay_args(df, slide))


def _read_tifffile_overlay(path: str, df: pd.DataFrame) -> pd.Series:
    """Read batch of overlays from an OME-TIFF file using tifffile."""
    with TiffFile(path) as slide:
        return _read_tifffile_tiles(slide, _scale_overlay_args(df, slide))


def tile_overlay(
    batch: dict[str, Any], overlay_path_key: str, roi: BaseGeometry
) -> pd.Series:
    """Read overlay tiles for a batch of tiles.

    For each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    Args:
        batch (dict[str, Any]):
            - tile_x: X coordinates of the underlying tiles
            - tile_y: Y coordinates of the underlying
            - tile_extent_x: Widths of the underlying tiles
            - tile_extent_y: Heights of the underlying tiles
            - level (Optional): Level of the underlying slide
            - mpp_x (Optional): Physical resolution (µm/px) of the underlying slide in X direction
            - mpp_y (Optional): Physical resolution (µm/px) of the underlying slide in Y direction
        overlay_path_key: Key in the batch dict that contains the path to the overlay whole-slide image
        roi: The region of interest geometry.

    Returns:
        A pandas Series containing the overlay tiles as numpy masked arrays.

    Raises:
        ValueError: If neither 'mpp_x' and 'mpp_y' nor 'level' are present in the batch.
    """
    # Convert batch to DataFrame and rename overlay path column
    df = pd.DataFrame(batch).rename(columns={overlay_path_key: "overlay_path"})

    # Get ROI bounds
    sx, sy, ex, ey = map(round, roi.bounds)

    # Adjust tile coordinates and extents based on ROI
    w, h = ex - sx, ey - sy
    x: pd.Series[int] = df["tile_x"] - sx
    y: pd.Series[int] = df["tile_y"] - sy

    df["tile_x"] = x
    df["tile_y"] = y
    df["tile_extent_x"] = w
    df["tile_extent_y"] = h

    # Read overlay tiles
    tiles = pd.Series(index=df.index, dtype=object)

    for path, group in df.groupby("overlay_path"):
        assert isinstance(path, str)
        if path.lower().endswith((".ome.tiff", ".ome.tif")):
            tiles.loc[group.index] = _read_tifffile_overlay(path, group)
        else:
            tiles.loc[group.index] = _read_openslide_overlay(path, group)

    # Adjust coordinates for masking
    sx = x.apply(lambda v: min(-sx, -sx + v))  # type: ignore[call-arg]
    sy = y.apply(lambda v: min(-sy, -sy + v))  # type: ignore[call-arg]

    masks = tiles.apply(
        lambda overlay: rasterio.features.geometry_mask(
            geometries=[roi],
            out_shape=overlay.shape[:2],
            # Scale and translate to tile coordinates - reflect ROI cropping
            transform=Affine.translation(-sx[overlay.index], -sy[overlay.index])
            * Affine.scale(df["tile_extent_x"] / w, df["tile_extent_y"] / h),
        )
    )

    # Create masked arrays for each overlay tile
    return tiles.apply(
        lambda overlay, mask: np.ma.masked_array(overlay, mask=mask),  # type: ignore[return-value, arg-type]
        mask=masks,
    )


def tile_overlay_overlap(
    batch: dict[str, Any], overlay_path_key: str, roi: BaseGeometry
) -> pd.Series:
    """Calculate the overlap of each overlay tile with the region of interest (ROI).

    For each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    Args:
        batch (dict[str, Any]):
            - tile_x: X coordinates of the underlying tiles
            - tile_y: Y coordinates of the underlying
            - tile_extent_x: Widths of the underlying tiles
            - tile_extent_y: Heights of the underlying tiles
            - level (Optional): Level of the underlying slide
            - mpp_x (Optional): Physical resolution (µm/px) of the underlying slide in X direction
            - mpp_y (Optional): Physical resolution (µm/px) of the underlying
        overlay_path_key: Key in the batch dict that contains the path to the overlay whole-slide image
        roi: The region of interest geometry.

    Returns:
        A pandas Series containing the overlap ratio of each overlay tile with the ROI.

    Raises:
        ValueError: If neither 'mpp_x' and 'mpp_y' nor 'level' are present in the batch.
    """
    # The overlay is a masked array where the mask is True for pixels outside the ROI.
    overlay = tile_overlay(batch, overlay_path_key, roi)

    return overlay.apply(
        lambda overlay: {
            value.item(): count.item() / overlay.count()
            for value, count in zip(
                *np.unique(overlay.compressed(), return_counts=True), strict=True
            )
        }
    )
