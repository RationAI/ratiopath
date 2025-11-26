from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

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
    scaling_factor_x = mpp.apply(lambda mpp: mpp[0]) / overlay_mpp.apply(
        lambda mpp: mpp[0]
    )
    scaling_factor_y = mpp.apply(lambda mpp: mpp[1]) / overlay_mpp.apply(
        lambda mpp: mpp[1]
    )

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


def _tile_overlay(batch: pd.DataFrame) -> pd.Series:
    """Read overlay tiles for a batch of tiles.

    For each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    Args:
        batch (pandas.DataFrame):
            - tile_x: X coordinates of the underlying tiles
            - tile_y: Y coordinates of the underlying
            - tile_extent_x: Widths of the underlying tiles
            - tile_extent_y: Heights of the underlying tiles
            - overlay_path: Path to the overlay whole-slide image
            - level (Optional): Level of the underlying slide
            - mpp_x (Optional): Physical resolution (µm/px) of the underlying slide in X direction
            - mpp_y (Optional): Physical resolution (µm/px) of the underlying slide in Y direction

    Returns:
        A pandas Series containing the numpy array overlays.

    Raises:
        ValueError: If neither 'mpp_x' and 'mpp_y' nor 'level' are present in the DataFrame after applying the ROI function.
    """
    tiles = pd.Series(index=batch.index, dtype=object)

    for path, group in batch.groupby("overlay_path"):
        assert isinstance(path, str)
        if path.lower().endswith((".ome.tiff", ".ome.tif")):
            tiles.loc[group.index] = _read_tifffile_overlay(path, group)
        else:
            tiles.loc[group.index] = _read_openslide_overlay(path, group)

    return tiles


def overlay_roi(
    roi_offset_x: float = 0,
    roi_offset_y: float = 0,
    roi_extent_x: float = 1,
    roi_extent_y: float = 1,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Adjust tile coordinates and extents to account for a region of interest (ROI) within the overlay.

    The ROI is computed as a fractional offset and extent relative to the tile_extent.

    Args:
        roi_offset_x: Fractional X offset of the ROI within the overlay. Default is 0.
        roi_offset_y: Fractional Y offset of the ROI within the overlay. Default is 0.
        roi_extent_x: Fractional width of the ROI within the overlay. Default is 1.
        roi_extent_y: Fractional height of the ROI within the overlay. Default is 1.

    Returns:
        Function that adjusts tile coordinates and extents in the batch to account for the ROI.
    """

    def wrapper(batch: pd.DataFrame) -> pd.DataFrame:
        """Adjust tile coordinates and extents to account for a region of interest (ROI) within the overlay.

        Args:
            batch (pandas.DataFrame):
                - tile_x: X coordinates of the underlying tiles
                - tile_y: Y coordinates of the underlying tiles
                - tile_extent_x: Widths of the underlying tiles
                - tile_extent_y: Heights of the underlying tiles

        Returns:
            The input batch with adjusted tile coordinates and extents.
        """
        batch["tile_x"] = batch["tile_x"] + (roi_offset_x * batch["tile_extent_x"])
        batch["tile_y"] = batch["tile_y"] + (roi_offset_y * batch["tile_extent_y"])
        batch["tile_extent_x"] = batch["tile_extent_x"] * roi_extent_x
        batch["tile_extent_y"] = batch["tile_extent_y"] * roi_extent_y
        return batch

    return wrapper


def tile_overlay(
    overlay_path_key: str,
    store_key: str,
    roi: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Constructs an overaly function that reads overlay tiles for a batch of tiles and stores them in the batch under the given key.

    The overlay function first applies the given ROI function to adjust tile coordinates and extents.
    Then, for each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    Args:
        overlay_path_key: Key in the batch dict that contains the path to the overlay whole-slide image
        store_key: The key under which to store the resulting overlay tiles in the batch.
        roi: A function that adjusts the batch (pandas DataFrame) to account for a region of interest (ROI). Defaults to an identity function over the batch.

    Returns:
        Function that adds overlay tiles to the batch under the specified key.
    """

    def wrapper(batch: dict[str, Any]) -> dict[str, Any]:
        """Reads overlay tiles for a batch of tiles and stores them in the batch under the given key.

        The function first applies the given ROI function to adjust tile coordinates and extents.
        Then, for each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
        The overlay is accessed at the slide level closest to each tile's mpp and the tile
        coordinates/extents are scaled to that level before reading.

        Args:
            batch (dict): Upon calliing the roi function the pandas DataFrame must contain:
                - tile_x: X coordinates of the underlying tiles
                - tile_y: Y coordinates of the underlying
                - tile_extent_x: Widths of the underlying tiles
                - tile_extent_y: Heights of the underlying tiles
                - level (Optional): Level of the underlying slide
                - mpp_x (Optional): Physical resolution (µm/px) of the underlying slide in X direction
                - mpp_y (Optional): Physical resolution (µm/px) of the underlying slide in Y direction

        Returns:
            The input batch (dict) with a new key `{store_key}` containing the overlay tiles.

        Raises:
            ValueError: If neither 'mpp_x' and 'mpp_y' nor 'level' are present in the DataFrame after applying the ROI function.
        """
        df = roi(pd.DataFrame(batch)).rename(columns={overlay_path_key: "overlay_path"})
        batch[store_key] = _tile_overlay(df).tolist()
        return batch

    return wrapper


def tile_overlay_overlap(
    overlay_path_key: str,
    store_key: str,
    roi: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Constructs a function that calculates the overlap of each unique value in the overlay tiles and stores the result in the batch under the specified overlay key.

    The overlay function first applies the given ROI function to adjust tile coordinates and extents.
    Then, for each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    Args:
        overlay_path_key: Key in the batch dict that contains the path to the overlay whole-slide image
        store_key: The key under which to store the resulting overlay tiles in the batch.
        roi: A function that adjusts the batch (pandas DataFrame) to account for a region of interest (ROI). Defaults to an identity function over the batch.

    Returns:
        Function that adds overlay overlaps to the batch under the specified key.
    """

    def wrapper(batch: dict[str, Any]) -> dict[str, Any]:
        """Calculates the overlap of each unique value in the overlay tiles and stores the result in the batch under the specified overlay key.

        The function first applies the given ROI function to adjust tile coordinates and extents.
        Then, for each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
        The overlay is accessed at the slide level closest to each tile's mpp and the tile
        coordinates/extents are scaled to that level before reading.

        Args:
            batch (dict): Upon calliing the roi function the pandas DataFrame must contain:
                - tile_x: X coordinates of the underlying tiles
                - tile_y: Y coordinates of the underlying
                - tile_extent_x: Widths of the underlying tiles
                - tile_extent_y: Heights of the underlying tiles
                - level (Optional): Level of the underlying slide
                - mpp_x (Optional): Physical resolution (µm/px) of the underlying slide in X direction
                - mpp_y (Optional): Physical resolution (µm/px) of the underlying slide in Y direction

        Returns:
            The input batch (dict) with a new key `{store_key}` containing the overlay overlaps.

        Raises:
            ValueError: If neither 'mpp_x' and 'mpp_y' nor 'level' are present in the DataFrame after applying the ROI function.
        """
        df = roi(pd.DataFrame(batch)).rename(columns={overlay_path_key: "overlay_path"})

        batch[store_key] = (
            _tile_overlay(df)
            .apply(
                lambda overlay: {
                    value.item(): count.item() / np.prod(overlay.shape[:2])
                    for value, count in zip(
                        *np.unique(overlay, return_counts=True), strict=True
                    )
                }
            )
            .tolist()
        )
        return batch

    return wrapper
