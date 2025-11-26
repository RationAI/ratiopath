from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ratiopath.tiling.utils import _read_openslide_tiles, _read_tifffile_tiles


def _scale_overlay_args(
    df: pd.DataFrame,
    overlay_mpp: pd.DataFrame,  # It is indeed Series[tuple[float, float]], but it is not valid Series type annotation
    levels: pd.Series,
) -> pd.DataFrame:
    """Adjust overlay tile arguments based on slide and overlay resolutions."""
    scaling_factor_x: pd.Series = df["mpp_x"] / overlay_mpp.apply(lambda mpp: mpp[0])  # type: ignore[call-arg]
    scaling_factor_y: pd.Series = df["mpp_y"] / overlay_mpp.apply(lambda mpp: mpp[1])  # type: ignore[call-arg]

    def scale(df: pd.Series, scale: pd.Series) -> pd.Series:
        return (df * scale).round(0).astype(int)

    df["tile_x"] = scale(df["tile_x"], scaling_factor_x)
    df["tile_y"] = scale(df["tile_y"], scaling_factor_y)
    df["tile_extent_x"] = scale(df["tile_extent_x"], scaling_factor_x)
    df["tile_extent_y"] = scale(df["tile_extent_y"], scaling_factor_y)
    df["level"] = levels

    return df


def _read_openslide_overlay(path: str, df: pd.DataFrame) -> pd.Series:
    """Read batch of overlays from a whole-slide image using OpenSlide."""
    from ratiopath.openslide import OpenSlide

    with OpenSlide(path) as slide:
        levels = df.apply(lambda row: slide.closest_level((row["mpp_x"], row["mpp_y"])))
        overlay_mpp = df.apply(lambda row: slide.slide_resolution(levels[row.name]))

        return _read_openslide_tiles(
            slide, _scale_overlay_args(df, overlay_mpp, levels)
        )


def _read_tifffile_overlay(path: str, df: pd.DataFrame) -> pd.Series:
    """Read batch of overlays from an OME-TIFF file using tifffile."""
    from ratiopath.tifffile import TiffFile

    with TiffFile(path) as slide:
        levels = df.apply(lambda row: slide.closest_level((row["mpp_x"], row["mpp_y"])))
        overlay_mpp = df.apply(lambda row: slide.slide_resolution(levels[row.name]))

        return _read_tifffile_tiles(slide, _scale_overlay_args(df, overlay_mpp, levels))


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
            - mpp_x: Physical resolution (µm/px) of the underlying slide in X direction
            - mpp_y: Physical resolution (µm/px) of the underlying slide in Y direction
            - overlay_path: Path to the overlay whole-slide image

    Returns:
        A pandas Series containing the numpy array overlays.
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

    The ROI is cumputed as a fractional offset and extent relative to the tile_extent.

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
        batch["tile_x"] = batch["tile_x"] - (roi_offset_x * batch["tile_extent_x"])
        batch["tile_y"] = batch["tile_y"] - (roi_offset_y * batch["tile_extent_y"])
        batch["tile_extent_x"] = batch["tile_extent_x"] * roi_extent_x
        batch["tile_extent_y"] = batch["tile_extent_y"] * roi_extent_y
        return batch

    return wrapper


def tile_overlay(
    overlay_path_key: str,
    store_key: str,
    roi: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Constructs a overaly function that reads overlay tiles for a batch of tiles and stores them in the batch under the given key.

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
                - mpp_x: Physical resolution (µm/px) of the underlying slide in
                - mpp_y: Physical resolution (µm/px) of the underlying slide in Y direction
        Returns:
            The input batch (dict) with a new key `{store_key}` containing the overlay tiles.
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
                - mpp_x: Physical resolution (µm/px) of the underlying slide in
                - mpp_y: Physical resolution (µm/px) of the underlying slide in
        Returns:
            The input batch (dict) with a new key `{store_key}` containing the overlay overlaps.
        """
        df = roi(pd.DataFrame(batch)).rename(columns={overlay_path_key: "overlay_path"})

        batch[store_key] = (
            _tile_overlay(df)
            .apply(
                lambda overlay: {
                    value.item(): count.item() / (overlay.shape[:2].prod())
                    for value, count in zip(
                        *np.unique(overlay, return_counts=True), strict=True
                    )
                }
            )
            .tolist()
        )
        return batch

    return wrapper
