import operator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import rasterio
import rasterio.features
from rasterio.transform import Affine
from ray.data.datatype import DataType
from ray.data.expressions import udf
from shapely.geometry.base import BaseGeometry

from ratiopath.openslide import OpenSlide
from ratiopath.tifffile import TiffFile
from ratiopath.tiling.utils import (
    _pyarrow_group,
    _read_openslide_tiles,
    _read_tifffile_tiles,
)


def _scale_overlay(
    slide: OpenSlide | TiffFile,
    tile_x: pa.Array,
    tile_y: pa.Array,
    tile_extent_x: pa.Array,
    tile_extent_y: pa.Array,
    level: pa.Array | None = None,
    mpp_x: pa.Array | None = None,
    mpp_y: pa.Array | None = None,
) -> dict[str, pa.Array]:
    """Scale overlay tile arguments based on slide and overlay resolutions.

    Args:
        slide: The opened whole-slide image.
        tile_x: X coordinates of the underlying tiles.
        tile_y: Y coordinates of the underlying tiles.
        tile_extent_x: Widths of the underlying tiles.
        tile_extent_y: Heights of the underlying tiles.
        level: (Optional) Level of the underlying slide.
        mpp_x: (Optional) Physical resolution (µm/px) of the underlying slide in X direction.
        mpp_y: (Optional) Physical resolution (µm/px) of the underlying slide in Y direction.

    Returns:
        A dictionary containing the scaled tile arguments.

    Raises:
        ValueError: If neither 'mpp_x' and 'mpp_y' nor 'level' are present.
    """
    # Determine resolution of the underlying tile
    if mpp_x is None or mpp_y is None:
        if level is None:
            raise ValueError(
                "DataFrame must contain 'mpp_x' and 'mpp_y' columns or 'level' column."
            )
        mpp = level.to_pandas().apply(slide.slide_resolution)
        level_pd = level.to_pandas()
    else:
        mpp = pd.Series(zip(mpp_x.tolist(), mpp_y.tolist(), strict=True))  # type: ignore [call-overload]
        level_pd = mpp.apply(slide.closest_level)

    # Determine the overlay resolution
    overlay_mpp = level_pd.apply(slide.slide_resolution)

    x_scaling = pa.array(
        mpp.apply(operator.itemgetter(0)) / overlay_mpp.apply(operator.itemgetter(0))
    )
    y_scaling = pa.array(
        mpp.apply(operator.itemgetter(1)) / overlay_mpp.apply(operator.itemgetter(1))
    )

    def scale(x: pa.Array, scaling: pa.Array) -> pa.Array:
        return pc.round(pc.multiply(x, scaling))  # type: ignore []

    return {
        "tile_x": scale(tile_x, x_scaling),
        "tile_y": scale(tile_y, y_scaling),
        "tile_extent_x": scale(tile_extent_x, x_scaling),
        "tile_extent_y": scale(tile_extent_y, y_scaling),
        "level": pa.array(level_pd),
    }


def _read_openslide_overlay(
    path: str, kwargs: dict[str, pa.Array]
) -> tuple[np.ndarray, dict[str, pa.Array]]:
    """Read batch of overlays from a whole-slide image using OpenSlide."""
    with OpenSlide(path) as slide:
        kwargs = _scale_overlay(slide, **kwargs)
        return _read_openslide_tiles(slide, **kwargs), kwargs


def _read_tifffile_overlay(
    path: str, kwargs: dict[str, pa.Array]
) -> tuple[np.ndarray, dict[str, pa.Array]]:
    """Read batch of overlays from an OME-TIFF file using tifffile."""
    with TiffFile(path) as slide:
        kwargs = _scale_overlay(slide, **kwargs)
        return _read_tifffile_tiles(slide, **kwargs), kwargs


def _tile_overlay(
    roi: BaseGeometry,
    overlay_path: pa.Array,
    tile_x: pa.Array,
    tile_y: pa.Array,
    level: pa.Array | None = None,
    mpp_x: pa.Array | None = None,
    mpp_y: pa.Array | None = None,
) -> pa.Array[np.ma.MaskedArray]:
    """Read overlay tiles for a batch of tiles.

    For each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    Args:
        roi: The region of interest geometry.
        overlay_path: A pyarrow array of whole-slide image paths for the overlays.
        tile_x: A pyarrow array of tile x-coordinates.
        tile_y: A pyarrow array of tile y-coordinates.
        level: (Optional) A pyarrow array of slide levels to read the tiles from.
        mpp_x: (Optional) A pyarrow array of physical resolutions (µm/px) of the underlying slide in X direction.
        mpp_y: (Optional) A pyarrow array of physical resolutions (µm/px) of the underlying slide in Y direction.

    Returns:
        A pyarrow array of masked numpy arrays containing the read overlay tiles.

    Raises:
        ValueError: If neither 'mpp_x' and 'mpp_y' nor 'level' are present.
    """
    # Get ROI bounds
    sx, sy, ex, ey = map(round, roi.bounds)

    # Adjust tile coordinates and extents based on ROI
    w, h = ex - sx, ey - sy
    x = pc.subtract(tile_x, sx)  # type: ignore []
    y = pc.subtract(tile_y, sy)  # type: ignore []

    masked_tiles = np.empty(len(tile_x), dtype=object)

    def mask_tile(
        overlay: np.ndarray, sx: int, sy: int, extent_x: int, extent_y: int
    ) -> np.ma.MaskedArray:
        mask = rasterio.features.geometry_mask(
            geometries=[roi],
            out_shape=overlay.shape[:2],
            # Scale and translate to tile coordinates - reflect ROI cropping
            transform=Affine.translation(-sx, -sy)
            * Affine.scale(extent_x / w, extent_y / h),
        )
        return np.ma.masked_array(overlay, mask=mask)

    create_masked_array = np.vectorize(mask_tile, otypes=[np.ma.MaskedArray])

    for path, group in _pyarrow_group(overlay_path).items():
        assert isinstance(path, str)

        x = pc.take(tile_x, group)
        y = pc.take(tile_y, group)

        kwargs = {
            "path": path,
            "tile_x": x,
            "tile_y": y,
            "tile_extent_x": pa.repeat(w, len(group)),
            "tile_extent_y": pa.repeat(h, len(group)),
            "level": level and pc.take(level, group),
            "mpp_x": mpp_x and pc.take(mpp_x, group),
            "mpp_y": mpp_y and pc.take(mpp_y, group),
        }

        # Check if it's an OME-TIFF file
        if path.lower().endswith((".ome.tiff", ".ome.tif")):
            tiles, kwargs = _read_tifffile_overlay(path, kwargs)
        else:
            tiles, kwargs = _read_openslide_overlay(path, kwargs)

        masked_tiles[group] = create_masked_array(
            tiles,
            # Adjust coordinates for masking
            pc.min_element_wise(-sx, pc.add(-sx, x)).to_numpy(),  # type: ignore []
            pc.min_element_wise(-sy, pc.add(-sy, y)).to_numpy(),  # type: ignore []
            pc.take(kwargs["tile_extent_x"], group).to_numpy(),
            pc.take(kwargs["tile_extent_y"], group).to_numpy(),
        )

    return pa.array(masked_tiles, type=np.ma.MaskedArray)


@udf(return_dtype=DataType(np.ma.MaskedArray))
def tile_overlay(
    roi: BaseGeometry,
    overlay_path: pa.Array,
    tile_x: pa.Array,
    tile_y: pa.Array,
    level: pa.Array | None = None,
    mpp_x: pa.Array | None = None,
    mpp_y: pa.Array | None = None,
) -> pa.Array[np.ma.MaskedArray]:
    """Read overlay tiles for a batch of tiles.

    For each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    Args:
        roi: The region of interest geometry.
        overlay_path: A pyarrow array of whole-slide image paths for the overlays.
        tile_x: A pyarrow array of tile x-coordinates.
        tile_y: A pyarrow array of tile y-coordinates.
        level: (Optional) A pyarrow array of slide levels to read the tiles from.
        mpp_x: (Optional) A pyarrow array of physical resolutions (µm/px) of the underlying slide in X direction.
        mpp_y: (Optional) A pyarrow array of physical resolutions (µm/px) of the underlying slide in Y direction.

    Returns:
        A pyarrow array of masked numpy arrays containing the read overlay tiles.

    Raises:
        ValueError: If neither 'mpp_x' and 'mpp_y' nor 'level' are present.
    """
    return _tile_overlay(roi, overlay_path, tile_x, tile_y, level, mpp_x, mpp_y)


@udf(return_dtype=DataType(object))
def tile_overlay_overlap(
    roi: BaseGeometry,
    overlay_path: pa.Array,
    tile_x: pa.Array,
    tile_y: pa.Array,
    level: pa.Array | None = None,
    mpp_x: pa.Array | None = None,
    mpp_y: pa.Array | None = None,
) -> pa.Array[dict[int, float]]:
    """Calculate the overlap of each overlay tile with the region of interest (ROI).

    For each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    Args:
        roi: The region of interest geometry.
        overlay_path: A pyarrow array of whole-slide image paths for the overlays.
        tile_x: A pyarrow array of tile x-coordinates.
        tile_y: A pyarrow array of tile y-coordinates.
        level: (Optional) A pyarrow array of slide levels to read the tiles from.
        mpp_x: (Optional) A pyarrow array of physical resolutions (µm/px) of the underlying slide in X direction.
        mpp_y: (Optional) A pyarrow array of physical resolutions (µm/px) of the underlying slide in Y direction.

    Returns:
        A pyarrow array of dictionaries mapping overlay values to their overlap fraction with the ROI.

    Raises:
        ValueError: If neither 'mpp_x' and 'mpp_y' nor 'level' are present.
    """
    # The overlay is a masked array where the mask is True for pixels outside the ROI.
    overlay = _tile_overlay(roi, overlay_path, tile_x, tile_y, level, mpp_x, mpp_y)

    def overlap_fraction(overlay: np.ma.MaskedArray) -> dict[int, float]:
        """Calculate the overlap fraction of each unique value in the overlay."""
        return {
            value.item(): count.item() / overlay.count()
            for value, count in zip(
                *np.unique(overlay.compressed(), return_counts=True), strict=True
            )
        }

    overlap_vectorized = np.vectorize(overlap_fraction, otypes=[object])

    return pa.array(overlap_vectorized(overlay))
