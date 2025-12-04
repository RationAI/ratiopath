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
from ray.data.extensions import TensorArray
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
    mpp_x: pa.Array,
    mpp_y: pa.Array,
) -> dict[str, pa.Array]:
    """Scale overlay tile arguments based on slide and overlay resolutions.

    Args:
        slide: The opened whole-slide image.
        tile_x: X coordinates of the underlying tiles.
        tile_y: Y coordinates of the underlying tiles.
        tile_extent_x: Widths of the underlying tiles.
        tile_extent_y: Heights of the underlying tiles.
        mpp_x: Physical resolution (µm/px) of the underlying slide in X direction.
        mpp_y: Physical resolution (µm/px) of the underlying slide in Y direction.

    Returns:
        A dictionary containing the scaled tile arguments.
    """
    mpp = pd.Series(zip(mpp_x.to_numpy(), mpp_y.to_numpy(), strict=True))  # type: ignore [call-overload]
    level = mpp.apply(slide.closest_level)

    # Determine the overlay resolution
    overlay_mpp = level.apply(slide.slide_resolution)

    x_scaling = pa.array(
        mpp.apply(operator.itemgetter(0)) / overlay_mpp.apply(operator.itemgetter(0))
    )
    y_scaling = pa.array(
        mpp.apply(operator.itemgetter(1)) / overlay_mpp.apply(operator.itemgetter(1))
    )

    def scale(x: pa.Array, scaling: pa.Array) -> pa.Array:
        return pc.max_element_wise(  # type: ignore []
            0,
            pc.round(pc.multiply(x, scaling)).cast(pa.int32()),  # type: ignore []
        )

    return {
        "tile_x": scale(tile_x, x_scaling),
        "tile_y": scale(tile_y, y_scaling),
        "tile_extent_x": scale(tile_extent_x, x_scaling),
        "tile_extent_y": scale(tile_extent_y, y_scaling),
        "level": pa.array(level),
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
    mpp_x: pa.Array,
    mpp_y: pa.Array,
) -> np.ma.MaskedArray:
    """Read overlay tiles for a batch of tiles.

    For each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    Args:
        roi: The region of interest geometry.
        overlay_path: A pyarrow array of whole-slide image paths for the overlays.
        tile_x: A pyarrow array of tile x-coordinates.
        tile_y: A pyarrow array of tile y-coordinates.
        mpp_x: A pyarrow array of physical resolutions (µm/px) of the underlying slide in X direction.
        mpp_y: A pyarrow array of physical resolutions (µm/px) of the underlying slide in Y direction.

    Returns:
        A pyarrow array of masked numpy arrays containing the read overlay tiles.
    """
    # Get ROI bounds
    sx, sy, ex, ey = map(round, roi.bounds)

    # Adjust tile coordinates and extents based on ROI
    w, h = ex - sx, ey - sy
    x = pc.add(tile_x, sx)  # type: ignore []
    y = pc.add(tile_y, sy)  # type: ignore []

    masked_tiles = np.empty(len(tile_x), dtype=object)

    def mask_tile(
        overlay: np.ndarray, sx: int, sy: int, extent_x: int, extent_y: int
    ) -> np.ma.MaskedArray:
        mask = rasterio.features.geometry_mask(
            geometries=[roi],
            out_shape=overlay.shape[:2],
            # Scale and translate to tile coordinates
            transform=Affine.translation(sx, sy)
            * Affine.scale(extent_x / w, extent_y / h),
        )

        return np.ma.masked_array(overlay, mask=np.dstack([mask] * 3))

    for path, group in _pyarrow_group(overlay_path).items():
        assert isinstance(path, str)

        xp = pc.take(x, group)
        yp = pc.take(y, group)

        kwargs = {
            "tile_x": xp,
            "tile_y": yp,
            "tile_extent_x": pa.repeat(w, len(group)),
            "tile_extent_y": pa.repeat(h, len(group)),
            "mpp_x": pc.take(mpp_x, group),
            "mpp_y": pc.take(mpp_y, group),
        }

        # Check if it's an OME-TIFF file
        if path.lower().endswith((".ome.tiff", ".ome.tif")):
            tiles, kwargs = _read_tifffile_overlay(path, kwargs)
        else:
            tiles, kwargs = _read_openslide_overlay(path, kwargs)

        xp = pc.min_element_wise(sx, xp)  # type: ignore []
        yp = pc.min_element_wise(sy, yp)  # type: ignore []
        extent_x_p = pc.take(kwargs["tile_extent_x"], group)
        extent_y_p = pc.take(kwargs["tile_extent_y"], group)

        masked_tiles[group] = [
            mask_tile(
                tile,
                xp[i].as_py(),
                yp[i].as_py(),
                extent_x_p[i].as_py(),
                extent_y_p[i].as_py(),
            )
            for i, tile in enumerate(tiles)
        ]

    return masked_tiles


@udf(return_dtype=DataType(np.ndarray))
def tile_overlay(
    roi: BaseGeometry,
    overlay_path: pa.Array,
    tile_x: pa.Array,
    tile_y: pa.Array,
    mpp_x: pa.Array,
    mpp_y: pa.Array,
) -> pa.Array:
    """Read overlay tiles for a batch of tiles.

    Unfortunately, at the moment we cannot use masked arrays directly in Ray Dataset. So instead,
    we wrap both data and mask into a TensorArray.

    For each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    Args:
        roi: The region of interest geometry.
        overlay_path: A pyarrow array of whole-slide image paths for the overlays.
        tile_x: A pyarrow array of tile x-coordinates.
        tile_y: A pyarrow array of tile y-coordinates.
        mpp_x: A pyarrow array of physical resolutions (µm/px) of the underlying slide in X direction.
        mpp_y: A pyarrow array of physical resolutions (µm/px) of the underlying slide in Y direction.

    Returns:
        A pyarrow array of masked numpy arrays containing the read overlay tiles.
            - The first element is the tile data.
            - The second element is the mask (True for pixels outside the ROI).
    """
    overlays = _tile_overlay(roi, overlay_path, tile_x, tile_y, mpp_x, mpp_y)

    return pa.array(TensorArray([[overlay.data, overlay.mask] for overlay in overlays]))


@udf(return_dtype=DataType(dict))
def tile_overlay_overlap(
    roi: BaseGeometry,
    overlay_path: pa.Array,
    tile_x: pa.Array,
    tile_y: pa.Array,
    mpp_x: pa.Array,
    mpp_y: pa.Array,
) -> pa.Array:
    """Calculate the overlap of each overlay tile with the region of interest (ROI).

    For each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    Args:
        roi: The region of interest geometry.
        overlay_path: A pyarrow array of whole-slide image paths for the overlays.
        tile_x: A pyarrow array of tile x-coordinates.
        tile_y: A pyarrow array of tile y-coordinates.
        mpp_x: A pyarrow array of physical resolutions (µm/px) of the underlying slide in X direction.
        mpp_y: A pyarrow array of physical resolutions (µm/px) of the underlying slide in Y direction.

    Returns:
        A pyarrow array of dictionaries mapping overlay values to their overlap fraction with the ROI.
    """
    # The overlay is a masked array where the mask is True for pixels outside the ROI.
    overlay = _tile_overlay(roi, overlay_path, tile_x, tile_y, mpp_x, mpp_y)

    def overlap_fraction(overlay: np.ma.MaskedArray) -> dict[str, float]:
        """Calculate the overlap fraction of each unique value in the overlay."""
        return {
            # Pyarrow requires string keys in dictionaries
            str(value.item()): count.item() / overlay.count()
            for value, count in zip(
                *np.unique(overlay.compressed(), return_counts=True), strict=True
            )
        }

    overlap_vectorized = np.vectorize(overlap_fraction, otypes=[object])

    return pa.array(overlap_vectorized(overlay))
