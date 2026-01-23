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
from ratiopath.tiling.types import ReadOverlaysArguments, ReadTilesArguments
from ratiopath.tiling.utils import (
    _pyarrow_group_indices,
    _read_openslide_tiles,
    _read_tifffile_tiles,
)


def _scale_overlay(
    slide: OpenSlide | TiffFile,
    tile_x: pa.IntegerArray,
    tile_y: pa.IntegerArray,
    tile_extent_x: pa.IntegerArray,
    tile_extent_y: pa.IntegerArray,
    mpp_x: pa.FloatArray,
    mpp_y: pa.FloatArray,
) -> ReadTilesArguments:
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
        mpp.apply(operator.itemgetter(0)) / overlay_mpp.apply(operator.itemgetter(0)),
        pa.float32(),
    )
    y_scaling = pa.array(
        mpp.apply(operator.itemgetter(1)) / overlay_mpp.apply(operator.itemgetter(1)),
        pa.float32(),
    )

    def scale(x: pa.IntegerArray, scaling: pa.FloatArray) -> pa.IntegerArray:
        return pc.max_element_wise(
            pa.scalar(0),
            pc.round(pc.multiply(x, scaling)).cast(pa.int32()),
        )  # type: ignore [return-value]

    return {
        "tile_x": scale(tile_x, x_scaling),
        "tile_y": scale(tile_y, y_scaling),
        "tile_extent_x": scale(tile_extent_x, x_scaling),
        "tile_extent_y": scale(tile_extent_y, y_scaling),
        "level": pa.array(level, pa.int8()),
    }


def _read_openslide_overlay(
    path: str, kwargs: ReadOverlaysArguments
) -> tuple[np.ndarray, ReadTilesArguments]:
    """Read batch of overlays from a whole-slide image using OpenSlide."""
    with OpenSlide(path) as slide:
        new_kwargs = _scale_overlay(slide, **kwargs)
        return _read_openslide_tiles(slide, **new_kwargs), new_kwargs


def _read_tifffile_overlay(
    path: str, kwargs: ReadOverlaysArguments
) -> tuple[np.ndarray, ReadTilesArguments]:
    """Read batch of overlays from an OME-TIFF file using tifffile."""
    with TiffFile(path) as slide:
        new_kwargs = _scale_overlay(slide, **kwargs)
        return _read_tifffile_tiles(slide, **new_kwargs), new_kwargs


def _tile_overlay(
    roi: BaseGeometry,
    overlay_path: pa.StringArray,
    tile_x: pa.IntegerArray,
    tile_y: pa.IntegerArray,
    mpp_x: pa.FloatArray,
    mpp_y: pa.FloatArray,
) -> np.ma.MaskedArray:
    """Read overlay tiles for a batch of tiles.

    For each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    The region of interest (ROI) geometry is treated in the same image space (resolution) as the underlying slide tiles.
    The region can be an arbitrary polygon. However, a bounding box of the region is used for reading overlay tiles
    and then masked to respect the region defined by provided overlay.

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
    # Pre-calculate ROI constants
    roi_min_x, roi_min_y, roi_max_x, roi_max_y = map(round, roi.bounds)
    roi_width = roi_max_x - roi_min_x
    roi_height = roi_max_y - roi_min_y

    tile_x = pc.add(tile_x, pa.scalar(roi_min_x))
    tile_y = pc.add(tile_y, pa.scalar(roi_min_y))

    # Pre-allocate output array (Object array to hold MaskedArrays)
    masked_tiles = np.empty(len(tile_x), dtype=object)

    def mask_tile(
        overlay: np.ndarray, x: int, y: int, extent_x: int, extent_y: int
    ) -> np.ma.MaskedArray:
        mask = rasterio.features.geometry_mask(
            geometries=[roi],
            out_shape=overlay.shape[:2],
            # Scale and translate to tile coordinates
            transform=Affine.translation(x, y)
            * Affine.scale(extent_x / roi_width, extent_y / roi_height),
        )

        return np.ma.masked_array(overlay, mask=np.dstack([mask] * overlay.shape[2]))

    for path, group in _pyarrow_group_indices(overlay_path).items():
        assert isinstance(path, str)

        batch_x = pc.take(tile_x, group)
        batch_y = pc.take(tile_y, group)

        overlay_kwargs = ReadOverlaysArguments(
            tile_x=batch_x,
            tile_y=batch_y,
            tile_extent_x=pa.repeat(roi_width, len(group)),
            tile_extent_y=pa.repeat(roi_height, len(group)),
            mpp_x=pc.take(mpp_x, group),
            mpp_y=pc.take(mpp_y, group),
        )

        # Check if it's an OME-TIFF file
        if path.lower().endswith((".ome.tiff", ".ome.tif")):
            tiles, tile_kwargs = _read_tifffile_overlay(path, overlay_kwargs)
        else:
            tiles, tile_kwargs = _read_openslide_overlay(path, overlay_kwargs)

        batch_x: pa.IntegerArray = pc.min_element_wise(pa.scalar(roi_min_x), batch_x)  # type: ignore [no-redef]
        batch_y: pa.IntegerArray = pc.min_element_wise(pa.scalar(roi_min_y), batch_y)  # type: ignore [no-redef]

        masked_tiles[group] = [
            mask_tile(
                tile,
                batch_x[i].as_py(),
                batch_y[i].as_py(),
                tile_kwargs["tile_extent_x"][i].as_py(),
                tile_kwargs["tile_extent_y"][i].as_py(),
            )
            for i, tile in enumerate(tiles)
        ]

    return masked_tiles  # type: ignore [return-value]


@udf(return_dtype=DataType(np.ndarray))
def tile_overlay(
    roi: BaseGeometry,
    overlay_path: pa.StringArray,
    tile_x: pa.IntegerArray,
    tile_y: pa.IntegerArray,
    mpp_x: pa.FloatArray,
    mpp_y: pa.FloatArray,
) -> pa.Array:
    """Read overlay tiles for a batch of tiles.

    For each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    The region of interest (ROI) geometry is treated in the same image space (resolution) as the underlying slide tiles.
    The region can be an arbitrary polygon. However, a bounding box of the region is used for reading overlay tiles
    and then masked to respect the region defined by provided overlay.

    Unfortunately, at the moment we cannot use masked arrays directly in Ray Dataset. So instead of a numpy masked array,
    we provide the data and the mask as 2 separate arrays. The implementation is handled via TensorArray.

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
    overlay_path: pa.StringArray,
    tile_x: pa.IntegerArray,
    tile_y: pa.IntegerArray,
    mpp_x: pa.FloatArray,
    mpp_y: pa.FloatArray,
) -> pa.MapArray:
    """Calculate the overlap of each overlay tile with the region of interest (ROI).

    For each overlay path the corresponding whole-slide image is opened (OpenSlide or OME-TIFF).
    The overlay is accessed at the slide level closest to each tile's mpp and the tile
    coordinates/extents are scaled to that level before reading.

    The region of interest (ROI) geometry is treated in the same image space (resolution) as the underlying slide tiles.
    The region can be an arbitrary polygon.

    The Pyarrow array that is used inside ray dataset stores data in array like dictionary.
    This results in all rows having same set of keys and missing keys are filled with Nones.
    Furthermore Pyarrow only support string keys in dictionaries. Therefore the keys in the
    resulting dictionary are string representations of the overlay values.

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

    return pa.array(overlap_vectorized(overlay))  # type: ignore [return-value]
