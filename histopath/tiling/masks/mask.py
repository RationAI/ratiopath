from pathlib import Path
from typing import Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

from histopath.openslide import OpenSlide

RowData: TypeAlias = dict[str, Any]
TranslateDict: TypeAlias = dict[str, str] | None

T = TypeVar("T")
Dims: TypeAlias = tuple[int, int]


def read_tile(
    slide_path: str | Path,
    tile_coords: Dims,
    tile_extent: Dims,
    level: int | None = None,
    resolution: Dims | None = None,
) -> NDArray:
    assert level is not None or resolution is not None, (
        "Either level or resolution must be provided"
    )
    assert level is None or resolution is None, (
        "Only one of level or resolution must be provided"
    )

    with OpenSlide(slide_path) as slide:
        if level is None:
            level = slide.closest_level(resolution)

        slide_region = slide.read_region(
            slide.get_tile_dimensions(tile_coords, level), level, tile_extent
        )

        return slide_region.convert("RGB").to_numpy()


def tile_overlay(
    overlay_path: str | Path,
    resolution: Dims,
    roi_coords: Dims,
    roi_extent: Dims,
) -> NDArray:
    with OpenSlide(overlay_path) as overlay:
        level = overlay.closest_level(resolution)
        overlay_resolution = overlay.slide_resolution(level)

        resolution_factor = np.asarray(overlay_resolution) / np.asarray(resolution)

        roi_coords = tuple(
            np.round(np.asarray(roi_coords) * resolution_factor).astype(int)
        )
        roi_extent = tuple(
            np.round(np.asarray(roi_extent) * resolution_factor).astype(int)
        )

        overlay_region = overlay.read_region(
            overlay.get_tile_dimensions(roi_coords, level),
            level,
            roi_extent,
        )

        return overlay_region.convert("RGB").to_numpy()


def relative_tile_overlay(
    overlay_path: str | Path,
    resolution: Dims,
    tile_coords: Dims,
    relative_roi_coords: Dims,
    roi_extent: Dims,
) -> NDArray:
    return tile_overlay(
        overlay_path=overlay_path,
        resolution=resolution,
        roi_coords=tuple(np.asarray(tile_coords) + np.asarray(relative_roi_coords)),
        roi_extent=roi_extent,
    )
