from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ratiopath.openslide import OpenSlide


def read_tile(
    slide_path: str | Path,
    tile_coords: tuple[int, int],
    tile_extent: tuple[int, int],
    level: int | None = None,
    resolution: float | tuple[float, float] | None = None,
) -> NDArray:
    assert (level is None) != (resolution is None), (
        "Either level or resolution must be provided"
    )

    with OpenSlide(slide_path) as slide:
        if resolution is not None:
            level = slide.closest_level(resolution)

        slide_region = slide.read_region_relative(tile_coords, level, tile_extent)  # type: ignore[arg-type]

        return np.asarray(slide_region.convert("RGB"))


def tile_overlay(
    overlay_path: str | Path,
    resolution: tuple[int, int],
    roi_coords: tuple[int, int],
    roi_extent: tuple[int, int],
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

        overlay_region = overlay.read_region_relative(roi_coords, level, roi_extent)

        return np.asarray(overlay_region.convert("RGB"))


def relative_tile_overlay(
    overlay_path: str | Path,
    resolution: tuple[int, int],
    tile_coords: tuple[int, int],
    relative_roi_coords: tuple[int, int],
    roi_extent: tuple[int, int],
) -> NDArray:
    return tile_overlay(
        overlay_path=overlay_path,
        resolution=resolution,
        roi_coords=tuple(np.asarray(tile_coords) + np.asarray(relative_roi_coords)),
        roi_extent=roi_extent,
    )
