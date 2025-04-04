from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from histopath.openslide import OpenSlide


def read_tile(
    slide_path: str | Path,
    tile_coords: tuple[int, int],
    tile_extent: tuple[int, int],
    level: int | None = None,
    resolution: float | tuple[float, float] | None = None,
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
            slide.adjust_read_coords(tile_coords, level), level, tile_extent
        )

        return slide_region.convert("RGB").to_numpy()


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

        overlay_region = overlay.read_region(
            overlay.get_tile_dimensions(roi_coords, level),
            level,
            roi_extent,
        )

        return overlay_region.convert("RGB").to_numpy()


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
