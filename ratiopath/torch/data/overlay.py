from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ratiopath.openslide import OpenSlide
from ratiopath.torch.data.openslide_tile_reader import OpenSlideTileReader


class Overlay(OpenSlideTileReader):
    def __init__(
        self,
        overlay_path: str | Path,
        tile_extent: str | int | tuple[int, int],
        slide_resolution: float | tuple[float, float],
        overlay_resolution: float | tuple[float, float] | None = None,
        level: int | str | None = None,
        overlay_mode: str = "1",
        resample_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            path=overlay_path,
            level=level,
            resolution=overlay_resolution,
            background=None,
        )

        self.slide_resolution = slide_resolution
        self.tile_extent = tile_extent
        self.resample_kwargs = resample_kwargs
        self.overlay_mode = overlay_mode

    def __getitem__(self, coords: tuple[int, int], tile: pd.Series) -> Image.Image:
        level = self._get_from_tile(tile, self.level)
        tile_extent = self._get_from_tile(tile, self.tile_extent)

        with OpenSlide(self.path) as overlay:
            resolution = overlay.slide_resolution(level)

        resolution_factor = np.asarray(resolution) / np.asarray(self.slide_resolution)

        overlay_tile = self.get_openslide_tile(
            tile_coords=tuple(np.round(np.asarray(coords) * resolution_factor)),
            tile_extent=tuple(np.round(np.asarray(tile_extent) * resolution_factor)),
            tile=tile,
        )

        return overlay_tile.convert(self.overlay_mode).resize(
            np.broadcast_to(tile_extent, 2),  # type: ignore[call-overload]
            **(self.resample_kwargs if self.resample_kwargs else {}),
        )
