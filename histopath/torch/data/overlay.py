from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from PIL import MODES, Image

from histopath.openslide import OpenSlide
from histopath.torch.data.base import BaseReader

TodoMultI: TypeAlias = tuple[int, ...]
TodoMultF: TypeAlias = tuple[float, ...]


class Overlay(BaseReader):
    def __init__(
        self,
        overlay_path: str | Path,
        tile_extent: str | TodoMultI,
        slide_resolution: TodoMultF,
        overlay_resolution: TodoMultF | None = None,
        level: int | str | None = None,
        overlay_mode: Literal[*MODES] = "1",
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

    def __getitem__(self, coords: tuple[int, int], tile: pd.Series) -> Image:
        level = self._get_from_tile(tile, self.level)
        tile_extent = self._get_from_tile(tile, self.tile_extent)

        with OpenSlide(self.path) as overlay:
            resolution = overlay.slide_resolution(level)

        resolution_factor = np.asarray(resolution) / np.asarray(self.slide_resolution)

        overlay_tile = self.get_openslide_tile(
            tile_coords=np.round(np.asarray(coords) * resolution_factor).astype(int),
            tile_extent=np.round(np.asarray(tile_extent) * resolution_factor).astype(
                int
            ),
            tile=tile,
        )

        return overlay_tile.convert(self.overlay_mode).resize(
            tile_extent,
            **(self.resample_kwargs if self.resample_kwargs else {}),
        )
