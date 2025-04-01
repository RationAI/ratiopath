import os
from pathlib import Path
from typing import TypeAlias, TypeVar

import pandas as pd
from PIL import Image

from histopath.openslide import OpenSlide

TodoMultI: TypeAlias = tuple[int, ...]
TodoMultF: TypeAlias = tuple[float, ...]
T = TypeVar("T")


class BaseReader:
    def __init__(
        self,
        path: str | Path,
        level: int | str | None = None,
        resolution: TodoMultF | None = None,
        background: None | tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        # Check if one of level or slide_resolution is provided
        assert level is not None or resolution is not None, (
            "Either level or resolution must be provided"
        )
        assert level is None or resolution is None, (
            "Only one of level or resolution must be provided"
        )

        assert os.path.exists(path), f"Path {path} does not exist"

        self.path = Path(path)
        self.background = background

        # Assign level based on the provided arguments
        if resolution is not None:
            with OpenSlide(self.path) as slide:
                self.level = slide.closest_level(resolution)
        else:
            self.level: str | int = level  # type: ignore

    def get_openslide_tile(
        self, tile_coords: TodoMultI, tile_extent: TodoMultI, tile: pd.Series
    ) -> Image:
        """Returns tile from the slide image at the specified coordinates in RGB format."""
        level = self._get_from_tile(tile, self.level)

        with OpenSlide(self.path) as slide:
            rgba_region = slide.read_region(
                slide.adjust_read_coords(tile_coords, level), level, tile_extent
            )

            if self.background is not None:
                # Create a new image with the background color
                background = Image.new("RGBA", rgba_region.size, self.background)
                # Composite the region with the background
                rgba_region = Image.alpha_composite(background, rgba_region)

        return rgba_region.convert("RGB")

    def _get_from_tile(self, tile: pd.Series, key: T | str) -> T:
        return tile[key] if isinstance(key, str) else key  # type: ignore
