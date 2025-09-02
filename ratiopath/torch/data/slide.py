from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ratiopath.torch.data.openslide_tile_reader import OpenSlideTileReader


class SlideDataset(Dataset[pd.Series], OpenSlideTileReader):
    """Dataset for reading tiles from a single slide image.

    This dataset reads tiles from an OpenSlide image. The tiles are specified by a
    DataFrame with columns ["x", "y"]. The RGBA tiles are converted to RGB before
    being returned.

    Attributes:
        slide (Path): Path to the slide image.
        level (int | str): Level of the slide to read. If int, it is used as the level.
            If str, it is used as the column name in the tiles DataFrame.
        tile_extent_x (int | str): Width of the tile. If int, it is used as the width.
            If str, it is used as the column name in the tiles DataFrame.
        tile_extent_y (int | str): Height of the tile. If int, it is used as the height.
            If str, it is used as the column name in the tiles DataFrame.
        tiles (pd.DataFrame): DataFrame with columns ["x", "y"] specifying the tiles
            to be read.
    """

    def __init__(
        self,
        slide_tiles: pd.DataFrame,
        slide_path: str | Path,
        tile_extent: int | tuple[int, int] | str,
        level: int | str | None,
        slide_resolution: float | tuple[float, float] | None,
        background: None | tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Initialize OpenSlideTilesDataset dataset.

        Args:
            slide_tiles: DataFrame with columns ["x", "y"] specifying the tiles to be read.
            slide_path: Path to the slide image.
            tile_extent: Width and height of the tile. If int, it is used as both the
                width and height. If str, it is used as the column name in the tiles
                DataFrame.
            level: Level of the slide to read. If int, it is used as the level. If str,
                it is used as the column name in the tiles DataFrame.
            slide_resolution: Mpp resolution.
            background: Background color for the tiles.
        """
        super().__init__(
            path=slide_path,
            level=level,
            resolution=slide_resolution,
            background=background,
        )

        self.slide_tiles = slide_tiles
        self.tile_extent = tile_extent

    def __len__(self) -> int:
        return len(self.slide_tiles)

    def __getitem__(self, idx: int) -> pd.Series:
        return self.slide_tiles.iloc[idx]

    def get_tile(self, tile_coords: tuple[int, int], tile: pd.Series) -> Image.Image:
        """Returns tile from the slide image at the specified coordinates in RGB format."""
        tile_extent = self._get_from_tile(tile, self.tile_extent)

        return super().get_openslide_tile(
            tile_coords=tile_coords,
            tile_extent=tuple(np.broadcast_to(tile_extent, 2)),  # type: ignore[call-overload]
            tile=tile,
        )
