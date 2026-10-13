from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image, ImageDraw


if TYPE_CHECKING:
    import pandas as pd


def tile_mask(
    tiles: pd.DataFrame,
    tile_extent: tuple[int, int],
    size: tuple[int, int],
    outline_width: int = 4,
) -> Image.Image:
    """Draws the outlines of tiles into a single-channel mask.

    Useful as an overlay that shows which tiles of a slide were used, e.g. after
    tissue-based filtering. Coordinates and extents are in pixels of the level the
    mask is drawn at.

    Args:
        tiles: A DataFrame with columns `x` and `y`, the top-left corner of each tile.
        tile_extent: Width and height of a tile.
        size: Width and height of the mask.
        outline_width: Width of the tile outline in pixels.

    Returns:
        A mode `L` image with tile outlines set to 255 and everything else 0.

    Examples:
        ```python
        import pandas as pd

        from ratiopath.masks import tile_mask

        tiles = pd.DataFrame({"x": [0, 512], "y": [0, 0]})
        mask = tile_mask(tiles, tile_extent=(512, 512), size=(2048, 1024))
        mask.save("tiles.png")
        ```
    """
    mask = Image.new("L", size)
    canvas = ImageDraw.Draw(mask)

    for x, y in tiles[["x", "y"]].itertuples(index=False, name=None):
        canvas.rectangle(
            [x, y, x + tile_extent[0], y + tile_extent[1]],
            outline=255,
            width=outline_width,
        )

    return mask
