import numpy as np
import pandas as pd

from ratiopath.masks import tile_mask


def test_tile_mask_draws_outlines_only():
    tiles = pd.DataFrame({"x": [0, 20], "y": [0, 10]})

    mask = np.asarray(
        tile_mask(tiles, tile_extent=(10, 10), size=(40, 30), outline_width=1)
    )

    assert mask.shape == (30, 40)
    assert set(np.unique(mask)) == {0, 255}
    # corners and edges of the first tile
    assert mask[0, 0] == 255
    assert mask[0, 10] == 255
    assert mask[10, 0] == 255
    # interior of the first tile stays empty
    assert mask[5, 5] == 0
    # second tile is placed at (x=20, y=10)
    assert mask[10, 20] == 255
    assert mask[15, 25] == 0
    # outside of any tile
    assert mask[25, 5] == 0


def test_tile_mask_extent_is_width_height():
    tiles = pd.DataFrame({"x": [0], "y": [0]})

    mask = np.asarray(
        tile_mask(tiles, tile_extent=(20, 5), size=(30, 30), outline_width=1)
    )

    assert mask[0, 20] == 255  # right edge at x = width
    assert mask[5, 0] == 255  # bottom edge at y = height
    assert mask[10, 0] == 0


def test_tile_mask_empty_tiles():
    tiles = pd.DataFrame({"x": [], "y": []})

    mask = np.asarray(tile_mask(tiles, tile_extent=(10, 10), size=(16, 8)))

    assert mask.shape == (8, 16)
    assert not mask.any()
