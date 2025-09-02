import pytest

from ratiopath.tiling import grid_tiles


@pytest.mark.parametrize(
    "slide_extent, tile_extent, stride, last, expected",
    [
        (
            (5, 5),
            (2, 2),
            (2, 2),
            "drop",
            [(0, 0), (2, 0), (0, 2), (2, 2)],
        ),
        (
            (5, 5),
            (2, 2),
            (2, 2),
            "keep",
            [(0, 0), (2, 0), (4, 0), (0, 2), (2, 2), (4, 2), (0, 4), (2, 4), (4, 4)],
        ),
        (
            (5, 5),
            (2, 2),
            (2, 2),
            "shift",
            [(0, 0), (2, 0), (3, 0), (0, 2), (2, 2), (3, 2), (0, 3), (2, 3), (3, 3)],
        ),
    ],
)
def test_grid_tiler(slide_extent, tile_extent, stride, last, expected):
    tiles = [
        tuple(tile) for tile in grid_tiles(slide_extent, tile_extent, stride, last)
    ]
    assert tiles == expected
