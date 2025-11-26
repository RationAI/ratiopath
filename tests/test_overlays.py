import pandas as pd
import pytest

from ratiopath.tiling.overlays import overlay_roi


@pytest.mark.parametrize(
    "tile_x,tile_y,tile_extent_x,tile_extent_y,roi_offset_x,roi_offset_y,roi_extent_x,roi_extent_y,expected_x,expected_y,expected_extent_x,expected_extent_y",
    [
        # Default parameters (no adjustment)
        (100, 200, 256, 256, 0, 0, 1, 1, 100, 200, 256, 256),
        # Only X offset
        (100, 200, 256, 256, 0.5, 0, 1, 1, 228, 200, 256, 256),
        # Only Y offset
        (100, 200, 256, 256, 0, 0.5, 1, 1, 100, 328, 256, 256),
        # Both offsets
        (100, 200, 256, 256, 0.25, 0.75, 1, 1, 164, 392, 256, 256),
        # Only X extent reduction
        (100, 200, 256, 256, 0, 0, 0.5, 1, 100, 200, 128, 256),
        # Only Y extent reduction
        (100, 200, 256, 256, 0, 0, 1, 0.5, 100, 200, 256, 128),
        # Both extent reductions
        (100, 200, 256, 256, 0, 0, 0.5, 0.5, 100, 200, 128, 128),
        # Combined offset and extent - center quarter
        (100, 200, 256, 256, 0.25, 0.25, 0.5, 0.5, 164, 264, 128, 128),
        # Edge case: zero extents
        (100, 200, 256, 256, 0, 0, 0, 0, 100, 200, 0, 0),
        # Different tile extents in x and y
        (0, 0, 512, 256, 0.5, 0.5, 0.5, 0.5, 256, 128, 256, 128),
    ],
)
def test_overlay_roi_single_tile(
    tile_x,
    tile_y,
    tile_extent_x,
    tile_extent_y,
    roi_offset_x,
    roi_offset_y,
    roi_extent_x,
    roi_extent_y,
    expected_x,
    expected_y,
    expected_extent_x,
    expected_extent_y,
):
    """Test overlay_roi with single tile for various offset and extent values."""
    batch = pd.DataFrame(
        {
            "tile_x": [tile_x],
            "tile_y": [tile_y],
            "tile_extent_x": [tile_extent_x],
            "tile_extent_y": [tile_extent_y],
        }
    )

    roi_func = overlay_roi(roi_offset_x, roi_offset_y, roi_extent_x, roi_extent_y)
    result = roi_func(batch)

    assert result["tile_x"].iloc[0] == expected_x
    assert result["tile_y"].iloc[0] == expected_y
    assert result["tile_extent_x"].iloc[0] == expected_extent_x
    assert result["tile_extent_y"].iloc[0] == expected_extent_y


def test_overlay_roi_multiple_tiles():
    """Test overlay_roi correctly adjusts all tiles in a batch."""
    batch = pd.DataFrame(
        {
            "tile_x": [0, 256, 512],
            "tile_y": [0, 256, 512],
            "tile_extent_x": [256, 256, 256],
            "tile_extent_y": [256, 256, 256],
        }
    )

    roi_func = overlay_roi(0.25, 0.25, 0.5, 0.5)
    result = roi_func(batch)

    # Each tile should have the same adjustment formula applied
    assert list(result["tile_x"]) == [64, 320, 576]
    assert list(result["tile_y"]) == [64, 320, 576]
    assert list(result["tile_extent_x"]) == [128, 128, 128]
    assert list(result["tile_extent_y"]) == [128, 128, 128]


def test_overlay_roi_varying_tile_extents():
    """Test overlay_roi with tiles having different extents."""
    batch = pd.DataFrame(
        {
            "tile_x": [0, 100, 200],
            "tile_y": [0, 100, 200],
            "tile_extent_x": [100, 200, 300],
            "tile_extent_y": [50, 100, 150],
        }
    )

    roi_func = overlay_roi(0.5, 0.5, 0.5, 0.5)
    result = roi_func(batch)

    # Offset calculation: tile_x + (offset * tile_extent_x)
    assert list(result["tile_x"]) == [50, 200, 350]  # 0+50, 100+100, 200+150
    assert list(result["tile_y"]) == [25, 150, 275]  # 0+25, 100+50, 200+75
    # Extent calculation: tile_extent * extent_factor
    assert list(result["tile_extent_x"]) == [50, 100, 150]
    assert list(result["tile_extent_y"]) == [25, 50, 75]


def test_overlay_roi_preserves_other_columns():
    """Test that overlay_roi preserves other columns in the DataFrame."""
    batch = pd.DataFrame(
        {
            "tile_x": [100],
            "tile_y": [200],
            "tile_extent_x": [256],
            "tile_extent_y": [256],
            "level": [0],
            "mpp_x": [0.5],
            "mpp_y": [0.5],
            "overlay_path": ["/path/to/overlay.tiff"],
        }
    )

    roi_func = overlay_roi(0.25, 0.25, 0.5, 0.5)
    result = roi_func(batch)

    # Check that other columns are preserved
    assert result["level"].iloc[0] == 0
    assert result["mpp_x"].iloc[0] == 0.5
    assert result["mpp_y"].iloc[0] == 0.5
    assert result["overlay_path"].iloc[0] == "/path/to/overlay.tiff"


def test_overlay_roi_fractional_results():
    """Test overlay_roi with fractional offset and extent values that produce float results."""
    batch = pd.DataFrame(
        {
            "tile_x": [0],
            "tile_y": [0],
            "tile_extent_x": [100],
            "tile_extent_y": [100],
        }
    )

    # Use values that produce fractional results
    roi_func = overlay_roi(0.33, 0.67, 0.33, 0.67)
    result = roi_func(batch)

    # Results should be floats (33.0, 67.0, 33.0, 67.0)
    assert result["tile_x"].iloc[0] == pytest.approx(33.0)
    assert result["tile_y"].iloc[0] == pytest.approx(67.0)
    assert result["tile_extent_x"].iloc[0] == pytest.approx(33.0)
    assert result["tile_extent_y"].iloc[0] == pytest.approx(67.0)


def test_overlay_roi_default_parameters():
    """Test that overlay_roi with default parameters returns unchanged coordinates."""
    batch = pd.DataFrame(
        {
            "tile_x": [100, 200, 300],
            "tile_y": [50, 100, 150],
            "tile_extent_x": [256, 256, 256],
            "tile_extent_y": [128, 128, 128],
        }
    )

    roi_func = overlay_roi()  # Use all defaults
    result = roi_func(batch)

    assert list(result["tile_x"]) == [100, 200, 300]
    assert list(result["tile_y"]) == [50, 100, 150]
    assert list(result["tile_extent_x"]) == [256, 256, 256]
    assert list(result["tile_extent_y"]) == [128, 128, 128]
