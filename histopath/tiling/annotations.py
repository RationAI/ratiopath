"""Annotation processing functions for ray.Dataset operations."""

from pathlib import Path
from typing import Callable

import numpy as np
from shapely import Polygon, STRtree, transform
from shapely.geometry.base import BaseGeometry


def get_annotation_path(
    rows: dict[str, np.ndarray], annotation_path_column: str
) -> Path:
    """Get the annotation file path from the input rows.

    Args:
        rows: Dictionary containing batch data with numpy arrays as values.
        annotation_path_column: Name of the column containing annotation file paths.

    Returns:
        Path: The path to the annotation file.
    """
    if annotation_path_column not in rows:
        raise ValueError(f"Column '{annotation_path_column}' not found in input rows.")

    annotation_file = Path(rows[annotation_path_column][0])
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file '{annotation_file}' does not exist.")

    for i in range(1, len(rows[annotation_path_column])):
        if Path(rows[annotation_path_column][i]) != annotation_file:
            raise ValueError("All annotation files must be the same.")

    return annotation_file


def get_roi(rows: dict[str, np.ndarray], roi: BaseGeometry | None) -> BaseGeometry:
    """Get the region of interest (ROI) from the input rows.

    Args:
        rows: Dictionary containing batch data with numpy arrays as values.
        roi: Optional region of interest geometry.

    Returns:
        BaseGeometry: The region of interest geometry.
    """
    if (
        not np.all(rows["tile_extent_x"] == rows["tile_extent_x"][0])
        or not np.all(rows["tile_extent_y"] == rows["tile_extent_y"][0])
        or not np.all(rows["downsample"] == rows["downsample"][0])
    ):
        raise ValueError(
            "All tiles in the batch must have the same extent and downsample values."
        )

    tile_extent_x = rows["tile_extent_x"][0]
    tile_extent_y = rows["tile_extent_y"][0]
    downsample = rows["downsample"][0]

    if roi is None:
        roi = Polygon(
            [
                (0, 0),
                (tile_extent_x, 0),
                (tile_extent_x, tile_extent_y),
                (0, tile_extent_y),
            ]
        )

    minx, miny, maxx, maxy = roi.bounds
    if minx < 0 or miny < 0 or maxx > tile_extent_x or maxy > tile_extent_y:
        raise ValueError("ROI is out of bounds.")

    return transform(roi, lambda x: x * downsample)


def shift_roi(
    rows: dict[str, np.ndarray], roi: BaseGeometry, tile_index: int
) -> BaseGeometry:
    """Shift the region of interest (ROI) for a specific tile.

    Args:
        rows: Dictionary containing batch data with numpy arrays as values.
        roi: The current region of interest geometry.
        tile_index: The index of the tile to shift the ROI for.

    Returns:
        BaseGeometry: The shifted region of interest geometry.
    """
    tile_x = rows["tile_x"][tile_index]
    tile_y = rows["tile_y"][tile_index]
    downsample = rows["downsample"][tile_index]
    return transform(roi, lambda x: x + [tile_x * downsample, tile_y * downsample])


def map_annotations(
    rows: dict[str, np.ndarray],
    tree_factory: Callable[[Path], STRtree],
    annotation_path_column: str = "annotation_path",
    roi: BaseGeometry | None = None,
    annotation_name: str = "annotation",
) -> dict[str, np.ndarray]:
    """Process annotation files and add parsed annotation data to the dataset.

    This function is designed to be used with ray.Dataset.map_batches(). It expects
    a batch of all tiles from a single slide!

    Args:
        rows: Dictionary containing batch data with numpy arrays as values.
        annotation_path_column: Name of the column containing annotation file paths.
        roi: Optional region of interest geometry to filter annotations.

    Returns:
        Dictionary with the same structure as input plus new columns for annotation data.
    """
    annotation_file = get_annotation_path(rows, annotation_path_column)
    roi = get_roi(rows, roi)
    tree = tree_factory(annotation_file)
    downsample = rows["downsample"][0]

    rows[f"{annotation_name}_coverage_area"] = np.zeros(
        len(rows[annotation_path_column]), dtype=np.float32
    )

    for i in range(len(rows[annotation_path_column])):
        roi_shifted = shift_roi(rows, roi, i)
        polygon = Polygon()
        for polygon_index in tree.query(roi_shifted, predicate="intersects"):
            polygon = polygon.union(
                tree.geometries[int(polygon_index)].intersection(roi_shifted)
            )

        rows[f"{annotation_name}_coverage_area"][i] = polygon.area / downsample**2

    rows[f"{annotation_name}_coverage"] = (
        rows[f"{annotation_name}_coverage_area"] / roi.area
    )

    return rows
