"""Annotation processing functions for ray.Dataset operations."""

from pathlib import Path
from typing import Optional

import numpy as np
from shapely import Polygon, STRtree, transform
from shapely.geometry.base import BaseGeometry

from histopath.parsers import AbstractParser, ASAPParser, GeoJSONParser


def _get_parser_for_file(file_path: Path) -> AbstractParser:
    """Determine the appropriate parser based on file extension.

    Args:
        file_path: Path to the annotation file.

    Returns:
        Parser class appropriate for the file type.

    Raises:
        ValueError: If the file extension is not supported.
    """

    if file_path.suffix == ".xml":
        return ASAPParser(file_path)
    elif file_path.suffix == ".geojson":
        return GeoJSONParser(file_path)
    else:
        raise ValueError(f"Unsupported annotation file type: {file_path.suffix}")


def map_annotations(
    rows: dict[str, np.ndarray],
    annotation_path_column: str = "annotation_path",
    roi: Optional[BaseGeometry] = None,
    **kwargs,
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
    if annotation_path_column not in rows:
        raise ValueError(f"Column '{annotation_path_column}' not found in input rows.")

    annotation_file = Path(rows[annotation_path_column][0])
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file '{annotation_file}' does not exist.")

    parser = _get_parser_for_file(annotation_file)

    # Validate that all tiles have the same extent and downsample values
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

    roi = transform(roi, lambda x: x * downsample)

    polygons = list(parser.get_polygons(**kwargs))
    tree = STRtree(polygons)

    rows["annotation_coverage_px"] = np.zeros(
        len(rows[annotation_path_column]), dtype=np.float32
    )
    rows["annotation_coverage_percent"] = np.zeros(
        len(rows[annotation_path_column]), dtype=np.float32
    )

    for i in range(len(rows[annotation_path_column])):
        roi_transformed = transform(
            roi,
            lambda x: x
            + [rows["tile_x"][i] * downsample, rows["tile_y"][i] * downsample],
        )

        polygon = Polygon()
        for polygon_index in tree.query(roi_transformed, predicate="intersects"):
            polygon = polygon.union(
                polygons[int(polygon_index)].intersection(roi_transformed)
            )

        rows["annotation_coverage_px"][i] = polygon.area / downsample**2

    rows["annotation_coverage_percent"] = (
        rows["annotation_coverage_px"] / roi.area * 100
    )

    return rows
