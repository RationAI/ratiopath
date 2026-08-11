import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

import numpy as np
from shapely import Point, Polygon


class EMPAIAParser:
    """Parser for EMPAIA format annotation files.

    EMPAIA uses JSON format for storing annotations. This parser supports
    both polygon and point geometry features from the EMPAIA standardized schema.
    """

    def __init__(self, file_path: Path | str | TextIO) -> None:
        """Initialize the EMPAIA parser.

        Args:
            file_path: Path to the EMPAIA JSON annotation file or a file-like object.
        """
        if isinstance(file_path, Path | str):
            with open(file_path) as f:
                self.annotations = json.load(f)
        else:
            self.annotations = json.load(file_path)

    def _get_filtered_annotations(
        self, name: str, annotation_type: str
    ) -> Iterable[dict]:
        """Get annotations that match the provided regex filters.

        Args:
            name: Regex pattern to match annotation names.
            annotation_type: Type of annotation to match (e.g., 'polygon', 'point').

        Yields:
            Dictionary annotation elements that match the filters.
        """
        name_regex = re.compile(name)
        for annotation in self.annotations["items"]:
            if (
                name_regex.match(annotation["name"])
                and annotation["type"] == annotation_type
            ):
                yield annotation

    def get_polygons(self, name: str = ".*") -> Iterable[Polygon]:
        """Get polygon annotations that match the given name pattern.

        Args:
            name: Regex pattern to match annotation names. Default is ".*" (all).

        Yields:
            Polygon representations of the matching annotations.
        """
        for annotation in self._get_filtered_annotations(name, "polygon"):
            coords = np.array(
                [
                    (float(coordinate[0]), float(coordinate[1]))
                    for coordinate in annotation["coordinates"]
                ]
            )
            filtered = self._remove_outlier_points(coords)
            if len(filtered) < 3:
                continue
            poly = Polygon(filtered)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area == 0:
                continue
            yield poly

    def _remove_outlier_points(self, coords: np.ndarray) -> np.ndarray:
        """Remove outlier coordinates using the IQR method.

        Args:
            coords: Array of shape (N, 2) with (x, y) coordinates.

        Returns:
            Array of coordinates with outliers removed.
        """
        if len(coords) < 4:
            return coords

        def iqr_bounds(values: np.ndarray) -> tuple[float, float]:
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)

        x_min, x_max = iqr_bounds(coords[:, 0])
        y_min, y_max = iqr_bounds(coords[:, 1])

        mask = (
            (coords[:, 0] >= x_min)
            & (coords[:, 0] <= x_max)
            & (coords[:, 1] >= y_min)
            & (coords[:, 1] <= y_max)
        )
        return coords[mask]

    def get_points(self, name: str = ".*") -> Iterable[Point]:
        """Get point annotations that match the given name pattern.

        Args:
            name: Regex pattern to match annotation names. Default is ".*" (all).

        Yields:
            Point representations of the matching annotations.
        """
        for annotation in self._get_filtered_annotations(name, "point"):
            yield Point(
                float(annotation["coordinates"][0]), float(annotation["coordinates"][1])
            )
