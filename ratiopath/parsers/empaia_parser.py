import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

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
            yield Polygon(
                [
                    (float(coordinate[0]), float(coordinate[1]))
                    for coordinate in annotation["coordinates"]
                ]
            )

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
