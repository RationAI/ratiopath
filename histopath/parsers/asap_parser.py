"""ASAP format annotation parser."""

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

from shapely.geometry import Point, Polygon


class ASAPParser:
    """Parser for ASAP format annotation files.

    ASAP (Automated Slide Analysis Platform) uses XML format for storing annotations.
    This parser supports both polygon and point annotations.
    """

    def __init__(self, file_path: Path | str | TextIO):
        self.tree = ET.parse(file_path)
        self.root = self.tree.getroot()

    def _get_filtered_annotations(
        self, name: str, part_of_group: str
    ) -> Iterable[ET.Element]:
        """Get annotations that match the provided regex filters.

        Args:
            name: Regex pattern to match annotation names.
            part_of_group: Regex pattern to match annotation groups.

        Yields:
            XML annotation elements that match the filters.
        """
        name_regex = re.compile(name)
        part_of_group_regex = re.compile(part_of_group)

        for annotation in self.root.findall(".//Annotation"):
            if name_regex.match(
                annotation.attrib["Name"]
            ) and part_of_group_regex.match(annotation.attrib["PartOfGroup"]):
                yield annotation

    def _extract_coordinates(self, annotation: ET.Element) -> list[Point]:
        """Extract coordinates from an annotation element.

        Args:
            annotation: XML annotation element.

        Returns:
            List of (x, y) coordinate tuples.
        """
        return [
            Point(float(coordinate.attrib["X"]), float(coordinate.attrib["Y"]))
            for coordinate in annotation.findall(".//Coordinate")
        ]

    def get_polygons(
        self, name: str = ".*", part_of_group: str = ".*"
    ) -> Iterable[Polygon]:
        """Parse polygon annotations from ASAP XML file.

        Args:
            name: Regex pattern to match annotation names.
            part_of_group: Regex pattern to match annotation groups.

        Returns:
            An iterable of shapely Polygon objects.
        """
        for annotation in self._get_filtered_annotations(name, part_of_group):
            if annotation.attrib["Type"] in ["Polygon", "Spline"]:
                yield Polygon(self._extract_coordinates(annotation))

    def get_points(
        self, name: str = ".*", part_of_group: str = ".*"
    ) -> Iterable[Point]:
        """Parse point annotations from ASAP XML file.

        Args:
            name: Regex pattern to match annotation names.
            part_of_group: Regex pattern to match annotation groups.

        Returns:
            An iterable of shapely Point objects.
        """
        for annotation in self._get_filtered_annotations(name, part_of_group):
            if annotation.attrib["Type"] in ["Point", "Dot"]:
                yield from self._extract_coordinates(annotation)
