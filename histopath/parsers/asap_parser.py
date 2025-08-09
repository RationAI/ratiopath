"""ASAP format annotation parser."""

import re
import xml.etree.ElementTree as ET
from typing import Iterable

from shapely.geometry import Point, Polygon

from histopath.parsers.abstract_parser import AbstractParser


class ASAPParser(AbstractParser):
    """Parser for ASAP format annotation files.

    ASAP (Automated Slide Analysis Platform) uses XML format for storing annotations.
    This parser supports both polygon and point annotations.
    """

    def _get_filtered_annotations(self, **kwargs):
        """Get annotations that match the provided regex filters.

        Args:
            **kwargs: Keyword arguments containing optional regex patterns:
                - name: Pattern to match annotation names
                - part_of_group: Pattern to match annotation groups

        Yields:
            XML annotation elements that match the filters.
        """
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        name_regex = re.compile(kwargs.get("name", ".*"))
        part_of_group_regex = re.compile(kwargs.get("part_of_group", ".*"))

        for annotation in root.findall(".//Annotation"):
            if not name_regex.match(
                annotation.attrib["Name"]
            ) or not part_of_group_regex.match(annotation.attrib["PartOfGroup"]):
                continue

            yield annotation

    def _extract_coordinates(self, annotation: ET.Element) -> list[Point]:
        """Extract coordinates from an annotation element.

        Args:
            annotation: XML annotation element

        Returns:
            List of (x, y) coordinate tuples.
        """
        coordinates = []
        for coordinate in annotation.findall(".//Coordinate"):
            x = float(coordinate.attrib["X"])
            y = float(coordinate.attrib["Y"])
            coordinates.append(Point(x, y))
        return coordinates

    def get_polygons(self, **kwargs) -> Iterable[Polygon]:
        """Parse polygon annotations from ASAP XML file.

        Args:
            **kwargs: Optional keyword arguments for filtering annotations.

        Returns:
            An iterable of shapely Polygon objects.
        """
        for annotation in self._get_filtered_annotations(**kwargs):
            if annotation.attrib["Type"] in ["Polygon", "Spline"]:
                yield Polygon(self._extract_coordinates(annotation))

    def get_points(self, **kwargs) -> Iterable[Point]:
        """Parse point annotations from ASAP XML file.

        Args:
            **kwargs: Optional keyword arguments for filtering annotations.

        Returns:
            An iterable of shapely Point objects.
        """
        for annotation in self._get_filtered_annotations(**kwargs):
            if annotation.attrib["Type"] in ["Point", "Dot"]:
                yield from self._extract_coordinates(annotation)
