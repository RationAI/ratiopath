"""ASAP format annotation parser."""

import json
import xml.etree.ElementTree as ET
from typing import Iterable

try:
    import shapely.geometry
    from shapely.geometry import Point, Polygon
except ImportError:
    # Fallback for when shapely is not available during development
    class MockShapely:
        class Polygon:
            def __init__(self, coords):
                self.coords = coords
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
    shapely = MockShapely()
    Point = shapely.Point
    Polygon = shapely.Polygon

from histopath.parsers.abstract_parser import AbstractParser


class ASAPParser(AbstractParser):
    """Parser for ASAP format annotation files.
    
    ASAP (Automated Slide Analysis Platform) uses XML format for storing annotations.
    This parser supports both polygon and point annotations.
    """

    def get_polygons(self) -> Iterable[Polygon]:
        """Parse polygon annotations from ASAP XML file.
        
        Returns:
            An iterable of shapely Polygon objects.
        """
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()
            
            # Look for annotation elements that contain polygons
            for annotation in root.findall('.//Annotation'):
                annotation_type = annotation.get('Type', '').lower()
                if annotation_type in ['polygon', 'spline']:
                    # Get coordinates from the annotation
                    coordinates = []
                    for coordinate in annotation.findall('.//Coordinate'):
                        x = float(coordinate.get('X', 0))
                        y = float(coordinate.get('Y', 0))
                        coordinates.append((x, y))
                    
                    if len(coordinates) >= 3:  # Need at least 3 points for a polygon
                        yield Polygon(coordinates)
                        
        except (ET.ParseError, FileNotFoundError, ValueError) as e:
            # Return empty iterator if parsing fails
            return
            yield  # This makes the function a generator even when empty

    def get_points(self) -> Iterable[Point]:
        """Parse point annotations from ASAP XML file.
        
        Returns:
            An iterable of shapely Point objects.
        """
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()
            
            # Look for annotation elements that contain points
            for annotation in root.findall('.//Annotation'):
                annotation_type = annotation.get('Type', '').lower()
                if annotation_type in ['point', 'dot']:
                    # Get coordinates from the annotation
                    for coordinate in annotation.findall('.//Coordinate'):
                        x = float(coordinate.get('X', 0))
                        y = float(coordinate.get('Y', 0))
                        yield Point(x, y)
                        
        except (ET.ParseError, FileNotFoundError, ValueError) as e:
            # Return empty iterator if parsing fails
            return
            yield  # This makes the function a generator even when empty