"""QuPath native format annotation parser."""

import json
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


class QuPathParser(AbstractParser):
    """Parser for QuPath native format annotation files.
    
    QuPath stores annotations in a JSON-based format with specific structure.
    This parser supports both polygon and point annotations from QuPath exports.
    """

    def get_polygons(self) -> Iterable[Polygon]:
        """Parse polygon annotations from QuPath JSON file.
        
        Returns:
            An iterable of shapely Polygon objects.
        """
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            # QuPath format typically has objects at the root or in an array
            objects = []
            if isinstance(data, list):
                objects = data
            elif isinstance(data, dict):
                # Could be a single object or contain objects array
                if 'objects' in data:
                    objects = data['objects']
                elif 'geometry' in data:
                    objects = [data]
                else:
                    objects = [data]
            
            for obj in objects:
                # Check if this is an annotation object
                geometry = obj.get('geometry', {})
                object_type = obj.get('objectType', '').lower()
                
                # Look for polygon-like annotations
                if object_type in ['annotation', 'detection'] or 'geometry' in obj:
                    geom_type = geometry.get('type', '').lower()
                    
                    if geom_type == 'polygon':
                        coordinates = geometry.get('coordinates', [])
                        if coordinates and len(coordinates) > 0:
                            # First coordinate array is the exterior ring
                            exterior_coords = coordinates[0]
                            if len(exterior_coords) >= 3:
                                yield Polygon(exterior_coords)
                    
                    elif geom_type == 'multipolygon':
                        coordinates = geometry.get('coordinates', [])
                        for poly_coords in coordinates:
                            if poly_coords and len(poly_coords) > 0:
                                exterior_coords = poly_coords[0]
                                if len(exterior_coords) >= 3:
                                    yield Polygon(exterior_coords)
                    
                    # QuPath specific: Check for ROI (Region of Interest) objects
                    elif 'roi' in obj:
                        roi = obj['roi']
                        if roi.get('type') == 'PolygonROI':
                            points = roi.get('points', [])
                            if len(points) >= 3:
                                coordinates = [(p['x'], p['y']) for p in points if 'x' in p and 'y' in p]
                                if len(coordinates) >= 3:
                                    yield Polygon(coordinates)
                        
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            # Return empty iterator if parsing fails
            return
            yield  # This makes the function a generator even when empty

    def get_points(self) -> Iterable[Point]:
        """Parse point annotations from QuPath JSON file.
        
        Returns:
            An iterable of shapely Point objects.
        """
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            # QuPath format typically has objects at the root or in an array
            objects = []
            if isinstance(data, list):
                objects = data
            elif isinstance(data, dict):
                # Could be a single object or contain objects array
                if 'objects' in data:
                    objects = data['objects']
                elif 'geometry' in data:
                    objects = [data]
                else:
                    objects = [data]
            
            for obj in objects:
                # Check if this is an annotation object
                geometry = obj.get('geometry', {})
                object_type = obj.get('objectType', '').lower()
                
                # Look for point-like annotations
                if object_type in ['annotation', 'detection'] or 'geometry' in obj:
                    geom_type = geometry.get('type', '').lower()
                    
                    if geom_type == 'point':
                        coordinates = geometry.get('coordinates', [])
                        if len(coordinates) >= 2:
                            yield Point(coordinates[0], coordinates[1])
                    
                    elif geom_type == 'multipoint':
                        coordinates = geometry.get('coordinates', [])
                        for point_coords in coordinates:
                            if len(point_coords) >= 2:
                                yield Point(point_coords[0], point_coords[1])
                    
                    # QuPath specific: Check for ROI (Region of Interest) objects
                    elif 'roi' in obj:
                        roi = obj['roi']
                        if roi.get('type') == 'PointROI':
                            x = roi.get('x')
                            y = roi.get('y')
                            if x is not None and y is not None:
                                yield Point(float(x), float(y))
                        elif roi.get('type') == 'PointsROI':
                            points = roi.get('points', [])
                            for p in points:
                                if 'x' in p and 'y' in p:
                                    yield Point(float(p['x']), float(p['y']))
                        
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            # Return empty iterator if parsing fails
            return
            yield  # This makes the function a generator even when empty