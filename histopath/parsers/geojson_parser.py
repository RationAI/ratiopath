"""GeoJSON format annotation parser."""

import json
from typing import Iterable

try:
    import shapely.geometry
    from shapely.geometry import Point, Polygon, shape
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
    
    def shape(geom_dict):
        """Mock shape function for when shapely is not available."""
        geom_type = geom_dict.get('type', '').lower()
        coords = geom_dict.get('coordinates', [])
        
        if geom_type == 'polygon':
            return Polygon(coords[0] if coords else [])
        elif geom_type == 'point':
            return Point(coords[0] if coords else 0, coords[1] if len(coords) > 1 else 0)
        else:
            return None

from histopath.parsers.abstract_parser import AbstractParser


class GeoJSONParser(AbstractParser):
    """Parser for GeoJSON format annotation files.
    
    GeoJSON is a format for encoding geographic data structures using JSON.
    This parser supports both polygon and point geometries.
    """

    def get_polygons(self) -> Iterable[Polygon]:
        """Parse polygon annotations from GeoJSON file.
        
        Returns:
            An iterable of shapely Polygon objects.
        """
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different GeoJSON structures
            features = []
            if data.get('type') == 'FeatureCollection':
                features = data.get('features', [])
            elif data.get('type') == 'Feature':
                features = [data]
            elif data.get('type') in ['Polygon', 'MultiPolygon']:
                # Direct geometry object
                features = [{'geometry': data}]
            
            for feature in features:
                geometry = feature.get('geometry', {})
                geom_type = geometry.get('type', '').lower()
                
                if geom_type == 'polygon':
                    polygon = shape(geometry)
                    if hasattr(polygon, 'coords') or hasattr(polygon, 'exterior'):
                        yield polygon
                elif geom_type == 'multipolygon':
                    multipolygon = shape(geometry)
                    # MultiPolygon contains multiple polygons
                    if hasattr(multipolygon, 'geoms'):
                        for poly in multipolygon.geoms:
                            yield poly
                    else:
                        # Fallback for mock implementation
                        coords = geometry.get('coordinates', [])
                        for poly_coords in coords:
                            if poly_coords:
                                yield Polygon(poly_coords[0])
                        
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            # Return empty iterator if parsing fails
            return
            yield  # This makes the function a generator even when empty

    def get_points(self) -> Iterable[Point]:
        """Parse point annotations from GeoJSON file.
        
        Returns:
            An iterable of shapely Point objects.
        """
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different GeoJSON structures
            features = []
            if data.get('type') == 'FeatureCollection':
                features = data.get('features', [])
            elif data.get('type') == 'Feature':
                features = [data]
            elif data.get('type') in ['Point', 'MultiPoint']:
                # Direct geometry object
                features = [{'geometry': data}]
            
            for feature in features:
                geometry = feature.get('geometry', {})
                geom_type = geometry.get('type', '').lower()
                
                if geom_type == 'point':
                    point = shape(geometry)
                    if hasattr(point, 'x') and hasattr(point, 'y'):
                        yield point
                    else:
                        # Fallback for mock implementation
                        coords = geometry.get('coordinates', [])
                        if len(coords) >= 2:
                            yield Point(coords[0], coords[1])
                elif geom_type == 'multipoint':
                    multipoint = shape(geometry)
                    # MultiPoint contains multiple points
                    if hasattr(multipoint, 'geoms'):
                        for point in multipoint.geoms:
                            yield point
                    else:
                        # Fallback for mock implementation
                        coords = geometry.get('coordinates', [])
                        for point_coords in coords:
                            if len(point_coords) >= 2:
                                yield Point(point_coords[0], point_coords[1])
                        
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            # Return empty iterator if parsing fails
            return
            yield  # This makes the function a generator even when empty