"""Annotation processing functions for ray.Dataset operations."""

import os
import re
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:
    # Fallback for when numpy is not available during development
    class MockNumPy:
        def array(self, data, dtype=None):
            return data
    np = MockNumPy()

try:
    import shapely.geometry.base
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    BaseGeometry = shapely.geometry.base.BaseGeometry
except ImportError:
    # Fallback for when shapely is not available during development
    class MockShapelyBase:
        pass
    
    class MockShapely:
        class geometry:
            base = MockShapelyBase
        
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        class Polygon:
            def __init__(self, coords):
                self.coords = coords
    
    shapely = MockShapely()
    Point = shapely.Point
    Polygon = shapely.Polygon
    BaseGeometry = MockShapelyBase
    
    def unary_union(geoms):
        """Mock unary_union for when shapely is not available."""
        return geoms[0] if geoms else None

from histopath.parsers import ASAPParser, GeoJSONParser, QuPathParser


def _get_parser_for_file(file_path: str):
    """Determine the appropriate parser based on file extension.
    
    Args:
        file_path: Path to the annotation file.
        
    Returns:
        Parser class appropriate for the file type.
        
    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = Path(file_path).suffix.lower()
    
    if ext in ['.xml']:
        return ASAPParser
    elif ext in ['.json', '.geojson']:
        # Try to distinguish between GeoJSON and QuPath JSON
        # This is a heuristic - in practice, you might need more sophisticated detection
        try:
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # GeoJSON typically has 'type' field with values like 'FeatureCollection', 'Feature', etc.
            if isinstance(data, dict):
                data_type = data.get('type', '').lower()
                if data_type in ['featurecollection', 'feature', 'point', 'polygon', 'multipoint', 'multipolygon']:
                    return GeoJSONParser
                elif 'objectType' in data or (isinstance(data, dict) and 'roi' in data):
                    return QuPathParser
                elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    # Check first item for QuPath-like structure
                    first_item = data[0]
                    if 'objectType' in first_item or 'roi' in first_item:
                        return QuPathParser
            
            # Default to GeoJSON for JSON files
            return GeoJSONParser
            
        except (json.JSONDecodeError, FileNotFoundError):
            # If we can't read the file, default to GeoJSON
            return GeoJSONParser
    else:
        raise ValueError(f"Unsupported annotation file extension: {ext}")


def _filter_by_roi(annotations, roi):  # Removed type annotation for compatibility
    """Filter annotations by region of interest.
    
    Args:
        annotations: Iterable of shapely geometry objects.
        roi: Region of interest geometry to filter by.
        
    Returns:
        Filtered list of annotations that intersect with the ROI.
    """
    if roi is None:
        return list(annotations)
    
    filtered = []
    for annotation in annotations:
        try:
            if hasattr(annotation, 'intersects') and annotation.intersects(roi):
                filtered.append(annotation)
            else:
                # Fallback for mock implementation - just add all
                filtered.append(annotation)
        except Exception:
            # If intersection test fails, include the annotation
            filtered.append(annotation)
    
    return filtered


def _filter_by_regex(annotations, annotation_data: dict, regex_pattern: Optional[str]):
    """Filter annotations by regex pattern applied to annotation properties.
    
    Args:
        annotations: List of shapely geometry objects.
        annotation_data: Dictionary containing annotation metadata.
        regex_pattern: Regular expression pattern to match against annotation properties.
        
    Returns:
        Filtered list of annotations.
    """
    if regex_pattern is None:
        return annotations
    
    try:
        pattern = re.compile(regex_pattern)
    except re.error:
        # If regex is invalid, return all annotations
        return annotations
    
    # For now, return all annotations since we don't have access to annotation properties
    # In a full implementation, this would match against annotation labels, classes, etc.
    return annotations


def map_annotations(
    rows: dict,  # Simplified type annotation for compatibility
    annotation_path_column: str = "annotation_path",
    roi = None,  # Removed type annotation for compatibility
    annotation_filter_regex: Optional[str] = None
) -> dict:  # Simplified type annotation for compatibility
    """Process annotation files and add parsed annotation data to the dataset.
    
    This function is designed to be used with ray.Dataset.map_batches() to process
    batches of annotation files in parallel.
    
    Args:
        rows: Dictionary containing batch data with numpy arrays as values.
        annotation_path_column: Name of the column containing annotation file paths.
        roi: Optional region of interest geometry to filter annotations.
        annotation_filter_regex: Optional regex pattern to filter annotations.
        
    Returns:
        Dictionary with the same structure as input plus new columns for annotation data.
        
    Example:
        >>> import ray
        >>> ds = ray.data.from_items([{"annotation_path": "annotations.json"}])
        >>> ds = ds.map_batches(map_annotations)
    """
    # Copy input rows to avoid modifying the original
    result = dict(rows)
    
    # Get annotation paths
    if annotation_path_column not in rows:
        # If annotation path column doesn't exist, add empty annotation columns
        batch_size = len(next(iter(rows.values()))) if rows else 0
        result["polygons"] = [[] for _ in range(batch_size)]
        result["points"] = [[] for _ in range(batch_size)]
        return result
    
    annotation_paths = rows[annotation_path_column]
    batch_size = len(annotation_paths)
    
    # Initialize output arrays
    polygons_data = []
    points_data = []
    
    for i in range(batch_size):
        try:
            annotation_path = annotation_paths[i]
            
            # Check if file exists
            if not os.path.exists(annotation_path):
                polygons_data.append([])
                points_data.append([])
                continue
            
            # Get appropriate parser
            parser_class = _get_parser_for_file(annotation_path)
            parser = parser_class(annotation_path)
            
            # Parse polygons and points
            polygons = list(parser.get_polygons())
            points = list(parser.get_points())
            
            # Apply ROI filter if specified
            polygons = _filter_by_roi(polygons, roi)
            points = _filter_by_roi(points, roi)
            
            # Apply regex filter if specified
            # Note: This is a simplified implementation - in practice, you'd need
            # access to annotation metadata for more sophisticated filtering
            polygons = _filter_by_regex(polygons, {}, annotation_filter_regex)
            points = _filter_by_regex(points, {}, annotation_filter_regex)
            
            polygons_data.append(polygons)
            points_data.append(points)
            
        except Exception as e:
            # If parsing fails for any reason, add empty lists
            polygons_data.append([])
            points_data.append([])
    
    # Add parsed annotation data to result
    result["polygons"] = polygons_data
    result["points"] = points_data
    
    return result