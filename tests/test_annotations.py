"""Tests for annotation processing functions."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

try:
    import numpy as np
except ImportError:
    # Fallback for when numpy is not available during development
    class MockNumPy:
        def array(self, data, dtype=None):
            return data
        def array_equal(self, a, b):
            return a == b
    np = MockNumPy()

from histopath.ray.annotations import map_annotations, _get_parser_for_file


class TestMapAnnotations:
    """Test the map_annotations function."""
    
    @pytest.fixture
    def sample_geojson_content(self):
        """Sample GeoJSON content for testing."""
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
                    },
                    "properties": {}
                },
                {
                    "type": "Feature", 
                    "geometry": {
                        "type": "Point",
                        "coordinates": [5, 5]
                    },
                    "properties": {}
                }
            ]
        }
    
    def test_map_annotations_basic(self, sample_geojson_content):
        """Test basic functionality of map_annotations."""
        # Create a temporary annotation file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_geojson_content, f)
            f.flush()
            
            try:
                # Create sample input data
                rows = {
                    "annotation_path": [f.name],
                    "other_data": [42]
                }
                
                # Call map_annotations
                result = map_annotations(rows)
                
                # Check that the function returns expected structure
                assert "annotation_path" in result
                assert "other_data" in result
                assert "polygons" in result
                assert "points" in result
                
                # Check that data is preserved
                assert result["annotation_path"] == rows["annotation_path"]
                assert result["other_data"] == rows["other_data"]
                
                # Check that annotation data is added
                assert len(result["polygons"]) == 1
                assert len(result["points"]) == 1
                
                # Check that we got some annotations
                polygons = result["polygons"][0]
                points = result["points"][0]
                assert len(polygons) > 0  # Should have parsed at least one polygon
                assert len(points) > 0   # Should have parsed at least one point
                
            finally:
                os.unlink(f.name)
    
    def test_map_annotations_missing_column(self):
        """Test map_annotations when annotation_path column is missing."""
        rows = {
            "other_data": [1, 2, 3]
        }
        
        result = map_annotations(rows)
        
        # Should add empty annotation columns
        assert "polygons" in result
        assert "points" in result
        assert len(result["polygons"]) == 3
        assert len(result["points"]) == 3
        
        # All annotation entries should be empty
        for i in range(3):
            assert len(result["polygons"][i]) == 0
            assert len(result["points"][i]) == 0
    
    def test_map_annotations_nonexistent_file(self):
        """Test map_annotations with nonexistent annotation files."""
        rows = {
            "annotation_path": ["/nonexistent/file.json"],
            "other_data": [42]
        }
        
        result = map_annotations(rows)
        
        # Should handle missing files gracefully
        assert "polygons" in result
        assert "points" in result
        assert len(result["polygons"]) == 1
        assert len(result["points"]) == 1
        
        # Should have empty annotations for missing file
        assert len(result["polygons"][0]) == 0
        assert len(result["points"][0]) == 0
    
    def test_map_annotations_custom_column_name(self, sample_geojson_content):
        """Test map_annotations with custom annotation path column name."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_geojson_content, f)
            f.flush()
            
            try:
                rows = {
                    "custom_annotation_path": [f.name],
                    "other_data": [42]
                }
                
                result = map_annotations(rows, annotation_path_column="custom_annotation_path")
                
                assert "polygons" in result
                assert "points" in result
                assert len(result["polygons"]) == 1
                assert len(result["points"]) == 1
                
            finally:
                os.unlink(f.name)
    
    def test_map_annotations_multiple_files(self, sample_geojson_content):
        """Test map_annotations with multiple annotation files."""
        # Create multiple temporary files
        temp_files = []
        try:
            for i in range(3):
                f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json.dump(sample_geojson_content, f)
                f.flush()
                f.close()
                temp_files.append(f.name)
            
            rows = {
                "annotation_path": temp_files,
                "slide_id": [1, 2, 3]
            }
            
            result = map_annotations(rows)
            
            # Should process all files
            assert len(result["polygons"]) == 3
            assert len(result["points"]) == 3
            
            # Each should have annotations
            for i in range(3):
                assert len(result["polygons"][i]) > 0
                assert len(result["points"][i]) > 0
                
        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)


class TestGetParserForFile:
    """Test the parser selection function."""
    
    def test_xml_file_returns_asap_parser(self):
        """Test that XML files return ASAPParser."""
        from histopath.parsers import ASAPParser
        parser_class = _get_parser_for_file("/path/to/file.xml")
        assert parser_class == ASAPParser
    
    def test_json_file_returns_geojson_parser_by_default(self):
        """Test that JSON files return GeoJSONParser by default."""
        from histopath.parsers import GeoJSONParser
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)
            f.flush()
            
            try:
                parser_class = _get_parser_for_file(f.name)
                assert parser_class == GeoJSONParser
            finally:
                os.unlink(f.name)
    
    def test_unsupported_extension_raises_error(self):
        """Test that unsupported file extensions raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported annotation file extension"):
            _get_parser_for_file("/path/to/file.txt")