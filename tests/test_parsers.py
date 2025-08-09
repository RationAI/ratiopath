"""Tests for annotation parsers."""

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from histopath.parsers import ASAPParser, AbstractParser, GeoJSONParser, QuPathParser


class TestAbstractParser:
    """Test the abstract base class."""
    
    def test_abstract_parser_cannot_be_instantiated(self):
        """Test that AbstractParser cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractParser("/path/to/file")


class TestASAPParser:
    """Test the ASAP parser."""
    
    @pytest.fixture
    def asap_xml_content(self):
        """Sample ASAP XML content."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
        <ASAP_Annotations>
            <Annotations>
                <Annotation Name="Annotation 1" Type="Polygon" PartOfGroup="None" Color="#FF0000">
                    <Coordinates>
                        <Coordinate Order="0" X="100.0" Y="200.0" />
                        <Coordinate Order="1" X="150.0" Y="200.0" />
                        <Coordinate Order="2" X="125.0" Y="250.0" />
                    </Coordinates>
                </Annotation>
                <Annotation Name="Annotation 2" Type="Point" PartOfGroup="None" Color="#00FF00">
                    <Coordinates>
                        <Coordinate Order="0" X="300.0" Y="400.0" />
                    </Coordinates>
                </Annotation>
            </Annotations>
        </ASAP_Annotations>'''
    
    def test_get_polygons(self, asap_xml_content):
        """Test parsing polygons from ASAP XML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(asap_xml_content)
            f.flush()
            
            try:
                parser = ASAPParser(f.name)
                polygons = list(parser.get_polygons())
                
                assert len(polygons) == 1
                # Check that we have a polygon-like object
                polygon = polygons[0]
                assert hasattr(polygon, 'coords') or hasattr(polygon, 'exterior')
            finally:
                os.unlink(f.name)
    
    def test_get_points(self, asap_xml_content):
        """Test parsing points from ASAP XML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(asap_xml_content)
            f.flush()
            
            try:
                parser = ASAPParser(f.name)
                points = list(parser.get_points())
                
                assert len(points) == 1
                # Check that we have a point-like object
                point = points[0]
                assert hasattr(point, 'x') and hasattr(point, 'y')
            finally:
                os.unlink(f.name)
    
    def test_empty_file_handling(self):
        """Test handling of empty or invalid files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write("")
            f.flush()
            
            try:
                parser = ASAPParser(f.name)
                polygons = list(parser.get_polygons())
                points = list(parser.get_points())
                
                assert len(polygons) == 0
                assert len(points) == 0
            finally:
                os.unlink(f.name)


class TestGeoJSONParser:
    """Test the GeoJSON parser."""
    
    @pytest.fixture
    def geojson_content(self):
        """Sample GeoJSON content."""
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[100.0, 200.0], [150.0, 200.0], [125.0, 250.0], [100.0, 200.0]]]
                    },
                    "properties": {"name": "polygon1"}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [300.0, 400.0]
                    },
                    "properties": {"name": "point1"}
                }
            ]
        }
    
    def test_get_polygons(self, geojson_content):
        """Test parsing polygons from GeoJSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(geojson_content, f)
            f.flush()
            
            try:
                parser = GeoJSONParser(f.name)
                polygons = list(parser.get_polygons())
                
                assert len(polygons) == 1
                # Check that we have a polygon-like object
                polygon = polygons[0]
                assert hasattr(polygon, 'coords') or hasattr(polygon, 'exterior')
            finally:
                os.unlink(f.name)
    
    def test_get_points(self, geojson_content):
        """Test parsing points from GeoJSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(geojson_content, f)
            f.flush()
            
            try:
                parser = GeoJSONParser(f.name)
                points = list(parser.get_points())
                
                assert len(points) == 1
                # Check that we have a point-like object
                point = points[0]
                assert hasattr(point, 'x') and hasattr(point, 'y')
            finally:
                os.unlink(f.name)


class TestQuPathParser:
    """Test the QuPath parser."""
    
    @pytest.fixture
    def qupath_content(self):
        """Sample QuPath JSON content."""
        return [
            {
                "objectType": "ANNOTATION",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[100.0, 200.0], [150.0, 200.0], [125.0, 250.0], [100.0, 200.0]]]
                },
                "properties": {"name": "polygon1"}
            },
            {
                "objectType": "DETECTION",
                "geometry": {
                    "type": "Point",
                    "coordinates": [300.0, 400.0]
                },
                "properties": {"name": "point1"}
            },
            {
                "objectType": "ANNOTATION",
                "roi": {
                    "type": "PolygonROI",
                    "points": [
                        {"x": 500.0, "y": 600.0},
                        {"x": 550.0, "y": 600.0},
                        {"x": 525.0, "y": 650.0}
                    ]
                }
            }
        ]
    
    def test_get_polygons(self, qupath_content):
        """Test parsing polygons from QuPath JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(qupath_content, f)
            f.flush()
            
            try:
                parser = QuPathParser(f.name)
                polygons = list(parser.get_polygons())
                
                assert len(polygons) == 2  # One from geometry, one from ROI
                # Check that we have polygon-like objects
                for polygon in polygons:
                    assert hasattr(polygon, 'coords') or hasattr(polygon, 'exterior')
            finally:
                os.unlink(f.name)
    
    def test_get_points(self, qupath_content):
        """Test parsing points from QuPath JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(qupath_content, f)
            f.flush()
            
            try:
                parser = QuPathParser(f.name)
                points = list(parser.get_points())
                
                assert len(points) == 1
                # Check that we have a point-like object
                point = points[0]
                assert hasattr(point, 'x') and hasattr(point, 'y')
            finally:
                os.unlink(f.name)