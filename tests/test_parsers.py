"""Tests for annotation parsers."""

import io
import json

import pytest

from ratiopath.parsers import ASAPParser, GeoJSONParser


class TestASAPParser:
    """Test the ASAP parser."""

    @pytest.fixture
    def asap_xml_content(self):
        """Sample ASAP XML content."""
        return """<?xml version="1.0" encoding="UTF-8"?>
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
        </ASAP_Annotations>"""

    def test_get_polygons(self, asap_xml_content):
        """Test parsing polygons from ASAP XML."""
        f = io.StringIO(asap_xml_content)

        parser = ASAPParser(f)
        polygons = list(parser.get_polygons())

        assert len(polygons) == 1
        # Check that we have a polygon-like object
        polygon = polygons[0]
        assert hasattr(polygon, "exterior")

    def test_get_points(self, asap_xml_content):
        """Test parsing points from ASAP XML."""
        f = io.StringIO(asap_xml_content)

        parser = ASAPParser(f)
        points = list(parser.get_points())

        assert len(points) == 1
        # Check that we have a point-like object
        point = points[0]
        assert hasattr(point, "x") and hasattr(point, "y")

    def test_get_polygons_with_filters(self, asap_xml_content):
        """Test parsing polygons with filters."""
        f = io.StringIO(asap_xml_content)
        parser = ASAPParser(f)
        polygons = list(parser.get_polygons(name="Annotation 1"))

        assert len(polygons) == 1
        assert polygons[0].exterior.coords[0] == (100.0, 200.0)

        polygons = list(parser.get_polygons(name="Nonexistent"))
        assert len(polygons) == 0


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
                        "coordinates": [
                            [
                                [100.0, 200.0],
                                [150.0, 200.0],
                                [125.0, 250.0],
                                [100.0, 200.0],
                            ]
                        ],
                    },
                    "properties": {"nested": {"property": "value"}},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [300.0, 400.0]},
                    "properties": {"name": "point1"},
                },
            ],
        }

    def test_get_polygons(self, geojson_content):
        """Test parsing polygons from GeoJSON."""
        f = io.StringIO(json.dumps(geojson_content))

        parser = GeoJSONParser(f)
        polygons = list(parser.get_polygons())

        assert len(polygons) == 1
        # Check that we have a polygon-like object
        polygon = polygons[0]
        assert hasattr(polygon, "exterior")

    def test_get_points(self, geojson_content):
        """Test parsing points from GeoJSON."""
        f = io.StringIO(json.dumps(geojson_content))

        parser = GeoJSONParser(f)
        points = list(parser.get_points())

        assert len(points) == 1
        # Check that we have a point-like object
        point = points[0]
        assert hasattr(point, "x") and hasattr(point, "y")

    def test_get_polygons_with_filters(self, geojson_content):
        """Test parsing polygons with filters."""
        f = io.StringIO(json.dumps(geojson_content))

        parser = GeoJSONParser(f)
        polygons = list(parser.get_polygons(nested_property="value"))

        assert len(polygons) == 1
        assert polygons[0].exterior.coords[0] == (100.0, 200.0)

        polygons = list(parser.get_polygons(name="nested"))
        assert len(polygons) == 0

        polygons = list(parser.get_polygons(name="nonexistent"))
        assert len(polygons) == 0
