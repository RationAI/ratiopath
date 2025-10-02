"""Tests for annotation processing functions."""

import io
import json
from math import isclose

import numpy as np
import pytest
from shapely import Polygon

from ratiopath.parsers import GeoJSONParser
from ratiopath.tiling.annotations import tile_annotations


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
                        "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                    },
                    "properties": {},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [5, 5]},
                    "properties": {},
                },
            ],
        }

    def test_map_annotations_basic(self, sample_geojson_content):
        """Test basic functionality of map_annotations."""
        # Create a temporary annotation file
        f = io.StringIO(json.dumps(sample_geojson_content))

        results = tile_annotations(
            annotations=list(GeoJSONParser(f).get_polygons()),
            roi=Polygon([(0, 0), (8, 0), (8, 8), (0, 8)]),
            coordinates=np.array([[0, 0], [8, 0], [0, 8], [8, 8]]),
            downsample=1,
        )

        for result_polygon, area in zip(results, [64.0, 16.0, 16.0, 4.0], strict=True):
            assert result_polygon.is_valid
            assert isclose(result_polygon.area, area)

    def test_map_annotations_custom_roi(self, sample_geojson_content):
        """Test map_annotations with a custom region of interest."""
        f = io.StringIO(json.dumps(sample_geojson_content))

        results = tile_annotations(
            annotations=list(GeoJSONParser(f).get_polygons()),
            roi=Polygon([(1, 1), (7, 1), (7, 7), (1, 7)]),
            coordinates=np.array([[0, 0], [8, 0], [0, 8], [8, 8]]),
            downsample=1,
        )

        for result_polygon, area in zip(results, [36.0, 6.0, 6.0, 1.0], strict=True):
            assert result_polygon.is_valid
            assert isclose(result_polygon.area, area)
