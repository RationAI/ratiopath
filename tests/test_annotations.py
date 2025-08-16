"""Tests for annotation processing functions."""

import json
import os
import tempfile

import numpy as np
import pytest
from shapely import Polygon, STRtree

from histopath.parsers.geojson_parser import GeoJSONParser
from histopath.tiling.annotations import map_annotations


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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".geojson", delete=False
        ) as f:
            json.dump(sample_geojson_content, f)
            f.flush()

            try:
                # Create sample input data
                rows = {"annotation_path": [f.name], "other_data": [42]}
                rows = {
                    "annotation_path": [f.name for _ in range(4)],
                    "tile_x": np.array([0, 8, 0, 8]),
                    "tile_y": np.array([0, 0, 8, 8]),
                    "tile_extent_x": np.array([8, 8, 8, 8]),
                    "tile_extent_y": np.array([8, 8, 8, 8]),
                    "downsample": np.array([1, 1, 1, 1]),
                }

                # Call map_annotations
                result = map_annotations(
                    rows, lambda x: STRtree(list(GeoJSONParser(x).get_polygons()))
                )

                # Check that the function returns expected structure
                assert "annotation_path" in result
                assert "tile_x" in result
                assert "tile_y" in result
                assert "tile_extent_x" in result
                assert "tile_extent_y" in result
                assert "downsample" in result
                assert "annotation_coverage_px" in result
                assert "annotation_coverage_percent" in result

                # Check that data is preserved
                assert result["annotation_path"] == rows["annotation_path"]
                assert np.array_equal(result["tile_x"], rows["tile_x"])
                assert np.array_equal(result["tile_y"], rows["tile_y"])
                assert np.array_equal(result["tile_extent_x"], rows["tile_extent_x"])
                assert np.array_equal(result["tile_extent_y"], rows["tile_extent_y"])
                assert np.array_equal(result["downsample"], rows["downsample"])

                # Check that annotation data is added
                assert np.allclose(
                    result["annotation_coverage_px"], np.array([64.0, 16.0, 16.0, 4.0])
                )
                assert np.allclose(
                    result["annotation_coverage_percent"],
                    np.array([100.0, 25.0, 25.0, 6.25]),
                )

            finally:
                os.unlink(f.name)

    def test_map_annotations_custom_roi(self, sample_geojson_content):
        """Test map_annotations with a custom region of interest."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".geojson", delete=False
        ) as f:
            json.dump(sample_geojson_content, f)
            f.flush()

            try:
                # Create sample input data
                rows = {"annotation_path": [f.name], "other_data": [42]}
                rows = {
                    "annotation_path": [f.name for _ in range(4)],
                    "tile_x": np.array([0, 8, 0, 8]),
                    "tile_y": np.array([0, 0, 8, 8]),
                    "tile_extent_x": np.array([8, 8, 8, 8]),
                    "tile_extent_y": np.array([8, 8, 8, 8]),
                    "downsample": np.array([1, 1, 1, 1]),
                }

                # Call map_annotations
                result = map_annotations(
                    rows,
                    lambda x: STRtree(list(GeoJSONParser(x).get_polygons())),
                    roi=Polygon([(1, 1), (7, 1), (7, 7), (1, 7)]),
                )

                # Check that the function returns expected structure
                assert "annotation_path" in result
                assert "tile_x" in result
                assert "tile_y" in result
                assert "tile_extent_x" in result
                assert "tile_extent_y" in result
                assert "downsample" in result
                assert "annotation_coverage_px" in result
                assert "annotation_coverage_percent" in result

                # Check that data is preserved
                assert result["annotation_path"] == rows["annotation_path"]
                assert np.array_equal(result["tile_x"], rows["tile_x"])
                assert np.array_equal(result["tile_y"], rows["tile_y"])
                assert np.array_equal(result["tile_extent_x"], rows["tile_extent_x"])
                assert np.array_equal(result["tile_extent_y"], rows["tile_extent_y"])
                assert np.array_equal(result["downsample"], rows["downsample"])

                # Check that annotation data is added
                assert np.allclose(
                    result["annotation_coverage_px"], np.array([36.0, 6.0, 6.0, 1.0])
                )
                assert np.allclose(
                    result["annotation_coverage_percent"],
                    np.array([100.0, 16.66667, 16.66667, 2.77778]),
                )

            finally:
                os.unlink(f.name)
