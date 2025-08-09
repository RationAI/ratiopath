"""GeoJSON format annotation parser."""

import json
import re
from typing import Iterable, cast

from shapely.geometry import Point, Polygon

from histopath.parsers.abstract_parser import AbstractParser
from histopath.parsers.typing.geojson import (
    GeoJSONFeature,
    GeoJSONFeatureCollection,
    GeoJSONGeometry,
    GeoJSONGeometryCollection,
    GeoJSONObject,
    GeoJSONType,
)


class GeoJSONParser(AbstractParser):
    """Parser for GeoJSON format annotation files.

    GeoJSON is a format for encoding geographic data structures using JSON.
    This parser supports both polygon and point geometries.
    """

    def _get_filtered_geojson_geometries(self, **kwargs) -> Iterable[GeoJSONGeometry]:
        """Get geometries that match the provided filters.

        Args:
            **kwargs: Keyword arguments containing optional filters.

        Yields:
            GeoJSONGeometry objects that match the filters.
        """
        with open(self.file_path, "r") as f:
            data = cast(GeoJSONObject, json.load(f))

        # Handle different GeoJSON structures
        features: list[GeoJSONFeature]
        geojson_type = GeoJSONType(data["type"])
        match geojson_type:
            case GeoJSONType.FEATURE_COLLECTION:
                features = cast(GeoJSONFeatureCollection, data)["features"]
            case GeoJSONType.FEATURE:
                features = [cast(GeoJSONFeature, data)]
            case (
                GeoJSONType.POLYGON
                | GeoJSONType.MULTIPOLYGON
                | GeoJSONType.POINT
                | GeoJSONType.MULTIPOINT
            ):
                features = [
                    GeoJSONFeature(
                        type=geojson_type,
                        geometry=cast(GeoJSONGeometry, data),
                        properties={},
                    )
                ]
            case GeoJSONType.GEOMETRY_COLLECTION:
                features = [
                    GeoJSONFeature(type=geojson_type, geometry=geometry, properties={})
                    for geometry in cast(GeoJSONGeometryCollection, data)["geometries"]
                ]
            case _:
                raise ValueError("Unsupported GeoJSON type")

        filters = {
            key: re.compile(value)
            for key, value in kwargs.items()
            if key != "separator"
        }
        separator = str(filters.get("separator", "."))

        for feature in features:
            valid = True
            for key, pattern in filters.items():
                subkeys = key.split(separator)
                properties = feature["properties"]
                for subkey in subkeys:
                    if subkey not in properties:
                        valid = False
                        break
                    properties = properties[subkey]
                if not pattern.match(str(properties)):
                    valid = False
                    break

            if valid:
                geometry = feature["geometry"]
                yield cast(GeoJSONGeometry, geometry)

    def get_polygons(self, **kwargs) -> Iterable[Polygon]:
        """Parse polygon annotations from GeoJSON file.

        Returns:
            An iterable of shapely Polygon objects.
        """
        for geometry in self._get_filtered_geojson_geometries(**kwargs):
            match geometry["type"]:
                case GeoJSONType.POLYGON:
                    yield Polygon(
                        geometry["coordinates"][0], geometry["coordinates"][1:]
                    )
                case GeoJSONType.MULTIPOLYGON:
                    for coords in geometry["coordinates"]:
                        yield Polygon(coords[0], coords[1:])
                case _:
                    pass

    def get_points(self, **kwargs) -> Iterable[Point]:
        """Parse point annotations from GeoJSON file.

        Returns:
            An iterable of shapely Point objects.
        """
        for geometry in self._get_filtered_geojson_geometries(**kwargs):
            match geometry["type"]:
                case GeoJSONType.POINT:
                    yield Point(geometry["coordinates"])
                case GeoJSONType.MULTIPOINT:
                    for coords in geometry["coordinates"]:
                        yield Point(coords)
                case _:
                    pass
