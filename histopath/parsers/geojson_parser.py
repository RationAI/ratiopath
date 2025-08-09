"""GeoJSON format annotation parser."""

import re
from typing import Any, Iterable

import geojson
import shapely
from geojson import Feature, GeoJSON
from geojson.geometry import Geometry

from histopath.parsers.abstract_parser import AbstractParser


class GeoJSONParser(AbstractParser):
    """Parser for GeoJSON format annotation files.

    GeoJSON is a format for encoding geographic data structures using JSON.
    This parser supports both polygon and point geometries.
    """

    def _parse_geometry(self, geometry: dict[str, Any]) -> list[Geometry]:
        """Parse a GeoJSON geometry into a Geometry object.

        Args:
            geometry: A dictionary representing a GeoJSON geometry.

        Returns:
            A Geometry object.
        """
        if "type" not in geometry:
            raise ValueError("Geometry must contain 'type' key")

        if geometry["type"] == "GeometryCollection":
            if "geometries" not in geometry:
                raise ValueError("GeometryCollection must contain 'geometries' key")
            geometries = []
            for geom in geometry["geometries"]:
                geometries.extend(self._parse_geometry(geom))
            return geometries

        if "coordinates" not in geometry:
            raise ValueError("Geometry must contain 'coordinates' key")

        match geometry["type"]:
            case "Point":
                return [geojson.Point(geometry["coordinates"], validate=True)]
            case "Polygon":
                return [geojson.Polygon(geometry["coordinates"], validate=True)]
            case "MultiPoint":
                return [geojson.MultiPoint(geometry["coordinates"], validate=True)]
            case "MultiPolygon":
                return [geojson.MultiPolygon(geometry["coordinates"], validate=True)]
            case _:
                raise ValueError(f"Unsupported geometry type: {geometry['type']}")

    def _parse_feature(self, feature: dict[str, Any]) -> list[Feature]:
        """Parse a single GeoJSON feature into a Feature object.

        Args:
            feature: A dictionary representing a GeoJSON feature.

        Returns:
            A Feature object.
        """
        if "geometry" not in feature or "properties" not in feature:
            raise ValueError("Feature must contain 'geometry' and 'properties' keys")
        return [
            Feature(
                geometry=geometry,
                properties=feature["properties"],
            )
            for geometry in self._parse_geometry(feature["geometry"])
        ]

    def _parse_feature_collection(
        self, feature_collection: dict[str, Any]
    ) -> list[Feature]:
        """Parse a GeoJSON feature collection into a list of Feature objects.

        Args:
            feature_collection: A dictionary representing a GeoJSON feature collection.

        Returns:
            A FeatureCollection object.
        """
        if "features" not in feature_collection:
            raise ValueError("FeatureCollection must contain 'features' key")
        features = []
        for feature in feature_collection["features"]:
            features.extend(self._parse_feature(feature))
        return features

    def _get_list_of_features(self, data: GeoJSON) -> list[Feature]:
        """Get list of features from the GeoJSON and flatten all GeometryCollections.

        Args:
            data: A GeoJSON object.

        Returns:
            A FeatureCollection object.
        """
        match data["type"]:
            case "FeatureCollection":
                return self._parse_feature_collection(data)
            case "Feature":
                return self._parse_feature(data)
            case (
                "Point"
                | "MultiPoint"
                | "Polygon"
                | "MultiPolygon"
                | "GeometryCollection"
            ):
                return [
                    Feature(
                        geometry=geometry,
                        properties={},
                    )
                    for geometry in self._parse_geometry(data)
                ]
            case _:
                raise ValueError("Unsupported GeoJSON type")

    def _get_filtered_geojson_geometries(self, **kwargs) -> Iterable[Geometry]:
        """Get geometries that match the provided filters.

        Args:
            **kwargs: Keyword arguments containing optional filters.

        Yields:
            GeoJSONGeometry objects that match the filters.
        """
        with open(self.file_path, "r") as f:
            data = GeoJSON(geojson.load(f))

        features = self._get_list_of_features(data)

        filters = {
            key: re.compile(value)
            for key, value in kwargs.items()
            if key != "separator"
        }
        separator = str(kwargs.get("separator", "_"))

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
                if not isinstance(properties, str):
                    valid = False
                    break
                if not pattern.match(properties):
                    valid = False
                    break

            if valid:
                yield feature["geometry"]  # this is not GeometryCollection!!!

    def get_polygons(self, **kwargs) -> Iterable[shapely.Polygon]:
        """Parse polygon annotations from GeoJSON file.

        Args:
            **kwargs: Optional keyword arguments for filtering annotations.

        Returns:
            An iterable of shapely Polygon objects.
        """
        for geometry in self._get_filtered_geojson_geometries(**kwargs):
            match geometry["type"]:
                case "Polygon":
                    yield shapely.Polygon(
                        geometry["coordinates"][0], geometry["coordinates"][1:]
                    )
                case "MultiPolygon":
                    for coords in geometry["coordinates"]:
                        yield shapely.Polygon(coords[0], coords[1:])
                case _:
                    pass

    def get_points(self, **kwargs) -> Iterable[shapely.Point]:
        """Parse point annotations from GeoJSON file.

        Args:
            **kwargs: Optional keyword arguments for filtering annotations.

        Returns:
            An iterable of shapely Point objects.
        """
        for geometry in self._get_filtered_geojson_geometries(**kwargs):
            match geometry["type"]:
                case "Point":
                    yield shapely.Point(geometry["coordinates"])
                case "MultiPoint":
                    for coords in geometry["coordinates"]:
                        yield shapely.Point(coords)
                case _:
                    pass
