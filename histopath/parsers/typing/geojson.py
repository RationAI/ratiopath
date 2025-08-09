from enum import Enum
from typing import Any, TypedDict


class GeoJSONType(Enum):
    FEATURE = "Feature"
    FEATURE_COLLECTION = "FeatureCollection"
    POINT = "Point"
    MULTIPOINT = "MultiPoint"
    LINESTRING = "LineString"
    MULTILINESTRING = "MultiLineString"
    POLYGON = "Polygon"
    MULTIPOLYGON = "MultiPolygon"
    GEOMETRY_COLLECTION = "GeometryCollection"


class GeoJSONObject(TypedDict):
    type: GeoJSONType


class GeoJSONGeometry(GeoJSONObject):
    coordinates: list[Any]


class GeoJSONGeometryCollection(GeoJSONObject):
    geometries: list[GeoJSONGeometry]


class GeoJSONFeature(GeoJSONObject):
    geometry: GeoJSONGeometry | GeoJSONGeometryCollection
    properties: dict[str, Any]


class GeoJSONFeatureCollection(GeoJSONObject):
    features: list[GeoJSONFeature]
