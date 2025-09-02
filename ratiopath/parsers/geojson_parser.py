import json
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

import geopandas as gpd
from geopandas import GeoDataFrame
from shapely import Point, Polygon


class GeoJSONParser:
    """Parser for GeoJSON format annotation files.

    GeoJSON is a format for encoding geographic data structures using JSON.
    This parser supports both polygon and point geometries.
    """

    def __init__(self, file_path: Path | str | TextIO) -> None:
        self.gdf = gpd.read_file(file_path)

        if not self.gdf.empty:
            # Explode Multi-part geometries to simplify geometry handling
            self.gdf = self.gdf.explode(index_parts=True)

    def get_filtered_geodataframe(
        self, separator: str = "_", **kwargs: str
    ) -> GeoDataFrame:
        """Filter the GeoDataFrame based on property values.

        Args:
            separator: The string used to separate keys in the filtering.
            **kwargs: Keyword arguments for filtering. Keys are column names
                (e.g., 'classification.name') and values are regex patterns to match
                against.

        Returns:
            A filtered GeoDataFrame.
        """
        filtered_gdf = self.gdf
        for key, pattern in kwargs.items():
            subkeys = key.split(separator)
            if not subkeys or subkeys[0] not in filtered_gdf.columns:
                # If the first part of the key doesn't exist, return an empty frame
                return self.gdf.iloc[0:0]

            series = filtered_gdf[subkeys[0]].astype(str)
            if len(subkeys) > 1:
                mask = series.apply(is_json_dict)
                series = series[mask].apply(lambda x: json.loads(x))
                filtered_gdf = filtered_gdf[mask]

            for subkey in subkeys[1:]:
                mask = series.apply(
                    lambda x, subkey=subkey: isinstance(x, dict) and subkey in x
                )
                series = series[mask].apply(lambda x, subkey=subkey: x[subkey])
                filtered_gdf = filtered_gdf[mask]

            series = series.astype(str)
            mask = series.str.match(pattern, na=False)
            filtered_gdf = filtered_gdf[mask]

        return filtered_gdf

    def get_polygons(self, **kwargs: str) -> Iterable[Polygon]:
        """Get polygons from the GeoDataFrame.

        Args:
            **kwargs: Keyword arguments for filtering properties.

        Yields:
            Shapely Polygon objects.
        """
        filtered_gdf = self.get_filtered_geodataframe(**kwargs)
        for geom in filtered_gdf.geometry:
            if isinstance(geom, Polygon):
                yield geom

    def get_points(self, **kwargs: str) -> Iterable[Point]:
        """Get points from the GeoDataFrame.

        Args:
            **kwargs: Keyword arguments for filtering properties.

        Yields:
            Shapely Point objects.
        """
        filtered_gdf = self.get_filtered_geodataframe(**kwargs)
        for geom in filtered_gdf.geometry:
            if isinstance(geom, Point):
                yield geom


def is_json_dict(string: str) -> bool:
    try:
        valid_json = json.loads(string)
        if isinstance(valid_json, dict):
            return True
    except ValueError:
        return False
    return False
