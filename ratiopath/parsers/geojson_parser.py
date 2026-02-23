import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely import Point, Polygon


class GeoJSONParser:
    """Parser for GeoJSON format annotation files.

    GeoJSON is a format for encoding geographic data structures using JSON.
    This parser supports both polygon and point geometries.

    Extended capabilities:
    - Relational metadata integration: Maps properties from geometry-less definition
      features to spatial annotation features via a shared join key (solve_relations).

    Expected relational schema for solve_relations:
    FeatureCollection
    ├── Feature (Definition)
    │   ├── geometry: null
    │   └── properties
    │       ├── presetID: "a376..."  <──────┐ (join_key)
    │       └── meta: { "Category": "..." } │
    └── Feature (Annotation)                │
        ├── geometry: { "type": "Polygon" } │
        └── properties                      │
            └── presetID: "a376..."  <──────┘
    """

    def __init__(self, file_path: Path | str | TextIO) -> None:
        self.gdf = gpd.read_file(file_path)

        if not self.gdf.empty:
            # Isolate definitions (no geometry) from physical annotations
            has_null_geometry = self.gdf.geometry.isna() | self.gdf.geometry.is_empty
            definitions = self.gdf[has_null_geometry]
            annotations = self.gdf[~has_null_geometry]

            if not annotations.empty:
                annotations = annotations.explode(index_parts=True) # Decompose MultiPolygons into individual Shapely geometries

            self.gdf = gpd.GeoDataFrame(pd.concat([annotations, definitions], ignore_index=True), geometry="geometry")



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
                # If the first part of the key doesn't exist, return an empty frame with "geometry" column
                return gpd.GeoDataFrame(self.gdf.iloc[0:0], geometry="geometry")

            series = filtered_gdf[subkeys[0]]
            if len(subkeys) > 1:
                mask = series.apply(is_json_dict)
                series = series[mask].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                filtered_gdf = filtered_gdf[mask]

            # Protection against Pandas dropping all columns when applying masks to 0-row DataFrames
            if filtered_gdf.empty:
                return filtered_gdf

            for subkey in subkeys[1:]:
                mask = series.apply(
                    lambda x, subkey=subkey: isinstance(x, dict) and subkey in x
                )
                series = series[mask].apply(lambda x, subkey=subkey: x[subkey])
                filtered_gdf = filtered_gdf[mask]

                if filtered_gdf.empty:
                    return filtered_gdf

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

    def solve_relations(self, join_key: str) -> None:
        """Merge properties from non-geometry features into geometry features based on a join key.

        Args:
            join_key: The column name used to link non-geometry definitions to geometry features.
        """
        if join_key not in self.gdf.columns:
            return

        is_empty_geom = self.gdf.geometry.isna() | self.gdf.geometry.is_empty
        definitions = self.gdf[is_empty_geom].drop(columns=["geometry"], errors="ignore").dropna(axis=1, how="all")
        annotations = self.gdf[~is_empty_geom]

        if definitions.empty or annotations.empty:
            return

        # Suffixes prevent naming conflicts; empty attributes in annotations become '_orig'
        merged_df = annotations.merge(
            definitions,
            on=join_key,
            how="left",
            suffixes=("_orig", "")
        )

        self.gdf = gpd.GeoDataFrame(merged_df, geometry="geometry")

def is_json_dict(obj: Any) -> bool:
    if isinstance(obj, dict):
        return True
    if isinstance(obj, str):
        try:
            return isinstance(json.loads(obj), dict)
        except json.JSONDecodeError:
            return False
    return False
