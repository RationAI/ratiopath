import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

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
    │       └── meta: { "category": { "name": "Category", "value": "Healthy Tissue" } }
    └── Feature (Annotation)                │
        ├── geometry: { "type": "Polygon" } │
        └── properties                      │
            └── presetID: "a376..."  <──────┘
    """

    def __init__(
        self, file_path: Path | str | TextIO, join_key: str | None = "presetID"
    ) -> None:
        self.gdf = gpd.read_file(file_path)

        if not self.gdf.empty:
            has_geometry = ~(self.gdf.geometry.is_empty | self.gdf.geometry.isna())
            annotations = self.gdf[has_geometry].explode(index_parts=True)
            definitions = self.gdf[~has_geometry]

            if join_key in self.gdf.columns and not definitions.empty:
                self.gdf = self._solve_relations(annotations, definitions, join_key)  # type: ignore[arg-type]
            else:
                self.gdf = annotations

    @staticmethod
    def _solve_relations(
        annotations: GeoDataFrame, definitions: GeoDataFrame, join_key: str
    ) -> GeoDataFrame:
        """Merge definition properties into annotations using the join key.

        Columns that exist only in the definitions are folded into the result.
        Columns that exist in both get a ``_def`` suffix for the definition side.
        """
        # Drop all-null columns from annotations so they don't shadow definition values
        ann_null_cols = [
            c
            for c in annotations.columns
            if c != "geometry" and c != join_key and annotations[c].isna().all()
        ]
        annotations_clean = annotations.drop(columns=ann_null_cols)

        merged = annotations_clean.merge(
            definitions.drop(columns=["geometry"]),
            on=join_key,
            how="left",
            suffixes=("", "_def"),
        )
        return merged

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

            series = filtered_gdf[subkeys[0]]

            for subkey in subkeys[1:]:
                series = series.apply(safe_to_dict)
                mask = series.apply(
                    lambda x, sk=subkey: isinstance(x, dict) and sk in x
                )
                series = series[mask].apply(lambda x, sk=subkey: x[sk])
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


def safe_to_dict(x: str | Any) -> Any:
    """Safely converts potential JSON strings to dict, preserving existing dicts and NaNs."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except (json.JSONDecodeError, TypeError):
            return x
    return x
