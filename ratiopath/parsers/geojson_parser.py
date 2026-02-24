from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

import geopandas as gpd
import pandas as pd
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
                annotations = annotations.explode(
                    index_parts=True
                )  # Decompose MultiPolygons into individual Shapely geometries

            self.gdf = gpd.GeoDataFrame(
                pd.concat([annotations, definitions], ignore_index=True),
                geometry="geometry",
            )

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

    def solve_relations(self, join_key: str) -> None:
        """Merge properties from non-geometry features into geometry features based on a join key.

        Side effects:
        - Non-geometry features (definitions) are permanently removed from self.gdf.
        - Annotations without a matching definition receive NaN values for the imported attributes.

        Args:
            join_key: The column name used to link non-geometry definitions to geometry features.
        """
        is_empty_geom = self.gdf.geometry.isna() | self.gdf.geometry.is_empty
        annotations = self.gdf[~is_empty_geom].copy()

        if join_key not in self.gdf.columns:
            self.gdf = annotations
            return

        definitions = self.gdf[is_empty_geom].copy()

        if definitions.empty or annotations.empty:
            self.gdf = annotations
            return

        if definitions[join_key].isna().all() or annotations[join_key].isna().all():
            self.gdf = annotations
            return

        if definitions[join_key].duplicated().any():
            raise ValueError(f"Duplicate definition for key '{join_key}' found.")

        definitions = definitions.drop(columns=["geometry"], errors="ignore").dropna(
            axis=1, how="all"
        )
        annotations = annotations.dropna(axis=1, how="all")

        self.gdf = gpd.GeoDataFrame(
            annotations.merge(
                definitions, on=join_key, how="left", suffixes=("_orig", "_def")
            ),
            geometry="geometry",
            crs=self.gdf.crs,
        )
