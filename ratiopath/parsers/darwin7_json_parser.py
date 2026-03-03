import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry


class Darwin7JSONParser:
    """Parser for Darwin7 JSON format annotation files.

    Extracts geometries and metadata from Darwin JSON structures into a GeoDataFrame.
    Supports Polygon, Point, and bounding box geometries.
    Expected JSON schema:
    Root
    └── annotations (list)
        ├── Feature (Polygon)
        │   ├── id: "uuid"
        │   ├── name: "class_name"
        │   ├── properties: [...] or {...}
        │   ├── slot_names: [...]
        │   └── polygon
        │       └── paths: [ [ {x, y}, ... ], [holes...] ]
        ├── Feature (Bounding Box)
        │   ├── id: "uuid"
        │   └── bounding_box: {x, y, w, h}
        └── Feature (Point)
            ├── id: "uuid"
            └── point: {x, y}
    """

    def __init__(self, file_path: Path | str | TextIO) -> None:
        if isinstance(file_path, (str, Path)):
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.load(file_path)

        records = []
        for ann in data.get("annotations", []):
            props = ann.get("properties", [])
            records.append(
                {
                    "id": ann.get("id"),
                    "name": ann.get("name"),
                    "properties": json.dumps(props)
                    if isinstance(props, (list, dict))
                    else props,
                    "slot_names": json.dumps(ann.get("slot_names", [])),
                    "geometry": self._parse_geometry(ann),
                }
            )

        if records:
            self.gdf = GeoDataFrame(pd.DataFrame(records), geometry="geometry")
        else:
            self.gdf = GeoDataFrame(
                columns=["id", "name", "properties", "slot_names", "geometry"],
                geometry="geometry",
            )

    @staticmethod
    def _parse_geometry(ann: dict[str, Any]) -> BaseGeometry | None:
        """Construct Shapely geometry objects from Darwin annotation dictionaries."""
        if "polygon" in ann:
            paths = ann["polygon"].get("paths", [])
            if paths:
                exterior = [(pt["x"], pt["y"]) for pt in paths[0]]
                holes = [
                    [(pt["x"], pt["y"]) for pt in hole]
                    for hole in paths[1:]
                    if len(hole) >= 3
                ]
                if len(exterior) >= 3:
                    return Polygon(exterior, holes)
        elif "bounding_box" in ann:
            bbox = ann["bounding_box"]
            return box(
                bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]
            )
        elif "point" in ann:
            return Point(ann["point"]["x"], ann["point"]["y"])

        return None

    @staticmethod
    def extract_nested(val: Any, path: list[str]) -> Any:
        """Extract a nested value from a JSON-like structure using a list of keys."""
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except json.JSONDecodeError:
                return None
        for key in path:
            if isinstance(val, dict) and key in val:
                val = val[key]
            elif isinstance(val, list) and key.isdigit():
                idx = int(key)
                if idx < len(val):
                    val = val[idx]
                else:
                    return None
            else:
                return None
        return val

    def get_filtered_geodataframe(
        self, separator: str = "_", **kwargs: str
    ) -> GeoDataFrame:
        """Filter the GeoDataFrame based on property values.

        Supports filtering by top-level columns or nested attributes within JSON-like
        columns (e.g., 'properties'). Nested keys are accessed by joining column
        names and keys with the separator.

        Args:
            separator: String used to separate nested keys in kwargs.
            **kwargs: Keyword arguments where keys represent (possibly nested)
                columns and values are regex patterns to match.

        Returns:
            A GeoDataFrame containing only the rows that match all filter criteria.
            If a requested top-level column is missing, an empty GeoDataFrame
            with the original schema is returned.
        """
        filtered_gdf = self.gdf

        for key, pattern in kwargs.items():
            subkeys = key.split(separator)
            if not subkeys or subkeys[0] not in filtered_gdf.columns:
                return self.gdf.iloc[0:0]

            series = filtered_gdf[subkeys[0]]

            if len(subkeys) > 1:
                series = series.apply(
                    lambda x, sk=subkeys[1:]: self.extract_nested(x, sk)
                )
                mask = series.notna()
                filtered_gdf = filtered_gdf[mask]
                series = series[mask]

            if filtered_gdf.empty:
                return filtered_gdf

            series = series.astype(str)
            mask = series.str.match(pattern, na=False)
            filtered_gdf = filtered_gdf[mask]

        return filtered_gdf

    def get_polygons(self, **kwargs: str) -> Iterable[Polygon]:
        """Get polygons from the GeoDataFrame.

        Args:
            **kwargs: Keyword arguments containing regex patterns for filtering properties.

        Yields:
            Shapely Polygon objects.
        """
        filtered_gdf = self.get_filtered_geodataframe(**kwargs)
        yield from filtered_gdf[filtered_gdf.geom_type == "Polygon"].geometry

    def get_points(self, **kwargs: str) -> Iterable[Point]:
        """Get points from the GeoDataFrame.

        Args:
            **kwargs: Keyword arguments containing regex patterns for filtering properties.

        Yields:
            Shapely Point objects.
        """
        filtered_gdf = self.get_filtered_geodataframe(**kwargs)
        yield from filtered_gdf[filtered_gdf.geom_type == "Point"].geometry
