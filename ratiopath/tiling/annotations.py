from collections.abc import Iterable, Iterator

import numpy as np
from numpy.typing import NDArray
from shapely import Polygon, STRtree, transform
from shapely.geometry.base import BaseGeometry


def shift_roi(
    roi: BaseGeometry, coordinate: NDArray[np.int64], downsample: float
) -> BaseGeometry:
    """Shift the region of interest (ROI) for a specific tile.

    Args:
        roi: The current region of interest geometry.
        coordinate: The coordinates of the tile to shift the ROI for.
        downsample: The downsampling factor.

    Returns:
        BaseGeometry: The shifted region of interest geometry.
    """
    return transform(roi, lambda r: r + coordinate * downsample)


def annotations_intersection(tree: STRtree, roi: BaseGeometry) -> BaseGeometry:
    """Get the intersection of annotations with the region of interest (ROI).

    Args:
        tree: The spatial index of annotation geometries.
        roi: The region of interest geometry.

    Returns:
        BaseGeometry: The intersection of annotations with the ROI.
    """
    intersection = Polygon()
    for polygon_index in tree.query(roi, predicate="intersects"):
        intersection = intersection.union(
            tree.geometries[int(polygon_index)].intersection(roi)
        )
    return intersection


def tile_annotations(
    annotations: Iterable[BaseGeometry],
    roi: BaseGeometry,
    coordinates: Iterable[NDArray[np.int64]],
    downsample: float,
) -> Iterator[BaseGeometry]:
    """Yield annotated tiles from the annotation tree.

    Annotations are assumed to be at level 0. Yielded geometries are transformed to the
    same level as the ROI and tiles.

    Args:
        annotations: The list of annotation geometries.
        roi: The region of interest geometry.
        coordinates: The iterable of coordinates of the tiles.
        downsample: The downsampling factor.

    Yields:
        BaseGeometry: The annotated tile geometry.
    """
    tree = STRtree(annotations)
    # transform roi to level 0
    roi = transform(roi, lambda geom: geom * downsample)

    for coordinate in coordinates:
        shifted_roi = shift_roi(roi, coordinate, downsample)
        intersection = annotations_intersection(tree, shifted_roi)
        yield transform(intersection, lambda geom: geom / downsample)
