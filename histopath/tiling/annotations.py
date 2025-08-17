from typing import Iterator

from shapely import Polygon, STRtree, transform
from shapely.geometry.base import BaseGeometry


def shift_roi(roi: BaseGeometry, x: int, y: int, downsample: float) -> BaseGeometry:
    """Shift the region of interest (ROI) for a specific tile.

    Args:
        roi: The current region of interest geometry.
        x: The x-coordinate of the tile to shift the ROI for.
        y: The y-coordinate of the tile to shift the ROI for.
        downsample: The downsampling factor.

    Returns:
        BaseGeometry: The shifted region of interest geometry.
    """
    return transform(roi, lambda geom: geom + [x * downsample, y * downsample])


def annotations_intersection(tree: STRtree, roi: BaseGeometry) -> Polygon:
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
    tree: STRtree, roi: BaseGeometry, x: list[int], y: list[int], downsample: float
) -> Iterator[Polygon]:
    """Yield annotated tiles from the annotation tree.

    Annotations are assumed to be at level 0. Yielded geometries are transformed to the
    same level as the ROI and tiles.

    Args:
        tree: The spatial index of annotation geometries.
        roi: The region of interest geometry.
        x: The x-coordinates of the tiles.
        y: The y-coordinates of the tiles.
        downsample: The downsampling factor.

    Yields:
        BaseGeometry: The annotated tile geometry.
    """
    # transform roi to level 0
    roi = transform(roi, lambda geom: geom * downsample)

    for tile_x, tile_y in zip(x, y):
        shifted_roi = shift_roi(roi, tile_x, tile_y, downsample)
        intersection = annotations_intersection(tree, shifted_roi)
        yield transform(intersection, lambda geom: geom / downsample)
