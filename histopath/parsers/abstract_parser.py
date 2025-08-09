"""Abstract base class for annotation parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import shapely.geometry


class AbstractParser(ABC):
    """Abstract base class defining a common interface for annotation parsers.

    All concrete parser implementations should inherit from this class and
    implement the required methods to parse annotations from their respective formats.
    """

    def __init__(self, file_path: Path | str) -> None:
        """Initialize the parser with a file path.

        Args:
            file_path: Path to the annotation file to be parsed.
        """
        self.file_path = Path(file_path)

    @abstractmethod
    def get_polygons(self, **kwargs) -> Iterable[shapely.geometry.Polygon]:
        """Returns all polygon annotations from the annotation file.

        Args:
            **kwargs: Optional keyword arguments for filtering annotations.

        Returns:
            An iterable of shapely Polygon objects representing the polygon annotations.
        """
        pass

    @abstractmethod
    def get_points(self, **kwargs) -> Iterable[shapely.geometry.Point]:
        """Returns all point annotations from the annotation file.

        Args:
            **kwargs: Optional keyword arguments for filtering annotations.

        Returns:
            An iterable of shapely Point objects representing the point annotations.
        """
        pass
