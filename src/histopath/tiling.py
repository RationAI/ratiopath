"""
Image tiling functionality for histopathology images.

This module provides functions and classes for tiling large histopathology images
into smaller, manageable patches for analysis and processing.
"""

from dataclasses import dataclass
from typing import Iterator, List, Tuple, Union, cast
import numpy as np
from PIL import Image


@dataclass
class TileConfig:
    """
    Configuration for image tiling operations.

    This class defines the parameters for tiling an image into smaller patches.

    Attributes:
        tile_size: Size of each tile in pixels (width, height). If int, creates square tiles.
        overlap: Overlap between adjacent tiles in pixels. Can be int for uniform overlap
                or tuple for (x_overlap, y_overlap).
        padding_mode: How to handle image borders. Options: 'reflect', 'constant', 'edge'.
        padding_value: Value to use for padding when padding_mode='constant'.
        include_partial: Whether to include partial tiles at image edges.

    Example:
        >>> config = TileConfig(tile_size=512, overlap=64)
        >>> config = TileConfig(tile_size=(256, 256), overlap=(32, 32))
    """

    tile_size: Union[int, Tuple[int, int]]
    overlap: Union[int, Tuple[int, int]] = 0
    padding_mode: str = "reflect"
    padding_value: int = 0
    include_partial: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize configuration parameters."""
        # Normalize tile_size to tuple
        if isinstance(self.tile_size, int):
            self.tile_size = (self.tile_size, self.tile_size)

        # Normalize overlap to tuple
        if isinstance(self.overlap, int):
            self.overlap = (self.overlap, self.overlap)

        # Validate parameters
        if any(size <= 0 for size in self.tile_size):
            raise ValueError("Tile size must be positive")

        if any(overlap < 0 for overlap in self.overlap):
            raise ValueError("Overlap cannot be negative")

        if any(overlap >= size for overlap, size in zip(self.overlap, self.tile_size)):
            raise ValueError("Overlap must be smaller than tile size")

        if self.padding_mode not in ["reflect", "constant", "edge"]:
            raise ValueError(
                "Invalid padding mode. Must be 'reflect', 'constant', or 'edge'"
            )


def tile_image(
    image: Union[np.ndarray, Image.Image, str],
    config: TileConfig,
    return_coordinates: bool = False,
) -> List[Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
    """
    Tile an image into smaller patches according to the specified configuration.

    This function takes a large image and divides it into smaller tiles based on
    the provided configuration. It supports overlapping tiles and various padding
    modes for handling image boundaries.

    Args:
        image: Input image. Can be a numpy array, PIL Image, or file path.
        config: TileConfig object specifying tiling parameters.
        return_coordinates: If True, returns (tile, coordinates) tuples where
                          coordinates are (x_start, y_start, x_end, y_end).

    Returns:
        List of image tiles as numpy arrays, or list of (tile, coordinates) tuples
        if return_coordinates=True.

    Raises:
        ValueError: If image dimensions are invalid or config is invalid.
        FileNotFoundError: If image path does not exist.

    Example:
        >>> import numpy as np
        >>> from PIL import Image
        >>>
        >>> # Create sample image
        >>> image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        >>>
        >>> # Configure tiling
        >>> config = TileConfig(tile_size=256, overlap=32)
        >>>
        >>> # Generate tiles
        >>> tiles = tile_image(image, config)
        >>> print(f"Generated {len(tiles)} tiles")
        >>>
        >>> # Get tiles with coordinates
        >>> tiles_with_coords = tile_image(image, config, return_coordinates=True)
        >>> for tile, (x1, y1, x2, y2) in tiles_with_coords:
        >>>     print(f"Tile shape: {tile.shape}, coordinates: ({x1}, {y1}, {x2}, {y2})")
    """
    # Load and validate image
    img_array = _load_image(image)

    if img_array.ndim not in [2, 3]:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")

    height, width = img_array.shape[:2]
    tile_size_tuple = cast(Tuple[int, int], config.tile_size)
    overlap_tuple = cast(Tuple[int, int], config.overlap)
    tile_width, tile_height = tile_size_tuple
    overlap_x, overlap_y = overlap_tuple

    # Calculate step sizes (distance between tile origins)
    step_x = tile_width - overlap_x
    step_y = tile_height - overlap_y

    tiles: List[Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int, int, int]]]] = []

    # Generate tile coordinates
    y_positions = list(range(0, height, step_y))
    x_positions = list(range(0, width, step_x))

    # Add final positions if we want partial tiles
    if config.include_partial:
        if y_positions[-1] + tile_height < height:
            y_positions.append(height - tile_height)
        if x_positions[-1] + tile_width < width:
            x_positions.append(width - tile_width)

    for y_start in y_positions:
        for x_start in x_positions:
            # Calculate tile boundaries
            x_end = min(x_start + tile_width, width)
            y_end = min(y_start + tile_height, height)

            # Skip if tile would be too small (when not including partial tiles)
            if not config.include_partial:
                if (x_end - x_start) < tile_width or (y_end - y_start) < tile_height:
                    continue

            # Extract tile
            tile = _extract_tile(
                img_array,
                x_start,
                y_start,
                x_end,
                y_end,
                tile_width,
                tile_height,
                config,
            )

            if return_coordinates:
                tiles.append((tile, (x_start, y_start, x_end, y_end)))
            else:
                tiles.append(tile)

    return tiles


def _load_image(image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
    """
    Load an image from various input formats into a numpy array.

    Args:
        image: Input image in various formats.

    Returns:
        Image as numpy array.

    Raises:
        ValueError: If input format is not supported.
        FileNotFoundError: If file path does not exist.
    """
    if isinstance(image, np.ndarray):
        return image.copy()
    elif isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, str):
        try:
            pil_image = Image.open(image)
            return np.array(pil_image)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image}")
        except Exception as e:
            raise ValueError(f"Failed to load image from {image}: {e}")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def _extract_tile(
    img_array: np.ndarray,
    x_start: int,
    y_start: int,
    x_end: int,
    y_end: int,
    target_width: int,
    target_height: int,
    config: TileConfig,
) -> np.ndarray:
    """
    Extract a tile from an image array with padding if necessary.

    Args:
        img_array: Source image array.
        x_start, y_start, x_end, y_end: Tile boundaries.
        target_width, target_height: Desired tile dimensions.
        config: Tiling configuration.

    Returns:
        Extracted tile as numpy array.
    """
    # Extract the basic tile
    tile = img_array[y_start:y_end, x_start:x_end]

    # Check if padding is needed
    actual_height, actual_width = tile.shape[:2]

    if actual_width == target_width and actual_height == target_height:
        return tile

    # Apply padding if tile is smaller than target size
    pad_right = target_width - actual_width
    pad_bottom = target_height - actual_height

    if tile.ndim == 3:
        pad_width = ((0, pad_bottom), (0, pad_right), (0, 0))  # type: ignore[assignment]
    else:
        pad_width = ((0, pad_bottom), (0, pad_right))  # type: ignore[assignment]

    if config.padding_mode == "constant":
        tile = np.pad(
            tile, pad_width, mode="constant", constant_values=config.padding_value
        )
    elif config.padding_mode == "reflect":
        tile = np.pad(tile, pad_width, mode="reflect")
    elif config.padding_mode == "edge":
        tile = np.pad(tile, pad_width, mode="edge")

    return tile


def get_tile_iterator(
    image: Union[np.ndarray, Image.Image, str],
    config: TileConfig,
    return_coordinates: bool = False,
) -> Iterator[Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int, int, int]]]]:
    """
    Create an iterator that yields image tiles one at a time.

    This function is memory-efficient for processing very large images as it
    yields tiles one at a time instead of loading all tiles into memory.

    Args:
        image: Input image. Can be a numpy array, PIL Image, or file path.
        config: TileConfig object specifying tiling parameters.
        return_coordinates: If True, yields (tile, coordinates) tuples.

    Yields:
        Image tiles as numpy arrays, or (tile, coordinates) tuples if
        return_coordinates=True.

    Example:
        >>> config = TileConfig(tile_size=256, overlap=32)
        >>> for tile in get_tile_iterator("large_image.tiff", config):
        >>>     # Process each tile
        >>>     processed_tile = some_processing_function(tile)
    """
    # Load image once
    img_array = _load_image(image)

    if img_array.ndim not in [2, 3]:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")

    height, width = img_array.shape[:2]
    tile_size_tuple = cast(Tuple[int, int], config.tile_size)
    overlap_tuple = cast(Tuple[int, int], config.overlap)
    tile_width, tile_height = tile_size_tuple
    overlap_x, overlap_y = overlap_tuple

    # Calculate step sizes
    step_x = tile_width - overlap_x
    step_y = tile_height - overlap_y

    # Generate tile coordinates
    y_positions = list(range(0, height, step_y))
    x_positions = list(range(0, width, step_x))

    # Add final positions if we want partial tiles
    if config.include_partial:
        if y_positions[-1] + tile_height < height:
            y_positions.append(height - tile_height)
        if x_positions[-1] + tile_width < width:
            x_positions.append(width - tile_width)

    for y_start in y_positions:
        for x_start in x_positions:
            # Calculate tile boundaries
            x_end = min(x_start + tile_width, width)
            y_end = min(y_start + tile_height, height)

            # Skip if tile would be too small (when not including partial tiles)
            if not config.include_partial:
                if (x_end - x_start) < tile_width or (y_end - y_start) < tile_height:
                    continue

            # Extract tile
            tile = _extract_tile(
                img_array,
                x_start,
                y_start,
                x_end,
                y_end,
                tile_width,
                tile_height,
                config,
            )

            if return_coordinates:
                yield (tile, (x_start, y_start, x_end, y_end))
            else:
                yield tile


def calculate_tile_count(image_shape: Tuple[int, int], config: TileConfig) -> int:
    """
    Calculate the number of tiles that would be generated for an image.

    Args:
        image_shape: Image dimensions as (height, width).
        config: TileConfig object specifying tiling parameters.

    Returns:
        Number of tiles that would be generated.

    Example:
        >>> config = TileConfig(tile_size=256, overlap=32)
        >>> count = calculate_tile_count((1024, 1024), config)
        >>> print(f"Will generate {count} tiles")
    """
    height, width = image_shape
    tile_size_tuple = cast(Tuple[int, int], config.tile_size)
    overlap_tuple = cast(Tuple[int, int], config.overlap)
    tile_width, tile_height = tile_size_tuple
    overlap_x, overlap_y = overlap_tuple

    # Calculate step sizes
    step_x = tile_width - overlap_x
    step_y = tile_height - overlap_y

    # Calculate number of tiles in each dimension
    tiles_x = (width + step_x - 1) // step_x  # Ceiling division
    tiles_y = (height + step_y - 1) // step_y  # Ceiling division

    # Adjust for partial tiles
    if config.include_partial:
        # Check if we need an extra tile at the edges
        if (tiles_x - 1) * step_x + tile_width < width:
            tiles_x += 1
        if (tiles_y - 1) * step_y + tile_height < height:
            tiles_y += 1

    return tiles_x * tiles_y
