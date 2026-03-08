"""Unified mask builder with configurable storage, aggregation, and processing."""

from __future__ import annotations

import contextlib
import logging
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from jaxtyping import Int64, Shaped

from ratiopath.masks.mask_builders.mask_builder import (
    AccumulatorType,
    compute_acc_slices,
)


logger = logging.getLogger(__name__)


StorageType = Literal["memmap", "in-memory"]
AggregationType = Literal["mean", "max"]


class MaskBuilder:
    """Unified mask builder with configurable storage, aggregation, and processing pipeline.

    This class combines all mask-building functionality — storage, aggregation, coordinate
    scaling, scalar expansion, and edge clipping — into a single class configured via
    init arguments, replacing the previous mixin-based architecture.

    Args:
        storage: Storage backend for the accumulator.
            - ``"in-memory"``: numpy array backed by RAM (fast, limited by RAM size).
            - ``"memmap"``: memory-mapped file backed by disk (handles masks larger than RAM).
        aggregation: How overlapping tiles are combined.
            - ``"mean"``: averages tile values, tracking per-pixel overlap counts.
            - ``"max"``: keeps the maximum value at each position.
        auto_scale: If ``True``, automatically scales coordinates from source image resolution
            to mask resolution. Requires ``source_extents``, ``source_tile_extents``,
            ``source_tile_strides``, and ``mask_tile_extents``; mask extents are computed
            automatically. If ``False``, ``mask_extents`` must be provided directly and
            coordinates are used as-is.
        expand_scalars: If ``True``, expects per-tile scalar/vector inputs of shape ``(B, C)``
            and expands them into uniform tile regions using GCD-based compression. Requires
            ``mask_tile_extents``. When ``auto_scale=False``, also requires ``mask_tile_strides``.
            Edge clipping (``px_to_clip``) is ignored when ``expand_scalars=True``.
        mask_extents: Spatial dimensions of the output mask. Required when ``auto_scale=False``.
        source_extents: Spatial dimensions of the source image. Required when ``auto_scale=True``.
        source_tile_extents: Size of input tiles in source space. Required when ``auto_scale=True``.
        source_tile_strides: Stride between input tiles in source space. Required when
            ``auto_scale=True``.
        mask_tile_extents: Size of each tile in mask space. Required when ``auto_scale=True``
            or ``expand_scalars=True``.
        mask_tile_strides: Stride between tiles in mask space. Required when
            ``expand_scalars=True`` and ``auto_scale=False``.
        channels: Number of channels in the mask.
        px_to_clip: Pixels to remove from tile edges before accumulation. Ignored when
            ``expand_scalars=True``. Accepts:

            - ``int``: same clipping on all edges in all dimensions.
            - ``tuple`` of N ints: symmetric per-dimension clipping.
            - ``tuple`` of 2N ints: per-edge clipping ``(start_dim0, end_dim0, start_dim1, end_dim1, ...)``.
        accumulator_filepath: Path for the memmap file backing the accumulator. Only used
            when ``storage="memmap"``. If ``None``, a temporary file is created and deleted
            when the builder is garbage-collected.
        overlap_counter_filepath: Path for the overlap counter memmap. Only used when
            ``storage="memmap"`` and ``aggregation="mean"``. If ``None``, a temporary file
            is created (or auto-derived as ``<accumulator_filepath>.overlaps<suffix>``
            when ``accumulator_filepath`` is given).
        dtype: Data type for the accumulator array.

    Attributes:
        accumulator: The main accumulator array.
        overlap_counter: Per-pixel overlap count array. Only present when
            ``aggregation="mean"``.
        overflow_buffered_source_extents: Source extents padded to cover partial edge tiles.
            Only present when ``auto_scale=True``.
    """

    accumulator: AccumulatorType

    def __init__(
        self,
        *,
        storage: StorageType = "in-memory",
        aggregation: AggregationType = "mean",
        auto_scale: bool = False,
        expand_scalars: bool = False,
        mask_extents: Int64[AccumulatorType, " N"] | None = None,
        source_extents: Int64[AccumulatorType, " N"] | None = None,
        source_tile_extents: Int64[AccumulatorType, " N"] | None = None,
        source_tile_strides: Int64[AccumulatorType, " N"] | None = None,
        mask_tile_extents: Int64[AccumulatorType, " N"] | None = None,
        mask_tile_strides: Int64[AccumulatorType, " N"] | None = None,
        channels: int,
        px_to_clip: int | tuple[int, ...] = 0,
        accumulator_filepath: Path | None = None,
        overlap_counter_filepath: Path | None = None,
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        self._storage = storage
        self._aggregation = aggregation
        self._auto_scale = auto_scale
        self._expand_scalars = expand_scalars

        # Memmap cleanup tracking
        self._memmap_files_to_be_deleted: list[Path] = []
        self._memmap_accumulators_to_be_closed: list[np.memmap] = []

        # --- Resolve effective mask extents and tile parameters ---
        if auto_scale:
            if any(
                x is None
                for x in [
                    source_extents,
                    source_tile_extents,
                    source_tile_strides,
                    mask_tile_extents,
                ]
            ):
                raise ValueError(
                    "auto_scale=True requires source_extents, source_tile_extents, "
                    "source_tile_strides, and mask_tile_extents."
                )
            _source_extents = np.asarray(source_extents, dtype=np.int64)
            _source_tile_extents = np.asarray(source_tile_extents, dtype=np.int64)
            _source_tile_strides = np.asarray(source_tile_strides, dtype=np.int64)
            _mask_tile_extents = np.asarray(mask_tile_extents, dtype=np.int64)

            self._source_tile_extents = _source_tile_extents
            self._mask_tile_extents = _mask_tile_extents

            multiplied_ = _source_tile_strides * _mask_tile_extents
            if not np.all(multiplied_ % _source_tile_extents == 0):
                raise ValueError(
                    f"source_tile_strides * mask_tile_extents must be divisible by "
                    f"source_tile_extents in all dimensions, but "
                    f"{_source_tile_strides}*{_mask_tile_extents}={multiplied_}, "
                    f"which is not divisible by {_source_tile_extents}."
                )
            _adjusted_mask_tile_strides = multiplied_ // _source_tile_extents

            total_strides = np.ceil(
                (_source_extents - _source_tile_extents) / _source_tile_strides
            ).astype(np.int64)
            self.overflow_buffered_source_extents: Int64[AccumulatorType, " N"] = (
                total_strides * _source_tile_strides + _source_tile_extents
            )
            effective_mask_extents = (
                total_strides * _adjusted_mask_tile_strides + _mask_tile_extents
            )
            effective_mask_tile_strides = _adjusted_mask_tile_strides
        else:
            if mask_extents is None:
                raise ValueError("auto_scale=False requires mask_extents.")
            effective_mask_extents = np.asarray(mask_extents, dtype=np.int64)

            if expand_scalars:
                if mask_tile_extents is None or mask_tile_strides is None:
                    raise ValueError(
                        "expand_scalars=True with auto_scale=False requires "
                        "mask_tile_extents and mask_tile_strides."
                    )
                effective_mask_tile_strides = np.asarray(mask_tile_strides, dtype=np.int64)
                mask_tile_extents = np.asarray(mask_tile_extents, dtype=np.int64)
            else:
                effective_mask_tile_strides = None

        # --- Handle expand_scalars GCD compression ---
        if expand_scalars:
            if mask_tile_extents is None:
                raise ValueError("expand_scalars=True requires mask_tile_extents.")
            _mask_tile_extents_arr = np.asarray(mask_tile_extents, dtype=np.int64)
            _effective_strides = np.asarray(effective_mask_tile_strides, dtype=np.int64)
            self._compression_factors: Int64[AccumulatorType, " N"] = np.gcd(
                _effective_strides, _mask_tile_extents_arr
            )
            self._adjusted_tile_extents: Int64[AccumulatorType, " N"] = (
                _mask_tile_extents_arr // self._compression_factors
            )
            effective_mask_extents = effective_mask_extents // self._compression_factors

        # --- Handle edge clipping ---
        num_dims = len(effective_mask_extents)
        if not expand_scalars:
            if isinstance(px_to_clip, int):
                self._clip_start = np.full(num_dims, px_to_clip, dtype=np.int64)
                self._clip_end = np.full(num_dims, px_to_clip, dtype=np.int64)
            elif isinstance(px_to_clip, tuple) and len(px_to_clip) == num_dims:
                self._clip_start = np.asarray(px_to_clip, dtype=np.int64)
                self._clip_end = np.asarray(px_to_clip, dtype=np.int64)
            elif isinstance(px_to_clip, tuple) and len(px_to_clip) == 2 * num_dims:
                self._clip_start = np.asarray(px_to_clip[::2], dtype=np.int64)
                self._clip_end = np.asarray(px_to_clip[1::2], dtype=np.int64)
            else:
                raise ValueError(
                    f"px_to_clip must be an int, a tuple of {num_dims} ints, "
                    f"or a tuple of {2 * num_dims} ints."
                )
        else:
            self._clip_start = np.zeros(num_dims, dtype=np.int64)
            self._clip_end = np.zeros(num_dims, dtype=np.int64)

        # --- Allocate accumulators ---
        if storage == "memmap":
            self.accumulator = self._allocate_memmap(
                effective_mask_extents,
                channels,
                dtype=dtype,
                filepath=accumulator_filepath,
            )
            if aggregation == "mean":
                if overlap_counter_filepath is None and accumulator_filepath is not None:
                    suffix = accumulator_filepath.suffix
                    counter_path: Path | None = accumulator_filepath.with_suffix(
                        f".overlaps{suffix}"
                    )
                else:
                    counter_path = overlap_counter_filepath
                self.overlap_counter: AccumulatorType = self._allocate_memmap(
                    effective_mask_extents,
                    1,
                    dtype=np.uint16,
                    filepath=counter_path,
                )
        else:  # in-memory
            self.accumulator = np.zeros(
                (channels, *effective_mask_extents), dtype=dtype
            )
            if aggregation == "mean":
                self.overlap_counter = np.zeros(
                    (1, *effective_mask_extents), dtype=np.uint16
                )

    def _allocate_memmap(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: npt.DTypeLike = np.float32,
        filepath: Path | None = None,
    ) -> np.memmap:
        """Allocate a numpy memmap array, optionally backed by a temporary file."""
        if filepath is None:
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                temp_filename = temp_file.name
                self._memmap_files_to_be_deleted.append(Path(temp_filename))
            mmap = np.lib.format.open_memmap(
                temp_filename,
                mode="w+",
                shape=(channels, *mask_extents),
                dtype=dtype,
                version=(3, 0),
            )
            self._memmap_accumulators_to_be_closed.append(mmap)
            return mmap
        if filepath.exists():
            raise FileExistsError(f"Memmap filepath {filepath} already exists.")
        return np.lib.format.open_memmap(
            filepath,
            mode="w+",
            shape=(channels, *mask_extents),
            dtype=dtype,
            version=(3, 0),
        )

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._cleanup_memmaps()

    def _cleanup_memmaps(self) -> None:
        """Delete any temporary memmap files when the builder is garbage-collected."""
        for filepath, mmap in zip(
            self._memmap_files_to_be_deleted,
            self._memmap_accumulators_to_be_closed,
            strict=True,
        ):
            try:
                mmap._mmap.close()  # type: ignore[attr-defined]
                filepath.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete memmap file {filepath}: {e}")

    def update_batch(
        self,
        data_batch: Shaped[AccumulatorType, "B C *SpatialDims"],
        coords_batch: Shaped[AccumulatorType, "N B"],
    ) -> None:
        """Update the accumulator with a batch of tiles.

        Args:
            data_batch: Tile data. Shape ``(B, C, *SpatialDims)`` for tile-based builders
                or ``(B, C)`` when ``expand_scalars=True``.
            coords_batch: Top-left coordinates for each tile. Shape ``(N, B)`` where
                ``N`` is the number of spatial dimensions.
        """
        # Step 1: Scale coordinates from source to mask space (auto_scale)
        if self._auto_scale:
            coords_batch = (
                coords_batch * self._mask_tile_extents[:, np.newaxis]
            ) // self._source_tile_extents[:, np.newaxis]

        # Step 2: Expand per-tile scalars to uniform tile regions (expand_scalars)
        if self._expand_scalars:
            adjusted_tiles = np.zeros(
                (*data_batch.shape, *self._adjusted_tile_extents), dtype=data_batch.dtype
            )
            adjusted_tiles += data_batch[
                ..., *[np.newaxis] * len(self._adjusted_tile_extents)
            ]
            data_batch = adjusted_tiles
            coords_batch = coords_batch // self._compression_factors[:, np.newaxis]

        # Step 3: Clip tile edges (only for tile-based builders, ignored when expand_scalars)
        elif np.any(self._clip_start > 0) or np.any(self._clip_end > 0):
            extents = np.asarray(data_batch.shape[2:], dtype=np.int64)
            slices = tuple(
                slice(int(start), int(end))
                for start, end in zip(
                    self._clip_start,
                    extents - self._clip_end,
                    strict=True,
                )
            )
            data_batch = data_batch[..., *slices]  # type: ignore[index]
            coords_batch = coords_batch + self._clip_start[:, np.newaxis]

        # Step 4: Accumulate tiles into the mask
        mask_tile_extents = np.asarray(data_batch.shape[2:])
        acc_slices_all_dims = compute_acc_slices(
            coords_batch=coords_batch,
            mask_tile_extents=mask_tile_extents,
        )
        for acc_slices, data in zip(
            zip(*acc_slices_all_dims, strict=True),
            data_batch,
            strict=True,
        ):
            if self._aggregation == "mean":
                self.accumulator[:, *acc_slices] += data
                self.overlap_counter[:, *acc_slices] += 1
            else:  # max
                self.accumulator[:, *acc_slices] = np.maximum(
                    self.accumulator[:, *acc_slices],
                    data,
                )

    def finalize(self) -> tuple[AccumulatorType, ...]:
        """Finalize and return the assembled mask.

        For ``aggregation="mean"``, divides the accumulator by the overlap count
        at each position to compute the average. For ``aggregation="max"``, returns
        the accumulator unchanged.

        Returns:
            ``(accumulator, overlap_counter)`` for ``aggregation="mean"``, or
            ``(accumulator,)`` for ``aggregation="max"``.
        """
        if self._aggregation == "mean":
            self.accumulator /= self.overlap_counter.clip(min=1)
            return self.accumulator, self.overlap_counter
        return (self.accumulator,)

    def get_vips_scale_factors(self) -> tuple[float, ...]:
        """Return scaling factors to map the assembled mask back to source resolution.

        Computes per-dimension ratios between ``overflow_buffered_source_extents``
        and the accumulator spatial dimensions. These can be passed to
        ``pyvips.Image.affine()`` to rescale the assembled mask. After rescaling,
        crop to the original source extents to remove overflow padding.

        Only available when ``auto_scale=True``.

        Returns:
            Scaling factor for each spatial dimension (e.g., height, width).

        Raises:
            RuntimeError: If called when ``auto_scale=False``.
        """
        if not self._auto_scale:
            raise RuntimeError(
                "get_vips_scale_factors() is only available when auto_scale=True."
            )
        scale_factors = (
            self.overflow_buffered_source_extents / self.accumulator.shape[1:]
        )
        return tuple(float(f) for f in scale_factors)
