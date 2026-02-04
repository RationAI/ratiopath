import contextlib
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from jaxtyping import Int64

from ratiopath.masks.mask_builders.mask_builder import AccumulatorType, MaskBuilderABC


logger = logging.getLogger(__name__)


class NumpyMemMapMaskBuilderAllocatorMixin(MaskBuilderABC):
    """Mixin class to allocate accumulators as numpy memory-mapped files (memmaps).

    This mixin provides disk-backed storage for large masks that exceed available RAM.
    Memory mapping allows the OS to manage paging between disk and memory transparently,
    enabling processing of masks that would otherwise cause out-of-memory errors.

    **Temporary Files (default behavior when `filepath=None`):**
    A temporary file is created and used as backing storage. The file is deleted when
    the memmap is closed or garbage collected. Disk space is consumed during processing
    but automatically reclaimed afterward.

    **Explicit Files (when `filepath` is provided):**
    The memmap is backed by the specified file path, which persists after processing.
    This is useful for caching results or processing masks too large for temporary storage.
    If the file already exists, a FileExistsError is raised to prevent accidental data loss.

    This mixin uses NumPy's NPY format version 3.0 for compatibility with large arrays (>4GB).
    """

    _memmap_files_to_be_deleted: list[Path]
    _memmap_accumulators_to_be_closed: list[np.memmap]

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._memmap_files_to_be_deleted = []
        self._memmap_accumulators_to_be_closed = []
        super().__init__(*args, **kwargs)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._cleanup_memmaps()

    def _cleanup_memmaps(self) -> None:
        """Ensure that any temporary memmap files are deleted when the builder is garbage collected."""
        for filepath, mmap in zip(
            self._memmap_files_to_be_deleted,
            self._memmap_accumulators_to_be_closed,
            strict=True,
        ):
            try:
                # Close the memmap to release file handles
                mmap._mmap.close()  # type: ignore[attr-defined]
                # del mmap  # Ensure the memmap object is deleted
                filepath.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete memmap file {filepath}: {e}")

    def allocate_accumulator(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: npt.DTypeLike = np.float32,
        filepath: Path | None = None,
        **kwargs: Any,
    ) -> np.memmap:
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
        else:
            if filepath.exists():
                raise FileExistsError(f"Memmap filepath {filepath} already exists.")
            return np.lib.format.open_memmap(
                filepath,
                mode="w+",
                shape=(channels, *mask_extents),
                dtype=dtype,
                version=(3, 0),
            )


class NumpyArrayMaskBuilderAllocatorMixin(MaskBuilderABC):
    """Mixin class to allocate accumulators as numpy arrays.

    This mixin implements the `allocate_accumulator` method to create
    numpy arrays for the accumulator.
    """

    def allocate_accumulator(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: npt.DTypeLike = np.float32,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.zeros((channels, *mask_extents), dtype=dtype)
