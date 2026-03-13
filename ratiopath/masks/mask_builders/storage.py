import logging
import tempfile
from os import PathLike
from pathlib import Path
from typing import Any, Self

import numpy as np


logger = logging.getLogger(__name__)


class inmemory[DType: np.generic](np.ndarray):
    """Storage allocator that uses in-memory NumPy arrays."""

    def __new__(cls, shape: tuple[int, ...], dtype: type[DType]) -> Self:
        return np.zeros(shape=shape, dtype=dtype).view(cls)


class memmap[DType: np.generic](np.memmap):
    """Storage allocator that uses numpy memory-mapped files (memmaps).

    This class provides disk-backed storage for large masks that exceed available RAM.
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

    This class uses NumPy's NPY format version 3.0 for compatibility with large arrays (>4GB).
    """

    def __new__(
        cls,
        shape: tuple[int, ...],
        dtype: type[DType],
        filename: str | PathLike[Any] | None = None,
    ) -> Self:
        temp_path = None
        if filename is None:
            # delete=False ensures Windows doesn't lock the file prematurely
            with tempfile.NamedTemporaryFile(delete=False) as file:
                temp_path = filename = file.name
        elif Path(filename).exists():
            raise FileExistsError(f"Memmap {filename} already exists.")

        obj = np.lib.format.open_memmap(
            filename, mode="w+", shape=shape, dtype=dtype, version=(3, 0)
        ).view(cls)

        obj._tempfile = temp_path
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        """Called automatically when a view or slice of the array is created."""
        if obj is None:
            return

        # CRITICAL: If someone slices this array (e.g., subset = my_memmap[:5]),
        # the slice should NOT own the tempfile. Only the original object should
        # delete the file during garbage collection.
        self._tempfile = None

    def cleanup(self) -> None:
        if hasattr(self, "_mmap") and self._mmap is not None:
            try:
                self._mmap.close()
            except Exception as e:
                logger.warning(f"Failed to close memmap: {e}")

        if self._tempfile is not None:
            try:
                Path(self._tempfile).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete memmap file {self._tempfile}: {e}")
            self._tempfile = None

    def __del__(self) -> None:
        self.cleanup()
