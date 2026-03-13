import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class StorageAllocator[DType: np.generic](ABC):
    """Abstract base class for storage allocation strategies.

    The allocator is responsible for creating and managing the lifecycle of the
    main accumulator array.
    """

    accumulator: NDArray[DType]

    @abstractmethod
    def __init__(
        self, shape: tuple[int, ...], dtype: type[DType], **kwargs: Any
    ) -> None:
        """Initialize and allocate the accumulator.

        Args:
            shape: The shape of the accumulator array.
            dtype: Data type of the array.
            **kwargs: Additional allocation arguments (e.g. filepath).
        """

    def cleanup(self) -> None:
        """Perform any necessary cleanup (e.g. closing files, deleting temporary memmaps)."""
        return


class MemMapStorage[DType: np.generic](StorageAllocator[DType]):
    """Storage allocator that uses numpy memory-mapped files (memmaps).

    This storage provides disk-backed allocation for large masks that exceed available RAM.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: type[DType],
        filename: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        if filename is None:
            with tempfile.NamedTemporaryFile(delete=False) as file:
                self.tempfile = file.name
                filename = file.name
        elif Path(filename).exists():
            raise FileExistsError(f"Memmap {filename} already exists.")

        self.accumulator = np.memmap(filename, mode="w+", shape=shape, dtype=dtype)

    def __del__(self) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        if hasattr(self, "accumulator"):
            try:
                self.accumulator._mmap.close()  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"Failed to close memmap: {e}")
            finally:
                del self.accumulator

        if hasattr(self, "tempfile"):
            try:
                Path(self.tempfile).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete memmap file {self.tempfile}: {e}")
            finally:
                del self.tempfile


class NumpyStorage[DType: np.generic](StorageAllocator[DType]):
    """Storage allocator that uses in-memory numpy arrays."""

    def __init__(
        self, shape: tuple[int, ...], dtype: type[DType], **kwargs: Any
    ) -> None:
        self.accumulator = np.zeros(shape, dtype=dtype)
