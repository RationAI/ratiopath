from ratiopath.masks.mask_builders.mask_builder import AccumulatorType, MaskBuilder


import tempfile
from pathlib import Path
from typing import Any


class NumpyMemMapMaskBuilderAllocatorMixin(MaskBuilder):
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

    def allocate_accumulator(
        self,
        mask_extents: Int64[AccumulatorType, " N"],
        channels: int,
        dtype: npt.DTypeLike = np.float32,
        filepath: Path | None = None,
        **kwargs: Any,
    ) -> np.memmap:
        if filepath is None:
            with tempfile.NamedTemporaryFile() as temp_file:
                return np.lib.format.open_memmap(
                    temp_file.name,
                    mode="w+",
                    shape=(channels, *mask_extents),
                    dtype=dtype,
                    version=(3, 0),
                )
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


class NumpyArrayMaskBuilderAllocatorMixin(MaskBuilder):
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