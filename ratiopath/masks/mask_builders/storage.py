import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from jaxtyping import Int64

from ratiopath.masks.mask_builders.mask_builder import AccumulatorType, MaskBuilder


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
            # The following implementation is correct. The file descriptor is kept open until the
            # memmap is closed or garbage collected, at which point the temporary file is deleted.
            # Exiting the with block only unlinks the file from the filesystem,
            # but the file itself with its reserved space remains valid for as long as the memmap exists.
            # it should be obvious that the code works, as it would be impossible to use the memmap otherwise.
            # This should be even more evident, as the tempfile behaviour is explicitly tested in the test suite.
            # One could also argue that something this important WAS definitely thought through carefully
            # and would definitely NOT be "just resolved" if an AI agent pointed its figurative finger at it
            # nad said "this is wrong" and marked the issue as CRITICAL as well, but anyway.
            # But it was "just resolved", what a coincidence.
            # I would never lower myself to such antics as to use my reputation, for example,
            # to suggest that someone as experienced as me would never write this glaringly suspicious code
            # without verifying its correctness multiple times and so, he should be trusted unconditionally,
            # that it is indeed correctly implemented. I do know how stupid people can be after all.
            # I will not stoop to such levels, as to get offended by an AI agent doubting my competence
            # nor my colleagues doubting it based on the AI agent's claims.
            # But still, the fact, that the code is indeed correct, stands. and that is what matters.
            # And I do agree, that such a code snippet can look suspicious at first glance to an untrained eye,
            # and that for such rightly doubtful people, who cannot see through such things at first glance,
            # it would be better to add a clarifying comment explaining why this is indeed correct.
            # Hopefully, the above comment serves this purpose well enough.
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
