from typing import cast

import numpy as np

from ray.data.aggregate import AggregateFnV2
from ray.data.block import Block, BlockAccessor


class TensorMean(AggregateFnV2[dict, np.ndarray | float]):
    """Calculates the mean (average) of a column containing Tensors.

    This aggregator treats the data column as a high-dimensional array where
    **axis 0 represents the batch dimension**. To satisfy the requirements
    of a reduction and prevent memory growth proportional to the number of rows,
    axis 0 must be included in the aggregation.


    Args:
        on: The name of the column containing tensors or numbers.
        axis: The axis or axes along which the reduction is computed.
            - `None`: Global reduction. Collapses all dimensions (including batch)
              to a single scalar.
            - `int`: Aggregates over both the batch (axis 0) AND the specified
              tensor dimension. For example, `axis=1` collapses the batch and
              the first dimension of the tensors.
            - `tuple`: A sequence of axes that **must** explicitly include `0`.
        ignore_nulls: Whether to ignore null values. Defaults to True.
        alias_name: Optional name for the resulting column. Defaults to "mean(<on>)".

    Raises:
        ValueError: If `axis` is provided as a tuple but does not include `0`.

    Note:
        This aggregator is designed for "reduction" operations. If you wish to
        calculate statistics per-row without collapsing the batch dimension,
        use `.map()` instead.

    Example:
        >>> import ray
        >>> import numpy as np
        >>> from ratiopath.ray.aggregate import TensorMean
        >>> # Dataset with 2x2 matrices: total shape (Batch=2, Dim1=2, Dim2=2)
        >>> ds = ray.data.from_items(
        ...     [
        ...         {"m": np.array([[1, 1], [1, 1]])},
        ...         {"m": np.array([[3, 3], [3, 3]])},
        ...     ]
        ... )
        >>> # 1. Global Mean (axis=None) -> Result: 2.0
        >>> ds.aggregate(TensorMean(on="m", axis=None))
        >>>
        >>> # 2. Batch Mean (axis=0) -> Result: np.array([[2, 2], [2, 2]])
        >>> ds.aggregate(TensorMean(on="m", axis=0))
        >>>
        >>> # 3. Mean across Batch and Rows (axis=(0, 1)) -> Result: np.array([2, 2])
        >>> ds.aggregate(TensorMean(on="m", axis=(0, 1)))
    """

    _aggregate_axis: tuple[int, ...] | None = None

    def __init__(
        self,
        on: str,
        axis: int | tuple[int, ...] | None = None,
        ignore_nulls: bool = True,
        alias_name: str | None = None,
    ):
        super().__init__(
            name=alias_name if alias_name else f"mean({on})",
            on=on,
            ignore_nulls=ignore_nulls,
            # Initialize with identity values for summation
            zero_factory=self.zero_factory,
        )

        if axis is not None:
            axes = {0, axis} if isinstance(axis, int) else set(axis)

            if 0 not in axes:
                raise ValueError(
                    f"Invalid axis configuration: {axis}. Axis 0 (the batch dimension) "
                    "must be included to perform a reduction. To process rows "
                    "independently without collapsing the batch, use .map() instead."
                )

            self._aggregate_axis = tuple(axes)

    @staticmethod
    def zero_factory() -> dict:
        return {"sum": 0, "shape": None, "count": 0}

    def aggregate_block(self, block: Block) -> dict:
        block_acc = BlockAccessor.for_block(block)

        # If there are no valid (non-null) entries, return the zero value
        if block_acc.count(self._target_col_name, self._ignore_nulls) == 0:  # type: ignore [arg-type]
            return self.zero_factory()

        col_np = cast("np.ndarray", block_acc.to_numpy(self._target_col_name))

        # Handle object dtype (triggered by nulls or ragged tensor shapes)
        if col_np.dtype == object:
            valid_tensors = [x for x in col_np if x is not None]

            # If lengths differ, we dropped at least one None.
            if len(valid_tensors) != col_np.size and not self._ignore_nulls:
                raise ValueError(
                    f"Column '{self._target_col_name}' contains null values, but "
                    "ignore_nulls is False."
                )

            # Handle the all-null block case
            if not valid_tensors:
                return self.zero_factory()

            # Reconstruct the contiguous numeric tensor
            col_np = np.stack(valid_tensors)

        # Perform the partial sum and calculate how many elements contributed
        block_sum = np.sum(col_np, axis=self._aggregate_axis)
        block_count = col_np.size // block_sum.size

        return {
            "sum": block_sum.flatten(),
            "shape": block_sum.shape,
            "count": block_count,
        }

    def combine(self, current_accumulator: dict, new: dict) -> dict:
        return {
            "sum": np.asarray(current_accumulator["sum"]) + np.asarray(new["sum"]),
            "shape": current_accumulator["shape"] or new["shape"],
            "count": current_accumulator["count"] + new["count"],
        }

    def finalize(self, accumulator: dict) -> np.ndarray | float:  # type: ignore [override]
        count = accumulator["count"]

        if count == 0:
            return np.nan

        # Reshape the flattened sum back to original aggregated dimensions
        return np.asarray(accumulator["sum"]).reshape(accumulator["shape"]) / count
