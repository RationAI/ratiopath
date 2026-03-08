from typing import cast

import numpy as np

from ray.data.aggregate import AggregateFnV2
from ray.data.block import Block, BlockAccessor


class TensorStd(AggregateFnV2[dict, np.ndarray | float]):
    """Calculates the standard deviation of a column containing Tensors.

    This aggregator treats the data column as a high-dimensional array where
    **axis 0 represents the batch dimension**. To satisfy the requirements
    of a reduction and prevent memory growth proportional to the number of rows,
    axis 0 must be included in the aggregation.

    It uses a parallel variance accumulation algorithm (Chan's method) to maintain
    numerical stability while processing data across multiple Ray blocks.

    Args:
        on: The name of the column containing tensors or numbers.
        axis: The axis or axes along which the reduction is computed.
            - `None`: Global reduction. Collapses all dimensions (including batch)
              to a single scalar.
            - `int`: Aggregates over both the batch (axis 0) AND the specified
              tensor dimension. For example, `axis=1` collapses the batch and
              the first dimension of the tensors.
            - `tuple`: A sequence of axes that **must** explicitly include `0`.
        ddof: Delta Degrees of Freedom. The divisor used in calculations
            is $N - ddof$, where $N$ represents the number of elements.
            Defaults to 1.0 (sample standard deviation).
        ignore_nulls: Whether to ignore null values. Defaults to True.
        alias_name: Optional name for the resulting column. Defaults to "std(<on>)".

    Raises:
        ValueError: If `axis` is provided as a tuple but does not include `0`.

    Note:
        This aggregator is designed for "reduction" operations. If you wish to
        calculate statistics per-row without collapsing the batch dimension,
        use `.map()` instead.

    Example:
        >>> import ray
        >>> import numpy as np
        >>> from ratiopath.ray.aggregate import TensorStd
        >>> ds = ray.data.from_items(
        ...     [
        ...         {"m": np.array([[1, 2], [1, 2]])},
        ...         {"m": np.array([[5, 6], [5, 6]])},
        ...     ]
        ... )
        >>> # 1. Global Std (axis=None) -> All elements reduced to one scalar
        >>> ds.aggregate(TensorStd(on="m", axis=None))
        >>> # 2. Batch Std (axis=0) -> Result is a 2x2 matrix of std values
        >>> # calculated across the dataset rows.
        >>> ds.aggregate(TensorStd(on="m", axis=0))
        >>> # 3. Int shorthand (axis=1) -> Internally uses axis=(0, 1)
        >>> # Collapses batch and the first dimension of the tensor.
        >>> ds.aggregate(TensorStd(on="m", axis=1))
    """

    _aggregate_axis: tuple[int, ...] | None = None

    def __init__(
        self,
        on: str,
        axis: int | tuple[int, ...] | None = None,
        ddof: float = 1.0,
        ignore_nulls: bool = True,
        alias_name: str | None = None,
    ):
        super().__init__(
            name=alias_name if alias_name else f"std({on})",
            on=on,
            ignore_nulls=ignore_nulls,
            # sum: partial sum, ssd: sum of squared differences, count: elements reduced
            zero_factory=self.zero_factory,
        )

        self._ddof = ddof

        if axis is not None:
            axes = {0, axis} if isinstance(axis, int) else set(axis)

            if 0 not in axes:
                raise ValueError(
                    f"Invalid axis configuration: {axis}. Axis 0 (the batch dimension) "
                    "must be included to perform a reduction. To process rows "
                    "independently without collapsing the batch, use .map() instead."
                )

            self._aggregate_axis = tuple(sorted(axes))

    @staticmethod
    def zero_factory() -> dict:
        return {"sum": None, "ssd": None, "shape": None, "count": 0}

    def aggregate_block(self, block: Block) -> dict:
        block_acc = BlockAccessor.for_block(block)

        # If there are no valid (non-null) entries, return the zero value
        if block_acc.count(self._target_col_name, self._ignore_nulls) == 0:  # type: ignore [arg-type]
            return self.zero_factory()

        col_np = cast("np.ndarray", block_acc.to_numpy(self._target_col_name))

        # Partial sum and element count
        block_sum = np.sum(col_np, axis=self._aggregate_axis)
        block_count = np.prod(col_np.shape) // np.prod(block_sum.shape)

        # SSD calculation: sum((x - mean)^2)
        if self._aggregate_axis is None:
            block_ssd = np.sum((col_np - (block_sum / block_count)) ** 2)
        else:
            # Re-expand sum for broadcasting
            expanded_sum = block_sum
            for ax in self._aggregate_axis:
                expanded_sum = np.expand_dims(expanded_sum, ax)
            block_ssd = np.sum(
                (col_np - (expanded_sum / block_count)) ** 2, axis=self._aggregate_axis
            )

        return {
            "sum": block_sum.flatten(),
            "ssd": block_ssd.flatten(),
            "shape": block_sum.shape,
            "count": block_count,
        }

    def combine(self, current_accumulator: dict, new: dict) -> dict:
        if new["count"] == 0:
            return current_accumulator

        if current_accumulator["count"] == 0:
            return new

        mean_a = np.asarray(current_accumulator["sum"]) / current_accumulator["count"]
        mean_b = np.asarray(new["sum"]) / new["count"]
        delta = mean_b - mean_a

        combined_sum = np.asarray(current_accumulator["sum"]) + np.asarray(new["sum"])
        combined_count = current_accumulator["count"] + new["count"]
        combined_ssd = (
            np.asarray(current_accumulator["ssd"])
            + np.asarray(new["ssd"])
            + (delta**2 * current_accumulator["count"] * new["count"] / combined_count)
        )

        return {
            "sum": combined_sum,
            "ssd": combined_ssd,
            "shape": new["shape"],
            "count": combined_count,
        }

    def finalize(self, accumulator: dict) -> np.ndarray | float:  # type: ignore [override]
        count = accumulator["count"]

        if count - self._ddof <= 0:
            return np.nan

        return np.sqrt(
            np.asarray(accumulator["ssd"]).reshape(accumulator["shape"])
            / (count - self._ddof)
        )
