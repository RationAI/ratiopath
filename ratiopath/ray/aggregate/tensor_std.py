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
        >>>
        >>> # 2. Batch Std (axis=0) -> Result is a 2x2 matrix of std values
        >>> # calculated across the dataset rows.
        >>> ds.aggregate(TensorStd(on="m", axis=0))
        >>>
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

            self._aggregate_axis = tuple(axes)

    @staticmethod
    def zero_factory() -> dict:
        return {"k": 0, "mean": 0, "ssd": 0, "shape": None, "count": 0}

    def aggregate_block(self, block: Block) -> dict:
        block_acc = BlockAccessor.for_block(block)

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

        # Partial sum and element count
        block_sum = np.sum(col_np, axis=self._aggregate_axis, keepdims=True)
        block_count = col_np.size // block_sum.size

        # Compute the reference point K for stable variance calculation#
        k = block_sum / block_count

        # Shift the data by K to improve numerical stability when calculating SSD
        shifted = col_np - k
        mean = np.sum(shifted, axis=self._aggregate_axis) / block_count

        block_ssd = np.sum((shifted - mean) ** 2, axis=self._aggregate_axis)

        return {
            "k": k.flatten(),
            "mean": mean.flatten(),
            "ssd": block_ssd.flatten(),
            "shape": mean.shape,
            "count": block_count,
        }

    def combine(self, current_accumulator: dict, new: dict) -> dict:
        if new["count"] == 0:
            return current_accumulator

        if current_accumulator["count"] == 0:
            return new

        n_current = current_accumulator["count"]
        n_new = new["count"]
        combined_count = n_current + n_new

        k_current = np.asarray(current_accumulator["k"])
        k_new = np.asarray(new["k"])

        mean_current = np.asarray(current_accumulator["mean"])
        mean_new = np.asarray(new["mean"])

        # Calculate delta stably using the reference points.
        # This is algebraically identical to (mean_b - mean_a), but Sterbenz Lemma
        # ensures (K2 - K1) is computed without catastrophic precision loss.
        delta = (k_new - k_current) + mean_new - mean_current

        # Chan's formula for the combined true mean
        combined_true_mean = (k_current + mean_current) + delta * n_new / combined_count

        combined_ssd = (
            np.asarray(current_accumulator["ssd"])
            + np.asarray(new["ssd"])
            + (delta**2 * n_current * n_new / combined_count)
        )

        return {
            "k": combined_true_mean,
            "mean": np.zeros_like(combined_true_mean),
            "ssd": combined_ssd,
            "shape": new["shape"],
            "count": combined_count,
        }

    def finalize(self, accumulator: dict) -> np.ndarray | float:  # type: ignore [override]
        count = accumulator["count"]

        if count - self._ddof <= 0:
            return np.nan

        # np.maximum added as a safety net. Floating point jitter can occasionally
        # result in trivially negative numbers (e.g., -1e-16), which crashes np.sqrt
        variance = np.maximum(
            0.0,
            np.asarray(accumulator["ssd"]).reshape(accumulator["shape"])
            / (count - self._ddof),
        )

        return np.sqrt(variance)
