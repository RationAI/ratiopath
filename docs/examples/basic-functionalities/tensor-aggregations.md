# Aggregate Tensor Columns In Ray

!!! abstract "Overview"
    **Problem solved:** compute dataset-wide means or standard deviations over tensor-valued columns without collecting all rows back to the driver.

    **Use this example when:**

    - your Ray dataset contains embeddings, feature maps, or image tensors,
    - you need normalization statistics,
    - or you want a reduction that runs across distributed Ray blocks.

## Why This Approach

`TensorMean` and `TensorStd` are custom Ray aggregations designed for tensor-valued columns.
They preserve Ray's distributed execution model and avoid building your own manual reduction logic.

## Example

```python
import numpy as np
import ray

from ratiopath.ray.aggregate import TensorMean, TensorStd

ray.init()

ds = ray.data.from_items(
    [
        {"embedding": np.array([1.0, 2.0, 3.0])},
        {"embedding": np.array([3.0, 4.0, 5.0])},
        {"embedding": np.array([5.0, 6.0, 7.0])},
    ]
)

summary = ds.aggregate(
    TensorMean(on="embedding", axis=0, alias_name="embedding_mean"),
    TensorStd(on="embedding", axis=0, ddof=1.0, alias_name="embedding_std"),
)

print(summary["embedding_mean"])
print(summary["embedding_std"])
```

??? example "Example output"
    ```text
    [3. 4. 5.]
    [2. 2. 2.]
    ```

??? info "Under the hood"
    These aggregators treat dataset rows as the batch dimension.
    That means axis `0` is always the row/batch axis of the distributed dataset, not just a dimension inside each tensor.

    `TensorMean` accumulates partial sums.
    `TensorStd` uses a parallel variance-combination algorithm so the standard deviation remains numerically stable across multiple Ray blocks.

## Choosing The Axis

Use `axis=None` when:

- you want one scalar over every value in every row.

Use `axis=0` when:

- you want the elementwise statistic across rows.

Use `axis=1` or `axis=(0, 1)` when:

- you want to reduce both across rows and across the first tensor dimension.

```python
feature_map_stats = ds.aggregate(
    TensorMean(on="feature_map", axis=(0, 1), alias_name="channel_mean"),
    TensorStd(on="feature_map", axis=(0, 1), alias_name="channel_std"),
)
```

??? info "Important constraint"
    These are reduction operators.
    If you supply a tuple of axes, it must include `0` because the implementation is designed to reduce across the Ray batch dimension.
    If you need per-row statistics instead, use `map()` or `map_batches()` and compute them directly on each row.

## Related API

- [`ratiopath.ray.aggregate.TensorMean`](../../reference/ray/aggregations/tensor_mean.md)
- [`ratiopath.ray.aggregate.TensorStd`](../../reference/ray/aggregations/tensor_std.md)
