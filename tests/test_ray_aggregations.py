import os

import numpy as np
import pytest
import ray


# Set environment variables before ray.init
os.environ["RAY_ENABLE_METRICS_EXPORT"] = "0"
os.environ["RAY_IGNORE_VENV_MISMATCH"] = "1"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

# Adjust imports based on your file structure
from ratiopath.ray.aggregate import TensorMean, TensorStd


@pytest.fixture(scope="module")
def ray_start():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


## --- TensorMean Tests ---


def test_tensor_mean_global(ray_start):
    """Tests axis=None: Global reduction to a single scalar."""
    data = [
        {"m": np.array([[2, 4], [6, 8]])},
        {"m": np.array([[0, 0], [0, 0]])},
        {"m": None},
    ]
    ds = ray.data.from_items(data)
    result = ds.aggregate(TensorMean(on="m", axis=None))
    # (2+4+6+8) / 8 = 2.5
    assert result["mean(m)"] == 2.5


def test_tensor_mean_int_shorthand(ray_start):
    """Tests axis=1: Should aggregate over batch (0) AND dim 1."""
    data = [
        {"m": np.array([[10, 20], [30, 40]])},  # Row sums: 30, 70
        {"m": np.array([[0, 0], [0, 0]])},  # Row sums: 0, 0
    ]
    ds = ray.data.from_items(data).repartition(
        2
    )  # Ensure multiple blocks for reduction
    # Aggregating over axis 1 (internal becomes (0, 1))
    result = ds.aggregate(TensorMean(on="m", axis=1))

    expected = np.array([10.0, 15.0])  # [(10+30+0+0)/4, (20+40+0+0)/4]
    np.testing.assert_array_equal(result["mean(m)"], expected)


def test_tensor_mean_batch_only(ray_start):
    """Tests axis=0: Should collapse only the batch dimension."""
    data = [
        {"m": np.array([[10, 10], [10, 10]])},
        {"m": np.array([[20, 20], [20, 20]])},
    ]
    ds = ray.data.from_items(data).repartition(
        2
    )  # Ensure multiple blocks for reduction
    result = ds.aggregate(TensorMean(on="m", axis=0))

    expected = np.array([[15.0, 15.0], [15.0, 15.0]])
    np.testing.assert_array_equal(result["mean(m)"], expected)


## --- TensorStd Tests ---


def test_tensor_std_global(ray_start):
    """Tests global standard deviation."""
    vals = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    data = [{"m": vals[:4].reshape(2, 2)}, {"m": vals[4:].reshape(2, 2)}]
    ds = ray.data.from_items(data).repartition(
        2
    )  # Ensure multiple blocks for reduction

    result = ds.aggregate(TensorStd(on="m", axis=None, ddof=0))
    expected = np.std(vals)
    assert pytest.approx(result["std(m)"], 0.0001) == expected


def test_tensor_std_batch_only(ray_start):
    """Tests STD across the batch dimension only."""
    # Two identical matrices with different offsets
    data = [
        {"m": np.array([10, 20])},  # Sample 1
        {"m": np.array([30, 40])},  # Sample 2
    ]
    ds = ray.data.from_items(data).repartition(
        2
    )  # Ensure multiple blocks for reduction
    result = ds.aggregate(TensorStd(on="m", axis=0, ddof=0))

    # Std of [10, 30] is 10; Std of [20, 40] is 10
    expected = np.array([10.0, 10.0])
    np.testing.assert_array_equal(result["std(m)"], expected)


## --- Validation & Logic Tests ---


def test_invalid_axis_tuple(ray_start):
    """Verifies that providing a tuple without axis 0 raises ValueError."""
    with pytest.raises(
        ValueError, match=r"Axis 0 \(the batch dimension\) must be included"
    ):
        TensorMean(on="m", axis=(1, 2))


def test_tensor_aggregate_groupby(ray_start):
    """Verifies Mean and Std work within groupby operations."""
    data = [
        {"id": "A", "m": np.array([1, 1])},
        {"id": "A", "m": np.array([3, 3])},
        {"id": "B", "m": np.array([10, 10])},
    ]
    ds = ray.data.from_items(data)

    # Test Mean Groupby
    res_mean = ds.groupby("id").aggregate(TensorMean(on="m", axis=0)).take_all()
    res_mean = sorted(res_mean, key=lambda x: x["id"])

    np.testing.assert_array_equal(res_mean[0]["mean(m)"], [2.0, 2.0])  # Mean of [1,3]
    np.testing.assert_array_equal(res_mean[1]["mean(m)"], [10.0, 10.0])

    # Test Std Groupby
    res_std = ds.groupby("id").aggregate(TensorStd(on="m", axis=0, ddof=0)).take_all()
    res_std = sorted(res_std, key=lambda x: x["id"])

    np.testing.assert_array_equal(res_std[0]["std(m)"], [1.0, 1.0])  # Std of [1,3]
    np.testing.assert_array_equal(res_std[1]["std(m)"], [0.0, 0.0])  # Std of [10,10]
