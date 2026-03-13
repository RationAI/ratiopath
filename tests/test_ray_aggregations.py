import numpy as np
import pytest
import ray

from ratiopath.ray.aggregate import TensorMean, TensorStd


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    """Initializes and tears down Ray for the test module."""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def sample_tensors():
    """Provides a list of dictionaries containing numpy arrays."""
    np.random.seed(42)
    return [{"m": np.random.rand(4, 5)} for _ in range(10)]


@pytest.fixture
def stacked_sample_tensors(sample_tensors):
    """Provides the exact NumPy equivalent of the stacked dataset for validation."""
    return np.stack([item["m"] for item in sample_tensors])


class TestTensorAggregatorInit:
    """Tests initialization and validation logic of the aggregators."""

    @pytest.mark.parametrize("AggClass", [TensorMean, TensorStd])
    def test_invalid_axis_raises_value_error(self, AggClass):
        with pytest.raises(ValueError, match="Axis 0 .* must be included"):
            AggClass(on="m", axis=(1, 2))

    @pytest.mark.parametrize("AggClass", [TensorMean, TensorStd])
    def test_valid_axis_initialization(self, AggClass):
        # Should not raise
        AggClass(on="m", axis=None)
        AggClass(on="m", axis=0)
        AggClass(on="m", axis=1)
        AggClass(on="m", axis=(0, 1))


class TestTensorMean:
    """End-to-end tests for TensorMean over Ray datasets."""

    @pytest.mark.parametrize(
        "axis, expected_axis",
        [
            (None, None),  # Global mean
            (0, 0),  # Batch mean
            (1, (0, 1)),  # Batch + Dim 1
            ((0, 2), (0, 2)),  # Batch + Dim 2
        ],
    )
    def test_mean_accuracy(
        self, sample_tensors, stacked_sample_tensors, axis, expected_axis
    ):
        ds = ray.data.from_items(sample_tensors).repartition(4)

        agg = TensorMean(on="m", axis=axis, alias_name="result")
        ray_result = ds.aggregate(agg)["result"]

        expected = np.mean(stacked_sample_tensors, axis=expected_axis)

        np.testing.assert_allclose(ray_result, expected, rtol=1e-6, atol=1e-8)

    def test_mean_ignore_nulls(self):
        data = [
            {"m": np.array([1.0, 2.0])},
            {"m": None},
            {"m": np.array([3.0, 4.0])},
        ]
        ds = ray.data.from_items(data)

        agg_ignore = TensorMean(on="m", axis=0, ignore_nulls=True, alias_name="res")

        res_ignore = ds.aggregate(agg_ignore)["res"]
        np.testing.assert_allclose(res_ignore, np.array([2.0, 3.0]))

        agg_strict = TensorMean(on="m", axis=0, ignore_nulls=False, alias_name="res")
        with pytest.raises(Exception, match="contains null values"):
            ds.aggregate(agg_strict)


class TestTensorStd:
    """End-to-end tests for TensorStd over Ray datasets."""

    @pytest.mark.parametrize(
        "axis, expected_axis",
        [
            (None, None),
            (0, 0),
            (1, (0, 1)),
            ((0, 2), (0, 2)),
        ],
    )
    @pytest.mark.parametrize("ddof", [0.0, 1.0])
    def test_std_accuracy(
        self, sample_tensors, stacked_sample_tensors, axis, expected_axis, ddof
    ):
        ds = ray.data.from_items(sample_tensors).repartition(4)

        agg = TensorStd(on="m", axis=axis, ddof=ddof, alias_name="result")

        ray_result = ds.aggregate(agg)["result"]

        expected = np.std(stacked_sample_tensors, axis=expected_axis, ddof=ddof)
        np.testing.assert_allclose(ray_result, expected, rtol=1e-6, atol=1e-8)

    def test_std_all_null_or_empty(self):
        ds_empty = ray.data.from_items([{"m": np.array([1, 2])}]).filter(
            lambda x: False
        )

        res_empty = ds_empty.aggregate(TensorStd(on="m", axis=0, alias_name="res"))[
            "res"
        ]
        assert res_empty is None or np.isnan(res_empty)

        ds_single = ray.data.from_items([{"m": np.array([1.0, 2.0])}])

        res_single = ds_single.aggregate(
            TensorStd(on="m", axis=0, ddof=1, alias_name="res")
        )["res"]
        assert np.isnan(res_single).all()

    def test_std_numerical_stability(self):
        """Tests Chan's algorithm with large numbers where naive variance fails."""
        base = 1e9
        data = [
            {"m": np.array([base + 1, base + 2])},
            {"m": np.array([base + 1, base + 2])},
            {"m": np.array([base + 3, base + 4])},
            {"m": np.array([base + 3, base + 4])},
        ]
        ds = ray.data.from_items(data).repartition(4)

        agg = TensorStd(on="m", axis=0, ddof=1.0, alias_name="res")
        ray_result = ds.aggregate(agg)["res"]

        stacked = np.stack([d["m"] for d in data])
        expected = np.std(stacked, axis=0, ddof=1.0)

        np.testing.assert_allclose(ray_result, expected, rtol=1e-6, atol=1e-8)
