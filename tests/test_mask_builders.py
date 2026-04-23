import tempfile
from pathlib import Path

import numpy as np
import pytest

from ratiopath.masks.mask_builders import MaskBuilder, MaxAggregator, MeanAggregator


@pytest.mark.parametrize("mask_extents", [(16, 16), (32, 64)])
@pytest.mark.parametrize("channels", [1, 3, 8])
@pytest.mark.parametrize("mask_tile_extents", [(4, 4), (4, 8), (8, 4), (3, 6), (9, 6)])
@pytest.mark.parametrize("mask_tile_strides", [(2, 2), (3, 3), (3, 2), (2, 3)])
def test_scalar_uniform_averaging_2d(
    mask_extents, channels, mask_tile_extents, mask_tile_strides
):
    mask_extents = np.asarray(mask_extents)
    mask_tile_extents = np.asarray(mask_tile_extents)
    mask_tile_strides = np.asarray(mask_tile_strides)

    batch_size = 4
    num_batches = 8

    gcds = np.gcd(mask_tile_extents, mask_tile_strides)
    adjusted_tile_extents = mask_tile_extents // gcds

    test_mask_builder = MaskBuilder(
        source_extents=tuple(mask_extents),
        source_tile_extent=tuple(mask_tile_extents),
        output_tile_extent=(1, 1),
        stride=tuple(mask_tile_strides),
        n_channels=channels,
        storage="inmemory",
        aggregation=MeanAggregator,
        dtype=np.float32,
    )

    # create scalar batch (B, C)
    example = np.ones((batch_size, channels), dtype=np.float32)

    for i in range(num_batches):
        coords_batch = np.random.rand(batch_size, len(mask_extents))
        coords_batch *= (np.array(mask_extents) - np.array(mask_tile_extents))[
            np.newaxis, :
        ]
        # force alignment to gcds
        coords_batch = (coords_batch.astype(np.int64) // gcds[np.newaxis, :]) * gcds[
            np.newaxis, :
        ]

        test_mask_builder.update_batch(example, coords_batch)

        assert np.isclose(
            test_mask_builder.storage.sum(),
            (i + 1) * batch_size * channels * np.prod(adjusted_tile_extents),
            atol=1e-6,
        ), (
            f"Accumulator sum after batch {i + 1} should be {(i + 1) * batch_size * channels * np.prod(adjusted_tile_extents)}, is {test_mask_builder.storage.sum()}"
        )

    result = test_mask_builder.finalize()
    # assembled has shape (channels, H_adj, W_adj)
    assert result["mask"].shape[0] == channels, (
        "Assembled mask should have correct number of channels"
    )
    assert result["overlap_counter"].shape[0] == 1, (
        "Overlap accumulator should have single channel"
    )
    assert np.isclose(
        result["overlap_counter"].sum(),
        num_batches * batch_size * np.prod(adjusted_tile_extents),
        atol=1e-6,
    ), (
        f"Overlap sum should be {num_batches * batch_size * np.prod(adjusted_tile_extents)}, is {result['overlap_counter'].sum()}"
    )
    assert np.isclose(result["mask"].max(), 1.0, atol=1e-6), (
        f"Assembled max should be close to 1.0, is {result['mask'].max()}"
    )
    test_mask_builder.cleanup()


@pytest.mark.parametrize("shape", [(1, 16, 16), (3, 32, 64), (8, 400, 35)])
@pytest.mark.parametrize("mask_tile_extents", [(4, 4), (4, 8), (8, 4), (3, 6), (9, 6)])
@pytest.mark.parametrize("mask_tile_strides", [(2, 2), (3, 3), (3, 2), (2, 3)])
def test_scalar_uniform_max_2d(shape, mask_tile_extents, mask_tile_strides):
    channels = shape[0]
    source_extents = np.asarray(shape[1:])
    source_tile_extent = np.asarray(mask_tile_extents)
    stride = np.asarray(mask_tile_strides)
    output_tile_extent = 1

    batch_size = 4
    num_batches = 10

    test_mask_builder = MaskBuilder(
        source_extents=tuple(source_extents),
        source_tile_extent=tuple(source_tile_extent),
        output_tile_extent=output_tile_extent,
        stride=tuple(stride),
        n_channels=channels,
        storage="inmemory",
        aggregation=MaxAggregator,
        dtype=np.float32,
    )

    for i in range(num_batches):
        coordinates_batch = np.random.rand(batch_size, len(source_extents))
        coordinates_batch *= (source_extents - source_tile_extent)[np.newaxis, :]
        # align to strides
        coordinates_batch = (
            coordinates_batch.astype(np.int64) // stride[np.newaxis, :]
        ) * stride[np.newaxis, :]

        value = float(i + 1)
        data = np.full((batch_size, channels), value, dtype=np.float32)
        test_mask_builder.update_batch(data, coordinates_batch)

        accumulator = test_mask_builder.storage
        assert accumulator.max() == value, (
            f"Max value in accumulator after batch {i + 1} should be {value}, is {accumulator.max()}"
        )

    accumulator = test_mask_builder.storage
    nonfinal_acc = accumulator.copy()
    final_acc = test_mask_builder.finalize()
    # final_acc should contain the maximum value (num_batches)
    assert (final_acc == nonfinal_acc).all(), (
        "Finalized accumulator should be equal to non-finalized accumulator for max aggregation"
    )
    test_mask_builder.cleanup()


@pytest.mark.parametrize("clip", [0, 1, 3])
@pytest.mark.parametrize("tile_extents", [(8, 8), (12, 9)])
@pytest.mark.parametrize("channels", [1, 5])
@pytest.mark.parametrize("mask_extents", [(32, 64), (64, 32), (100, 101)])
@pytest.mark.parametrize("filename", [None, "test_heatmap.npy"])
@pytest.mark.parametrize("overlap_counter_filename", [None, "test_overlap.counter.npy"])
def test_edge_clipping_heatmap_assembler(
    clip,
    tile_extents,
    channels,
    mask_extents,
    filename,
    overlap_counter_filename,
    tmp_path,
):
    source_extents = np.asarray(mask_extents)
    source_tile_extent = np.asarray(tile_extents)
    stride = np.asarray((4, 4), dtype=np.int64)
    # Dense output case
    output_tile_extent = source_tile_extent

    num_batches = 4
    batch_size = 8

    if filename is not None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        filename = Path((tmp_path / filename).as_posix())
    if overlap_counter_filename is not None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        overlap_counter_filename = Path(
            (tmp_path / overlap_counter_filename).as_posix()
        )

    assembler = MaskBuilder(
        source_extents=tuple(source_extents),
        source_tile_extent=tuple(source_tile_extent),
        output_tile_extent=tuple(output_tile_extent),
        stride=tuple(stride),
        n_channels=channels,
        storage="memmap",
        aggregation=MeanAggregator,
        dtype=np.float32,
        storage_kwargs={"filename": filename},
        aggregation_kwargs={"overlap_counter_filename": overlap_counter_filename},
    )

    # create dummy data
    example_tile_batch = np.ones(
        (batch_size, channels, *output_tile_extent), dtype=np.float32
    )

    # clipped sum per tile in mask resolution
    # Note: clip is in output (mask) resolution
    clipped_tile_extent = output_tile_extent - 2 * clip
    increment = batch_size * channels * np.prod(clipped_tile_extent)

    # add the tiles randomly to cover the heatmap
    for i in range(num_batches):
        coords = np.random.rand(batch_size, len(source_extents))
        coords *= (source_extents - source_tile_extent)[np.newaxis, :]
        coords = coords.astype(np.int64)
        assembler.update_batch(example_tile_batch, coords, edge_clipping=clip)

        accumulator = assembler.storage
        assert np.isclose(accumulator.sum(), (i + 1) * increment, atol=1e-5), (
            f"Checksum mismatch in heatmap accumulator after update {i + 1}. Is {accumulator.sum()}, but expected {(i + 1) * increment}"
        )

    results = assembler.finalize()
    assembled_heatmap = results["mask"]
    overlap_counter = results["overlap_counter"]

    assert assembled_heatmap.shape == (
        channels,
        *assembler.mask_extents,
    ), (
        f"Assembled heatmap has incorrect shape: {assembled_heatmap.shape}, shall be {(channels, *assembler.mask_extents)}"
    )
    assert overlap_counter.shape == (1, *assembler.mask_extents), (
        f"Overlap counter has incorrect shape: {overlap_counter.shape}, shall be {(1, *assembler.mask_extents)}"
    )
    assert np.isclose(
        overlap_counter.sum(),
        num_batches * batch_size * np.prod(clipped_tile_extent),
        atol=1e-5,
    ), (
        f"Checksum mismatch in overlap counter after finalization. Is {overlap_counter.sum()}, but expected {num_batches * batch_size * np.prod(clipped_tile_extent)}"
    )
    assert assembled_heatmap.max() == 1.0, (
        f"Assembled heatmap values should be normalized to 1.0 after finalization: {assembled_heatmap.max()}"
    )
    assembler.cleanup()


def test_edge_clipping_clips_edges():
    """Simple test that ensures, that a tile put at [0,0], if clipped, does not write to the [0,0] corner of the heatmap."""
    clip = 1
    channels = 1
    source_extents = (16, 16)
    source_tile_extent = (8, 8)
    output_tile_extent = (8, 8)
    stride = (4, 4)

    assembler = MaskBuilder(
        source_extents=source_extents,
        source_tile_extent=source_tile_extent,
        output_tile_extent=output_tile_extent,
        stride=stride,
        n_channels=channels,
        storage="memmap",
        aggregation=MeanAggregator,
        dtype=np.float32,
    )

    tile = np.ones((1, channels, *output_tile_extent), dtype=np.float32)
    assembler.update_batch(tile, coords=np.asarray([[0, 0]]), edge_clipping=clip)
    results = assembler.finalize()
    assembled_heatmap = results["mask"]
    overlap_counter = results["overlap_counter"]

    assert (assembled_heatmap[0, 0] == 0.0).all(), (
        "Top-left corner of assembled heatmap should be zero due to clipping"
    )
    assert (overlap_counter[0, 0] == 0.0).all(), (
        "Top-left corner of overlap counter should be zero due to clipping"
    )
    assembler.cleanup()


@pytest.mark.parametrize(
    "source_extents, source_tile_extent, output_tile_extent, stride, expected_mask_extents",
    [
        # Divisible extents: (source_extents - source_tile_extent) is divisible by stride.
        ((33, 53), (8, 8), (4, 4), (4, 4), (18, 28)),
        ((33, 53), (8, 8), (1, 1), (4, 4), (9, 14)),
    ],
)
def test_storage_shape_is_minimal_for_extent_and_stride(
    source_extents,
    source_tile_extent,
    output_tile_extent,
    stride,
    expected_mask_extents,
):
    """Storage shape must be the smallest shape that can fit all tiled outputs."""
    builder = MaskBuilder(
        source_extents=source_extents,
        source_tile_extent=source_tile_extent,
        output_tile_extent=output_tile_extent,
        stride=stride,
        n_channels=2,
        storage="inmemory",
        aggregation=MeanAggregator,
        dtype=np.float32,
    )

    expected_mask_extents_arr = np.asarray(expected_mask_extents, dtype=np.int64)

    assert np.array_equal(builder.mask_extents, expected_mask_extents_arr), (
        f"Mask extents {builder.mask_extents} should be minimal {expected_mask_extents_arr}"
    )
    assert builder.storage.shape == (2, *expected_mask_extents), (
        f"Storage shape {builder.storage.shape} should be {(2, *expected_mask_extents)}"
    )

    builder.cleanup()


def test_numpy_memmap_tempfile_management(monkeypatch):
    """Test that temporary files created by NamedTemporaryFile are properly deleted."""
    captured_files = []
    original_namedtempfile = tempfile.NamedTemporaryFile

    def intercepting_namedtempfile(*args, **kwargs):
        temp_file = original_namedtempfile(*args, **kwargs)
        captured_files.append(Path(temp_file.name))
        return temp_file

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", intercepting_namedtempfile)

    assembler = MaskBuilder(
        source_extents=(16, 16),
        source_tile_extent=(8, 8),
        output_tile_extent=(8, 8),
        stride=(4, 4),
        n_channels=1,
        storage="memmap",
        aggregation=MeanAggregator,
        dtype=np.float32,
    )

    assert len(captured_files) >= 2, (
        "Expected at least two temporary files for main and overlap accumulators"
    )
    temp_filepaths = captured_files.copy()
    tile = np.ones((1, 1, 8, 8), dtype=np.float32)
    assembler.update_batch(tile, coords=np.asarray([[0, 0]]), edge_clipping=1)
    assembler.cleanup()
    for temp_filepath in temp_filepaths:
        assert not temp_filepath.exists(), (
            f"Temporary file {temp_filepath} should be deleted"
        )


def test_numpy_memmap_persistent_file(tmp_path):
    """Test that a persistent file created by a composed memmap builder is not deleted upon finalization."""
    filename = tmp_path / "persistent_heatmap.npy"

    assembler = MaskBuilder(
        source_extents=(16, 16),
        source_tile_extent=(8, 8),
        output_tile_extent=(8, 8),
        stride=(4, 4),
        n_channels=1,
        storage="memmap",
        aggregation=MeanAggregator,
        dtype=np.float32,
        filename=filename,
        overlap_counter_filename=filename.with_suffix(".overlaps" + filename.suffix),
    )

    tile_batch = np.ones((1, 1, 8, 8), dtype=np.float32)
    assembler.update_batch(tile_batch, coords=np.asarray([[0, 0]]), edge_clipping=1)

    assembler.cleanup()

    assert filename.exists(), (
        f"Persistent file {filename} should exist after finalization"
    )
    assert filename.with_suffix(".overlaps" + filename.suffix).exists(), (
        f"Persistent overlap file {filename.with_suffix('.overlaps' + filename.suffix)} should exist after finalization"
    )

    # Clean up
    filename.unlink()
    filename.with_suffix(".overlaps" + filename.suffix).unlink()
