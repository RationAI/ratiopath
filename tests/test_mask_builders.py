import tempfile
from pathlib import Path

import numpy as np
import pytest

from ratiopath.masks.mask_builders import (
    AutoScalingPreprocessor,
    EdgeClippingPreprocessor,
    MaskBuilder,
    ScalarUniformExpansionPreprocessor,
)


@pytest.mark.parametrize("shape", [(1, 16, 16), (3, 32, 64)])
@pytest.mark.parametrize("mask_tile_extents", [(4, 4), (4, 8), (8, 4), (3, 6), (9, 6)])
@pytest.mark.parametrize("mask_tile_strides", [(2, 2), (3, 3), (3, 2), (2, 3)])
def test_scalar_uniform_averaging_2d(shape, mask_tile_extents, mask_tile_strides):
    channels = shape[0]
    mask_extents = np.asarray(shape[1:])
    mask_tile_extents = np.asarray(mask_tile_extents)
    mask_tile_strides = np.asarray(mask_tile_strides)

    batch_size = 4
    num_batches = 8

    gcds = np.gcd(mask_tile_extents, mask_tile_strides)
    adjusted_tile_extents = mask_tile_extents // gcds

    preprocessor = ScalarUniformExpansionPreprocessor(
        mask_tile_extents=mask_tile_extents, mask_tile_strides=mask_tile_strides
    )

    test_mask_builder = MaskBuilder(
        shape=shape,
        storage="inmemory",
        aggregation="mean",
        preprocessors=[preprocessor],
        dtype=np.float32,
    )

    # create scalar batch (B, C)
    example = np.ones((batch_size, channels), dtype=np.float32)

    for i in range(num_batches):
        coords_batch = np.random.rand(len(mask_extents), batch_size)
        coords_batch *= (mask_extents - mask_tile_extents)[:, np.newaxis]
        # force alignment to gcds
        coords_batch = (coords_batch.astype(np.int64) // gcds[:, np.newaxis]) * gcds[
            :, np.newaxis
        ]

        test_mask_builder.update_batch(example, coords_batch)

        # Access accumulator via storage
        accumulator = test_mask_builder.storage
        assert np.isclose(
            accumulator.sum(),
            (i + 1) * batch_size * channels * np.prod(adjusted_tile_extents),
            atol=1e-6,
        ), (
            f"Accumulator sum after batch {i + 1} should be {(i + 1) * batch_size * channels * np.prod(adjusted_tile_extents)}, is {accumulator.sum()}"
        )

    results = test_mask_builder.finalize()
    assert isinstance(results, dict)
    assembled = results["mask"]
    overlap = results["overlap_counter"]

    # assembled has shape (channels, H_adj, W_adj)
    assert assembled.shape[0] == channels, (
        "Assembled mask should have correct number of channels"
    )
    assert overlap.shape[0] == 1, "Overlap accumulator should have single channel"
    assert np.isclose(
        overlap.sum(),
        num_batches * batch_size * np.prod(adjusted_tile_extents),
        atol=1e-6,
    ), (
        f"Overlap sum should be {num_batches * batch_size * np.prod(adjusted_tile_extents)}, is {overlap.sum()}"
    )
    assert np.isclose(assembled.max(), 1.0, atol=1e-6), (
        f"Assembled max should be close to 1.0, is {assembled.max()}"
    )
    test_mask_builder.cleanup()


@pytest.mark.parametrize("shape", [(1, 16, 16), (3, 32, 64), (8, 400, 35)])
@pytest.mark.parametrize("mask_tile_extents", [(4, 4), (4, 8), (8, 4), (3, 6), (9, 6)])
@pytest.mark.parametrize("mask_tile_strides", [(2, 2), (3, 3), (3, 2), (2, 3)])
def test_scalar_uniform_max_2d(shape, mask_tile_extents, mask_tile_strides):
    channels = shape[0]
    mask_extents = np.asarray(shape[1:])
    mask_tile_extents = np.asarray(mask_tile_extents)
    mask_tile_strides = np.asarray(mask_tile_strides)

    batch_size = 4
    num_batches = 10

    gcds = np.gcd(mask_tile_extents, mask_tile_strides)
    adjusted_tile_extents = mask_tile_extents // gcds
    min_batch_increment = np.prod(adjusted_tile_extents) * channels

    preprocessor = ScalarUniformExpansionPreprocessor(
        mask_tile_extents=mask_tile_extents, mask_tile_strides=mask_tile_strides
    )

    test_mask_builder = MaskBuilder(
        shape=shape,
        storage="inmemory",
        aggregation="max",
        preprocessors=[preprocessor],
        dtype=np.float32,
    )

    for i in range(num_batches):
        coordinates_batch = np.random.rand(len(mask_extents), batch_size)
        coordinates_batch *= (mask_extents - mask_tile_extents)[:, np.newaxis]
        # force alignment to gcds
        coordinates_batch = (
            coordinates_batch.astype(np.int64) // gcds[:, np.newaxis]
        ) * gcds[:, np.newaxis]

        for coordinate in zip(*coordinates_batch, strict=True):
            for dim, coord in enumerate(coordinate):
                assert 0 <= coord <= mask_extents[dim] - mask_tile_extents[dim], (
                    f"Coordinate {coord} in dimension {dim} is out of bounds for mask dim {mask_extents[dim]} and tile extent {mask_tile_extents[dim]}"
                )
        value = float(i + 1)
        data = np.full((batch_size, channels), value, dtype=np.float32)
        test_mask_builder.update_batch(data, coordinates_batch)

        accumulator = test_mask_builder.storage
        assert accumulator.max() == value, (
            f"Max value in accumulator after batch {i + 1} should be {value}, is {accumulator.max()}"
        )
        assert (accumulator == float(i + 1)).sum() >= min_batch_increment, (
            f"Number of maxed pixels with value {float(i + 1)} after batch {i + 1} should be greater than {min_batch_increment}, is {(accumulator == float(i + 1)).sum()}"
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
    # no mismatch in source and mask extents here
    mask_extents = np.asarray(mask_extents)
    tile_extents = np.asarray(tile_extents)
    tile_strides = np.asarray((4, 4), dtype=np.int64)

    total_strides = np.ceil((mask_extents - tile_extents) / tile_strides).astype(
        np.int64
    )
    expected_mask_extents = total_strides * tile_strides + tile_extents

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

    edge_clip_prep = EdgeClippingPreprocessor(px_to_clip=clip)
    auto_scale_prep = AutoScalingPreprocessor(
        source_extents=mask_extents,
        source_tile_extents=tile_extents,
        source_tile_strides=tile_strides,
        mask_tile_extents=tile_extents,
    )

    assembler = MaskBuilder(
        shape=(channels, *auto_scale_prep.mask_extents),
        storage="memmap",
        aggregation="mean",
        preprocessors=[edge_clip_prep, auto_scale_prep],
        dtype=np.float32,
        filename=filename,
        overlap_counter_filename=overlap_counter_filename,
    )

    # create dummy data
    example_tile_batch = np.ones(
        (batch_size, channels, *tile_extents), dtype=np.float32
    )

    increment = example_tile_batch[
        ..., clip : tile_extents[0] - clip, clip : tile_extents[1] - clip
    ].sum()
    assert increment == batch_size * channels * (tile_extents[0] - 2 * clip) * (
        tile_extents[1] - 2 * clip
    ), (
        f"Checksum mismatch in example tile batch: Should be {batch_size * channels * (tile_extents[0] - 2 * clip) * (tile_extents[1] - 2 * clip)}, is {example_tile_batch[..., clip : tile_extents[0] - clip, clip : tile_extents[1] - clip].sum()}"
    )

    # add the tiles randomly to cover the heatmap
    for i in range(num_batches):
        coords_batch = np.random.rand(len(mask_extents), batch_size)
        coords_batch *= (mask_extents - (tile_extents - clip))[:, np.newaxis]
        coords_batch = coords_batch.astype(np.int64)
        assembler.update_batch(example_tile_batch, coords_batch)

        accumulator = assembler.storage
        assert accumulator.sum() == (i + 1) * increment, (
            f"Checksum mismatch in heatmap accumulator after update {i + 1}. Is {accumulator.sum()}, but expected {(i + 1) * increment}"
        )

    accumulator = assembler.storage
    assert (
        accumulator.sum()
        == num_batches
        * batch_size
        * (tile_extents[0] - 2 * clip)
        * (tile_extents[1] - 2 * clip)
        * channels
    ), (
        f"Checksum mismatch in heatmap accumulator after updates. Is {accumulator.sum()}, but expected {num_batches * batch_size * (tile_extents[0] - 2 * clip) * (tile_extents[1] - 2 * clip) * channels}"
    )

    results = assembler.finalize()
    assembled_heatmap = results["mask"]
    overlap_counter = results["overlap_counter"]

    assert assembled_heatmap.shape == (
        channels,
        *expected_mask_extents,
    ), (
        f"Assembled heatmap has incorrect shape: {assembled_heatmap.shape}, shall be {(channels, *expected_mask_extents)}"
    )
    assert overlap_counter.shape == (1, *expected_mask_extents), (
        f"Overlap counter has incorrect shape: {overlap_counter.shape}, shall be {(1, *expected_mask_extents)}"
    )
    assert overlap_counter.sum() == num_batches * batch_size * (
        tile_extents[0] - 2 * clip
    ) * (tile_extents[1] - 2 * clip), (
        f"Checksum mismatch in overlap counter after finalization. Is {overlap_counter.sum()}, but expected {num_batches * batch_size * (tile_extents[0] - 2 * clip) * (tile_extents[1] - 2 * clip)}"
    )
    assert assembled_heatmap.max() == 1.0, (
        f"Assembled heatmap values should be normalized to 1.0 after finalization: {assembled_heatmap.max()}"
    )
    assembler.cleanup()


def test_edge_clipping_clips_edges():
    """Simple test that ensures, that a tile put at [0,0], if clipped, does not write to the [0,0] corner of the heatmap."""
    clip = 1
    channels = 1
    mask_extents = np.asarray((16, 16))
    mask_tile_extents = np.asarray((8, 8))

    edge_clip_prep = EdgeClippingPreprocessor(px_to_clip=clip)
    auto_scale_prep = AutoScalingPreprocessor(
        source_extents=mask_extents,
        source_tile_extents=mask_tile_extents,
        source_tile_strides=np.asarray((4, 4)),
        mask_tile_extents=mask_tile_extents,
    )

    assembler = MaskBuilder(
        shape=(channels, *auto_scale_prep.mask_extents),
        storage="memmap",
        aggregation="mean",
        preprocessors=[edge_clip_prep, auto_scale_prep],
        dtype=np.float32,
    )

    tile = np.ones((1, channels, *mask_tile_extents), dtype=np.float32)
    assembler.update_batch(tile, coords_batch=np.asarray([[0], [0]]))
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


def test_numpy_memmap_tempfile_management(monkeypatch):
    """Test that temporary files created by NamedTemporaryFile are properly deleted."""
    captured_files = []
    original_namedtempfile = tempfile.NamedTemporaryFile

    def intercepting_namedtempfile(*args, **kwargs):
        temp_file = original_namedtempfile(*args, **kwargs)
        captured_files.append(Path(temp_file.name))
        return temp_file

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", intercepting_namedtempfile)

    mask_tile_extents = np.asarray([8, 8], dtype=np.int64)
    mask_extents = np.asarray([16, 16], dtype=np.int64)
    tile_strides = np.asarray([4, 4], dtype=np.int64)

    edge_clip_prep = EdgeClippingPreprocessor(px_to_clip=(1, 1, 1, 1))
    auto_scale_prep = AutoScalingPreprocessor(
        source_extents=mask_extents,
        source_tile_extents=mask_tile_extents,
        source_tile_strides=tile_strides,
        mask_tile_extents=mask_tile_extents,
    )

    assembler = MaskBuilder(
        shape=(1, *auto_scale_prep.mask_extents),
        storage="memmap",
        aggregation="mean",
        preprocessors=[edge_clip_prep, auto_scale_prep],
        dtype=np.float32,
    )

    assert len(captured_files) >= 2, (
        "Expected at least two temporary files for main and overlap accumulators"
    )
    temp_filepaths = captured_files.copy()
    tile = np.ones((1, 1, 8, 8), dtype=np.float32)
    assembler.update_batch(tile, coords_batch=np.asarray([[0], [0]]))
    del assembler
    for temp_filepath in temp_filepaths:
        assert not temp_filepath.exists(), (
            f"Temporary file {temp_filepath} should be deleted"
        )


def test_numpy_memmap_persistent_file(tmp_path):
    """Test that a persistent file created by a composed memmap builder is not deleted upon finalization."""
    filename = tmp_path / "persistent_heatmap.npy"

    mask_tile_extents = np.asarray([8, 8], dtype=np.int64)
    mask_extents = np.asarray([16, 16], dtype=np.int64)
    tile_strides = np.asarray([4, 4], dtype=np.int64)

    edge_clip_prep = EdgeClippingPreprocessor(px_to_clip=(1, 1, 1, 1))
    auto_scale_prep = AutoScalingPreprocessor(
        source_extents=mask_extents,
        source_tile_extents=mask_tile_extents,
        source_tile_strides=tile_strides,
        mask_tile_extents=mask_tile_extents,
    )

    assembler = MaskBuilder(
        shape=(1, *auto_scale_prep.mask_extents),
        storage="memmap",
        aggregation="mean",
        preprocessors=[edge_clip_prep, auto_scale_prep],
        dtype=np.float32,
        filename=filename,
    )

    tile_batch = np.ones((1, 1, *mask_tile_extents), dtype=np.float32)
    assembler.update_batch(tile_batch, coords_batch=np.asarray([[0], [0]]))

    del assembler

    assert filename.exists(), (
        f"Persistent file {filename} should exist after finalization"
    )
    assert filename.with_suffix(".overlaps" + filename.suffix).exists(), (
        f"Persistent overlap file {filename.with_suffix('.overlaps' + filename.suffix)} should exist after finalization"
    )

    # Clean up
    filename.unlink()
    filename.with_suffix(".overlaps" + filename.suffix).unlink()


@pytest.mark.parametrize("source_extents", [(32, 32), (64, 96)])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("source_tile_extents", [(8, 8), (6, 12)])
@pytest.mark.parametrize("mask_tile_extents", [(4, 4), (8, 8)])
def test_autoscaling_scalar_uniform_value_constant_stride(
    source_extents, channels, source_tile_extents, mask_tile_extents
):
    """Test composed builder with autoscaling and scalar tiling."""
    batch_size = 4
    num_batches = 8

    source_extents = np.asarray(source_extents)
    source_tile_extents = np.asarray(source_tile_extents)
    source_tile_strides = source_tile_extents // 2  # 50% overlap
    mask_tile_extents = np.asarray(mask_tile_extents)

    auto_scale_prep = AutoScalingPreprocessor(
        source_extents=source_extents,
        source_tile_extents=source_tile_extents,
        source_tile_strides=source_tile_strides,
        mask_tile_extents=mask_tile_extents,
    )

    mask_tile_strides = (source_tile_strides * mask_tile_extents) // source_tile_extents

    scalar_prep = ScalarUniformExpansionPreprocessor(
        mask_tile_extents=mask_tile_extents,
        mask_tile_strides=mask_tile_strides,  # strides in mask space
    )

    # Note the order of preprocessors!
    # 1. Scale coords (AutoScalingPreprocessor)
    # 2. Expand scalars and compress coords (ScalarUniformExpansionPreprocessor)
    builder = MaskBuilder(
        shape=(channels, *auto_scale_prep.mask_extents),
        storage="inmemory",
        aggregation="mean",
        preprocessors=[auto_scale_prep, scalar_prep],
        dtype=np.float32,
    )

    mask_tile_strides = (source_tile_strides * mask_tile_extents) // source_tile_extents
    total_strides = np.ceil(
        (source_extents - source_tile_extents) / source_tile_strides
    ).astype(np.int64)
    expected_mask_extents = total_strides * mask_tile_strides + mask_tile_extents

    gcds = np.gcd(mask_tile_extents, mask_tile_strides)
    adjusted_mask_tile_extents = mask_tile_extents // gcds
    compressed_mask_extents = expected_mask_extents // gcds

    # Verify accumulator shape matches adjusted dimensions
    accumulator = builder.storage
    assert accumulator.shape == (channels, *compressed_mask_extents), (
        f"Accumulator shape mismatch: {accumulator.shape} vs expected {(channels, *compressed_mask_extents)}"
    )

    # Create scalar batch data (B, C)
    scalar_data = np.ones((batch_size, channels), dtype=np.float32)

    for i in range(num_batches):
        # Generate coordinates in SOURCE space
        coords_batch = np.random.rand(len(source_extents), batch_size)
        coords_batch *= (source_extents - source_tile_extents)[:, np.newaxis]

        # Align to source tile strides
        coords_batch = (
            coords_batch // source_tile_strides[:, np.newaxis]
        ) * source_tile_strides[:, np.newaxis]
        coords_batch = coords_batch.astype(np.int64)

        builder.update_batch(scalar_data, coords_batch)

        # Verify accumulator sum increases
        expected_increment = (
            (i + 1) * batch_size * channels * np.prod(adjusted_mask_tile_extents)
        )
        assert np.isclose(accumulator.sum(), expected_increment, atol=1e-5), (
            f"Accumulator sum after batch {i + 1}: {accumulator.sum()} vs expected {expected_increment}"
        )

    results = builder.finalize()
    assembled = results["mask"]
    overlap = results["overlap_counter"]

    # Verify output shapes
    assert assembled.shape == (channels, *compressed_mask_extents), (
        f"Assembled mask shape mismatch: {assembled.shape} vs {(channels, *compressed_mask_extents)}"
    )
    assert overlap.shape == (1, *compressed_mask_extents), (
        f"Overlap counter shape mismatch: {overlap.shape} vs {(1, *compressed_mask_extents)}"
    )

    # Verify values are averaged correctly
    assert np.isclose(assembled.max(), 1.0, atol=1e-5), (
        f"Max value should be ~1.0 after averaging: {assembled.max()}"
    )
    assert overlap.sum() == num_batches * batch_size * np.prod(
        adjusted_mask_tile_extents
    ), (
        f"Overlap sum mismatch: {overlap.sum()} vs expected {num_batches * batch_size * np.prod(adjusted_mask_tile_extents)}"
    )
    builder.cleanup()
