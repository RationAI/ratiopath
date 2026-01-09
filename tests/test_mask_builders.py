import tempfile
from pathlib import Path

import numpy as np
import pytest

from ratiopath.masks.mask_builders import AveragingScalarUniformTiledNumpyMaskBuilder, MaxScalarUniformTiledNumpyMaskBuilder
from ratiopath.masks.mask_builders import AutoScalingScalarUniformValueConstantStrideMaskBuilder
from ratiopath.masks.mask_builders import (
    AutoScalingAveragingClippingNumpyMemMapMaskBuilder2D,
)


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

    test_mask_builder = AveragingScalarUniformTiledNumpyMaskBuilder(
        mask_extents=mask_extents,
        channels=channels,
        mask_tile_extents=mask_tile_extents,
        mask_tile_strides=mask_tile_strides,
    )

    # create scalar batch (B, C)
    example = np.ones((batch_size, channels), dtype=np.float32)

    for i in range(num_batches):
        coords_batch = np.random.rand(len(mask_extents), batch_size)
        coords_batch *= (np.array(mask_extents) - np.array(mask_tile_extents))[:, np.newaxis]
        # force alignment to gcds
        coords_batch = (coords_batch.astype(np.int64) // gcds[:, np.newaxis]) * gcds[
            :, np.newaxis
        ]

        test_mask_builder.update_batch(example, coords_batch)

        assert np.isclose(
            test_mask_builder.accumulator.sum(),
            (i + 1) * batch_size * channels * np.prod(adjusted_tile_extents),
            atol=1e-6,
        ), (
            f"Accumulator sum after batch {i + 1} should be {(i + 1) * batch_size * channels * np.prod(adjusted_tile_extents)}, is {test_mask_builder.accumulator.sum()}"
        )

    assembled, overlap = test_mask_builder.finalize()
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


@pytest.mark.parametrize("mask_extents", [(16, 16), (32, 64), (400, 35)])
@pytest.mark.parametrize("channels", [1, 3, 8])
@pytest.mark.parametrize("mask_tile_extents", [(4, 4), (4, 8), (8, 4), (3, 6), (9, 6)])
@pytest.mark.parametrize("mask_tile_strides", [(2, 2), (3, 3), (3, 2), (2, 3)])
def test_scalar_uniform_max_2d(mask_extents, channels, mask_tile_extents, mask_tile_strides):

    mask_extents = np.asarray(mask_extents)
    mask_tile_extents = np.asarray(mask_tile_extents)
    mask_tile_strides = np.asarray(mask_tile_strides)

    batch_size = 4
    num_batches = 10

    gcds = np.gcd(mask_tile_extents, mask_tile_strides)
    adjusted_tile_extents = mask_tile_extents // gcds
    min_batch_increment = np.prod(adjusted_tile_extents) * channels

    test_mask_builder = MaxScalarUniformTiledNumpyMaskBuilder(
        mask_extents=mask_extents,
        channels=channels,
        mask_tile_extents=mask_tile_extents,
        mask_tile_strides=mask_tile_strides,
    )

    for i in range(num_batches):
        coordinates_batch = np.random.rand(len(mask_extents), batch_size)
        coordinates_batch *= (np.array(mask_extents) - np.array(mask_tile_extents))[
            :, np.newaxis
        ]
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
        assert test_mask_builder.accumulator.max() == value, (
            f"Max value in accumulator after batch {i + 1} should be {value}, is {test_mask_builder.accumulator.max()}"
        )
        assert (
            test_mask_builder.accumulator == float(i + 1)
        ).sum() >= min_batch_increment, (
            f"Number of maxed pixels with value {float(i + 1)} after batch {i + 1} should be greater than {min_batch_increment}, is {(test_mask_builder.accumulator == float(i + 1)).sum()}"
        )
    nonfinal_acc = test_mask_builder.accumulator.copy()
    (final_acc,) = test_mask_builder.finalize()
    # final_acc should contain the maximum value (num_batches)
    assert (final_acc == nonfinal_acc).all(), (
        "Finalized accumulator should be equal to non-finalized accumulator for max aggregation"
    )


@pytest.mark.parametrize("clip", [0, 1, 3])
@pytest.mark.parametrize("tile_extents", [(8, 8), (12, 9)])
@pytest.mark.parametrize("channels", [1, 5])
@pytest.mark.parametrize("mask_extents", [(32, 64), (64, 32), (100, 101)])
@pytest.mark.parametrize("acc_filepath", [None, "test_heatamp.npy"])
@pytest.mark.parametrize("overlap_counter_filepath", [None, "test_overlap.counter.npy"])
def test_edge_clipping_heatmap_assembler(
    clip, tile_extents, channels, mask_extents, acc_filepath, overlap_counter_filepath, tmp_path
):
    # no mismatch in source and mask extents here
    mask_extents = np.asarray(mask_extents)
    tile_extents = np.asarray(tile_extents)
    tile_strides = np.asarray((4, 4), dtype=np.int64)

    total_strides = np.ceil((mask_extents - tile_extents) / tile_strides).astype(np.int64)
    expected_mask_extents = total_strides * tile_strides + tile_extents

    num_batches = 4
    batch_size = 8

    np.random.seed(0)
    if acc_filepath is not None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        acc_filepath = Path((tmp_path / acc_filepath).as_posix())
    if overlap_counter_filepath is not None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        overlap_counter_filepath = Path(
            (tmp_path / overlap_counter_filepath).as_posix()
        )

    assembler = AutoScalingAveragingClippingNumpyMemMapMaskBuilder2D(
        source_extents=mask_extents,
        source_tile_extents=tile_extents,
        source_tile_strides=tile_strides,
        mask_tile_extents=tile_extents,
        channels=channels,
        clip=clip,
        accumulator_filepath=acc_filepath,
        overlap_counter_filepath=overlap_counter_filepath,
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
        coords_batch *= (
            np.asarray(mask_extents) - (np.asarray(tile_extents) - clip)
        )[:, np.newaxis]
        coords_batch = coords_batch.astype(np.int64)
        assembler.update_batch(example_tile_batch, coords_batch)
        # plt.imshow((assembler.accumulator.transpose(1, 2, 0)*32).astype(np.uint8))
        # plt.show()
        assert assembler.accumulator.sum() == (i + 1) * increment, (
            f"Checksum mismatch in heatmap accumulator after update {i + 1}. Is {assembler.accumulator.sum()}, but expected {(i + 1) * increment}"
        )

    assert (
        assembler.accumulator.sum()
        == num_batches
        * batch_size
        * (tile_extents[0] - 2 * clip)
        * (tile_extents[1] - 2 * clip)
        * channels
    ), (
        f"Checksum mismatch in heatmap accumulator after updates. Is {assembler.accumulator.sum()}, but expected {num_batches * batch_size * (tile_extents[0] - 2 * clip) * (tile_extents[1] - 2 * clip) * channels}"
    )

    assembled_heatmap, overlap_counter = assembler.finalize()
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


def test_edge_clipping_clips_edges():
    """SImple test that ensures, that a tile put at [0,0], if clipped, does not write to the [0,0] corner of the heatmap."""
    clip = 1
    mask_extents = np.asarray((16, 16))
    mask_tile_extents = np.asarray((8, 8))
    channels = 1
    assembler = AutoScalingAveragingClippingNumpyMemMapMaskBuilder2D(
        source_extents=mask_extents,
        source_tile_extents=mask_tile_extents,
        source_tile_strides=np.asarray((4, 4)),
        mask_tile_extents=mask_tile_extents,
        clip=clip,
        channels=channels,
    )
    tile = np.ones((1, channels, *mask_tile_extents), dtype=np.float32)
    assembler.update_batch(tile, coords_batch=np.asarray([[0], [0]]))
    assembled_heatmap, overlap_counter = assembler.finalize()
    assert (assembled_heatmap[0, 0] == 0.0).all(), (
        "Top-left corner of assembled heatmap should be zero due to clipping"
    )
    assert (overlap_counter[0, 0] == 0.0).all(), (
        "Top-left corner of overlap counter should be zero due to clipping"
    )


def test_numpy_memmap_tempfile_management(monkeypatch):
    """Test that temporary files created by NamedTemporaryFile are properly deleted."""
    captured_files = []
    original_namedtempfile = (
        tempfile.NamedTemporaryFile
    )  # needs to be declared as a local variable to avoid recursion by monkeypatch

    def intercepting_namedtempfile(*args, **kwargs):
        temp_file = original_namedtempfile(*args, **kwargs)
        captured_files.append(Path(temp_file.name))
        return temp_file

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", intercepting_namedtempfile)

    mask_tile_extents = np.asarray([8, 8], dtype=np.int64)
    mask_extents = np.asarray([16, 16], dtype=np.int64)
    tile_strides = np.asarray([4, 4], dtype=np.int64)

    assembler = AutoScalingAveragingClippingNumpyMemMapMaskBuilder2D(
        source_extents=mask_extents,
        source_tile_extents=mask_tile_extents,
        source_tile_strides=tile_strides,
        mask_tile_extents=mask_tile_extents,
        channels=1,
        clip=1,
    )

    assert len(captured_files) >= 1, (
        "Expected at least one temporary file to be created"
    )
    temp_filepath = captured_files[0]

    tile = np.ones((1, 1, 8, 8), dtype=np.float32)
    assembler.update_batch(tile, coords_batch=np.asarray([[0], [0]]))

    del assembler

    assert not temp_filepath.exists(), (
        f"Temporary file {temp_filepath} should be deleted"
    )


def test_numpy_memmap_persistent_file(tmp_path):
    """Test that a persistent file created by AveragingClippingNumpyMemMapMaskBuilder2D is not deleted upon finalization."""
    filepath = tmp_path / "persistent_heatmap.npy"

    mask_tile_extents = np.asarray([8, 8], dtype=np.int64)
    mask_extents = np.asarray([16, 16], dtype=np.int64)
    tile_strides = np.asarray([4, 4], dtype=np.int64)

    assembler = AutoScalingAveragingClippingNumpyMemMapMaskBuilder2D(
        source_extents=mask_extents,
        source_tile_extents=mask_tile_extents,
        source_tile_strides=tile_strides,
        mask_tile_extents=mask_tile_extents,
        clip=1,
        channels=1,
        accumulator_filepath=filepath,
        overlap_counter_filepath=filepath.with_suffix(".overlaps"+filepath.suffix),
    )

    tile_batch = np.ones((1, 1, *mask_tile_extents), dtype=np.float32)
    assembler.update_batch(tile_batch, coords_batch=np.asarray([[0], [0]]))
    
    del assembler

    assert filepath.exists(), (
        f"Persistent file {filepath} should exist after finalization"
    )

    assert filepath.with_suffix(".overlap"+filepath.suffix).exists(), (
        f"Persistent overlap file {filepath.with_suffix('.overlap'+filepath.suffix)} should exist after finalization"
    )

    # Clean up
    filepath.unlink()


@pytest.mark.parametrize("source_extents", [(32, 32), (64, 96)])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("source_tile_extents", [(8, 8), (6, 12)])
@pytest.mark.parametrize("mask_tile_extents", [(4, 4), (8, 8)])
def test_autoscaling_scalar_uniform_value_constant_stride(
    source_extents, channels, source_tile_extents, mask_tile_extents
):
    """Test AutoScalingScalarUniformValueConstantStrideMaskBuilder with autoscaling and scalar tiling."""
    batch_size = 4
    num_batches = 8
    
    source_extents = np.asarray(source_extents)
    source_tile_extents = np.asarray(source_tile_extents)
    source_tile_strides = source_tile_extents // 2  # 50% overlap
    mask_tile_extents = np.asarray(mask_tile_extents)


    
    builder = AutoScalingScalarUniformValueConstantStrideMaskBuilder(
        source_extents=source_extents,
        channels=channels,
        source_tile_extents=source_tile_extents,
        source_tile_strides=source_tile_strides,
        mask_tile_extents=mask_tile_extents,
    )
    
    mask_tile_strides = (source_tile_strides * mask_tile_extents) // source_tile_extents
    total_strides = np.ceil((source_extents - source_tile_extents) / source_tile_strides).astype(np.int64)
    expected_mask_extents = total_strides * mask_tile_strides + mask_tile_extents
    
    gcds = np.gcd(mask_tile_extents, mask_tile_strides)
    adjusted_mask_tile_extents = mask_tile_extents // gcds
    compressed_mask_extents = expected_mask_extents // gcds
    
    # Verify accumulator shape matches adjusted dimensions
    assert builder.accumulator.shape == (channels, *compressed_mask_extents), (
        f"Accumulator shape mismatch: {builder.accumulator.shape} vs expected {(channels, *compressed_mask_extents)}"
    )
    
    # Create scalar batch data (B, C)
    scalar_data = np.ones((batch_size, channels), dtype=np.float32)
    
    for i in range(num_batches):
        # Generate coordinates in SOURCE space
        coords_batch = np.random.rand(len(source_extents), batch_size)
        coords_batch *= (source_extents - source_tile_extents)[:, np.newaxis]
        
        # Align to source tile strides
        coords_batch = (coords_batch // source_tile_strides[:, np.newaxis]) * source_tile_strides[:, np.newaxis]
        coords_batch = coords_batch.astype(np.int64)
        
        builder.update_batch(scalar_data, coords_batch)
        
        # Verify accumulator sum increases
        expected_increment = (i + 1) * batch_size * channels * np.prod(adjusted_mask_tile_extents)
        assert np.isclose(builder.accumulator.sum(), expected_increment, atol=1e-5), (
            f"Accumulator sum after batch {i + 1}: {builder.accumulator.sum()} vs expected {expected_increment}"
        )
    
    assembled, overlap = builder.finalize()
    
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
    assert overlap.sum() == num_batches * batch_size * np.prod(adjusted_mask_tile_extents), (
        f"Overlap sum mismatch: {overlap.sum()} vs expected {num_batches * batch_size * np.prod(adjusted_mask_tile_extents)}"
    )
