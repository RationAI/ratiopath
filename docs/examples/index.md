# Usage Examples

This hub collects task-focused examples that show how to use `ratiopath` for common pathology data tasks.

The examples are organized by user intent.
Start with **Basic Functionalities** if you need one focused operation.
Move to **Pipelines** when you want to combine multiple `ratiopath` components into an end-to-end workflow.

## Basic Functionalities

### [Read Slides And Choose Resolution](./basic-functionalities/slide-metadata-and-resolution.md)

Problem solved: create a Ray dataset of slide metadata and pick a consistent working resolution across mixed whole-slide image files.

### [Inspect Slides And Preview Tiles](./basic-functionalities/inspect-slides-and-preview-tiles.md)

Problem solved: validate slide opening, level selection, and tile geometry visually before scaling to a larger workflow.

### [Generate Tile Coordinates](./basic-functionalities/tile-grid-generation.md)

Problem solved: generate reproducible tile coordinates for dense extraction, overlapping tiles, or edge-aware tiling without reading image pixels yet.

### [Extract Tiles Locally Without Ray](./basic-functionalities/extract-tiles-locally-without-ray.md)

Problem solved: extract and save tiles from a single slide using the core tiling primitives without building a distributed Ray dataset.

### [Add Annotation And Overlay Signals](./basic-functionalities/annotation-and-overlay-enrichment.md)

Problem solved: enrich tile metadata with geometry-based annotations or overlay-derived coverage values so that later filtering and training steps stay metadata-driven.

### [Split Data With Groups](./basic-functionalities/group-aware-splitting.md)

Problem solved: create train/test splits that preserve class balance while preventing leakage between related samples such as patients, slides, or cases.

### [Generate A Tissue Mask Overlay](./basic-functionalities/tissue-mask-generation.md)

Problem solved: turn a whole-slide image into a reusable raster tissue mask that later overlay-aware pipelines can query cheaply.

### [Aggregate Tensor Columns In Ray](./basic-functionalities/tensor-aggregations.md)

Problem solved: compute distributed mean and standard-deviation statistics over tensor-valued Ray dataset columns.

### [Apply Stain Augmentation](./basic-functionalities/stain-augmentation.md)

Problem solved: perturb histology tiles in stain space so training data better reflects scanner and lab variation.

### [Write TIFF Outputs From Ray](./basic-functionalities/write-tiff-outputs-from-ray.md)

Problem solved: export image arrays from a Ray dataset directly to TIFF files for inspection or downstream tools.

## Pipelines

### [Build A Distributed Tiling Pipeline](./pipelines/distributed-tiling-pipeline.md)

Problem solved: go from raw slide files to a scalable tile dataset that can be filtered, persisted, and reused for training or batch inference.

### [Create An Annotation-Aware Tile Dataset](./pipelines/annotation-aware-dataset-pipeline.md)

Problem solved: combine slide metadata, tile coordinates, and annotation coverage into one dataset suitable for supervised pathology workflows.

### [Build A Mask-First Tile Filtering Pipeline](./pipelines/mask-first-tile-filtering-pipeline.md)

Problem solved: generate reusable tissue-mask overlays and use them to filter tiles before expensive RGB tile reads.

### [Prepare A Training Dataset With Leakage-Safe Splits](./pipelines/prepare-a-training-dataset-with-leakage-safe-splits.md)

Problem solved: split at the slide or patient level first, then build train and test tile datasets without information leakage.

## How To Use This Section

Pick the example that matches the problem you are solving, run the minimal version first, and then adapt it to your own slide format, tile size, annotation source, or storage layout.
Some pages build a slide-level or tile-level metadata table first and then reuse it in later snippets on the same page.
Each page links back to the relevant API surface so you can move from workflow guidance to exact function signatures without leaving the docs structure.
