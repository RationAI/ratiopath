"""
Annotation Parsing and Tiling for histopath
==========================================

This module provides support for parsing and processing annotations from multiple formats
for use with ray.Dataset for efficient batch processing of histopathology data.

Supported Annotation Formats
----------------------------

1. **ASAP Format** (.xml)
   - XML-based format from Automated Slide Analysis Platform
   - Supports polygon and point annotations

2. **GeoJSON Format** (.json, .geojson)  
   - Standard geographic JSON format
   - Supports Feature/FeatureCollection structures with Polygon and Point geometries

3. **QuPath Format** (.json)
   - JSON format from QuPath software
   - Supports annotation objects with geometry or ROI structures

Basic Usage
-----------

### Using Individual Parsers

```python
from histopath.parsers import ASAPParser, GeoJSONParser, QuPathParser

# Parse ASAP annotations
asap_parser = ASAPParser("annotations.xml")
polygons = list(asap_parser.get_polygons())
points = list(asap_parser.get_points())

# Parse GeoJSON annotations  
geojson_parser = GeoJSONParser("annotations.geojson")
polygons = list(geojson_parser.get_polygons())
points = list(geojson_parser.get_points())

# Parse QuPath annotations
qupath_parser = QuPathParser("annotations.json")
polygons = list(qupath_parser.get_polygons())
points = list(qupath_parser.get_points())
```

### Using with Ray Dataset

```python
import ray
from histopath.ray.annotations import map_annotations

# Create a dataset with annotation paths
ds = ray.data.from_items([
    {"slide_id": "slide_001", "annotation_path": "/path/to/annotations1.xml"},
    {"slide_id": "slide_002", "annotation_path": "/path/to/annotations2.geojson"},
    {"slide_id": "slide_003", "annotation_path": "/path/to/annotations3.json"},
])

# Process annotations in parallel
ds_with_annotations = ds.map_batches(map_annotations)

# The result will have additional columns: 'polygons' and 'points'
# Each containing lists of shapely geometry objects
```

### Advanced Usage with Filtering

```python
from shapely.geometry import Polygon

# Define a region of interest
roi = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])

# Process with ROI filtering
ds_filtered = ds.map_batches(
    lambda batch: map_annotations(
        batch, 
        roi=roi,
        annotation_filter_regex=r"tumor.*"  # Filter annotations by name pattern
    )
)
```

Implementation Details
----------------------

### Parser Selection
The system automatically selects the appropriate parser based on file extension and content:

- `.xml` files → ASAPParser
- `.json`/`.geojson` files → Automatically detected as GeoJSON or QuPath based on content structure

### Error Handling
- Missing files result in empty annotation lists rather than errors
- Malformed files are handled gracefully with empty results
- The system includes fallback implementations for development without external dependencies

### Performance
- Designed for batch processing with Ray for parallel execution
- Lazy evaluation of annotations (generators used where possible)
- Minimal memory footprint per annotation file

API Reference
-------------

### AbstractParser
Base class for all annotation parsers.

**Methods:**
- `get_polygons() -> Iterable[shapely.Polygon]`
- `get_points() -> Iterable[shapely.Point]`

### map_annotations(rows, annotation_path_column="annotation_path", roi=None, annotation_filter_regex=None)
Process annotation files and add parsed data to dataset batches.

**Parameters:**
- `rows`: Dictionary containing batch data
- `annotation_path_column`: Column name containing annotation file paths (default: "annotation_path")  
- `roi`: Optional shapely geometry for spatial filtering
- `annotation_filter_regex`: Optional regex pattern for annotation filtering

**Returns:**
Dictionary with original data plus "polygons" and "points" columns containing lists of shapely objects.

Dependencies
------------
- shapely: For geometric operations
- ray[data]: For dataset processing (optional - graceful fallback provided)
- Standard library: xml.etree.ElementTree, json, re, pathlib
"""