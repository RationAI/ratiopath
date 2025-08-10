import pytest

from histopath.ray.datasource import MetaDatasource, OmeTypesMetaDatasource, OpenSlideMetaDatasource


def test_metadatasource_imports():
    """Test that all metadata source classes can be imported."""
    
    # Just verify classes can be imported and instantiated with basic params
    # We can't test actual functionality without sample files
    
    assert MetaDatasource is not None
    assert OmeTypesMetaDatasource is not None  
    assert OpenSlideMetaDatasource is not None


def test_metadatasource_file_detection():
    """Test file type detection in MetaDatasource."""
    
    # Create a temporary instance to test the method
    datasource = MetaDatasource(
        paths=[],  # Empty paths for testing
        level=0,
        tile_extent=(256, 256),
        stride=(256, 256),
    )
    
    # Test OME-TIFF detection
    assert datasource._is_ome_tiff("test.ome.tiff") is True
    assert datasource._is_ome_tiff("TEST.OME.TIFF") is True
    assert datasource._is_ome_tiff("sample.ome.tif") is True
    assert datasource._is_ome_tiff("SAMPLE.OME.TIF") is True
    
    # Test non-OME files
    assert datasource._is_ome_tiff("test.svs") is False
    assert datasource._is_ome_tiff("sample.tiff") is False
    assert datasource._is_ome_tiff("image.ndpi") is False


def test_file_extensions_coverage():
    """Test that file extensions lists include expected formats."""
    
    from histopath.ray.datasource.openslide_metadatasource import FILE_EXTENSIONS as openslide_exts
    from histopath.ray.datasource.ometypes_metadatasource import FILE_EXTENSIONS as ome_exts
    from histopath.ray.datasource.metadatasource import FILE_EXTENSIONS as combined_exts
    
    # Check that OME extensions are included
    assert "ome.tiff" in ome_exts
    assert "ome.tif" in ome_exts
    
    # Check that OpenSlide extensions are included  
    assert "svs" in openslide_exts
    assert "tiff" in openslide_exts
    assert "ndpi" in openslide_exts
    
    # Check that combined list includes both
    assert "ome.tiff" in combined_exts
    assert "svs" in combined_exts
    
    # Verify combined list has more extensions than individual lists
    assert len(combined_exts) >= len(openslide_exts)
    assert len(combined_exts) >= len(ome_exts)


if __name__ == "__main__":
    test_metadatasource_imports()
    test_metadatasource_file_detection()
    test_file_extensions_coverage()
    print("All metadata tests passed!")