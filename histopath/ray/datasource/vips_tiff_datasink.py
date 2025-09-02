from typing import Any

import pyarrow

from ray.data.datasource.file_datasink import RowBasedFileDatasink


class VipsTiffDatasink(RowBasedFileDatasink):
    """A datasink for saving image data as TIFF files using libvips.

    This datasink uses pyvips to efficiently save image data with support for
    various TIFF formats, including BigTIFF.
    """

    def __init__(
        self,
        path: str,
        data_column: str,
        options_column: str | None = None,
        default_options: dict[str, Any] | None = None,
        **file_datasink_kwargs: Any,
    ) -> None:
        """Initialize a VipsTIFFDatasink.

        Args:
            path: Output path for the TIFF files
            data_column: Column name containing image data (numpy array)
            options_column: Optional column name containing TIFF save options as dict
            default_options: Default options for TIFF saving
            **file_datasink_kwargs: Additional arguments for the file datasink
        """
        super().__init__(path, file_format="tiff", **file_datasink_kwargs)
        self.data_column = data_column
        self.options_column = options_column
        self.default_options = default_options or {}

    def write_row_to_file(self, row: dict[str, Any], file: pyarrow.NativeFile) -> None:
        from pyvips import Image

        image = Image.new_from_array(row[self.data_column])
        options = self.default_options.copy()
        if self.options_column is not None:
            options.update(row.get(self.options_column, {}))

        buffer = image.tiffsave_buffer(**options)
        file.write(buffer)
