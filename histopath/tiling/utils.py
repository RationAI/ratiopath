import hashlib
from typing import Any


def row_hash(row: dict[str, Any], column: str = "id") -> dict[str, Any]:
    """Hashes a row (dictionary) using SHA256 and adds the hash as a new column.

    Args:
        row: The dictionary (row) to hash.
        column: The name of the column to store the hash. Defaults to "id".

    Returns:
        The modified row (dictionary) with the SHA256 hash added.
    """
    row[column] = hashlib.sha256(str(row).encode()).hexdigest()
    return row
