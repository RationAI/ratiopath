import hashlib
from collections.abc import Callable
from typing import Any


def row_hash(
    row: dict[str, Any],
    column: str = "id",
    algorithm: Callable[[bytes], hashlib._hashlib.HASH] = hashlib.sha256,  # type: ignore[name-defined]
) -> dict[str, Any]:
    """Hashes a row (dictionary) using SHA256 and adds the hash as a new column.

    Args:
        row: The dictionary (row) to hash.
        column: The name of the column to store the hash. Defaults to "id".
        algorithm: The hashing algorithm to use. Defaults to hashlib.sha256.

    Returns:
        The modified row (dictionary) with the SHA256 hash added.
    """
    row[column] = algorithm(str(row).encode()).hexdigest()
    return row
