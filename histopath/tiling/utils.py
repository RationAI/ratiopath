import hashlib
from typing import Any


def row_hash(row: dict[str, Any], column: str = "id") -> dict[str, Any]:
    row[column] = hashlib.sha256(str(row).encode()).hexdigest()
    return row
