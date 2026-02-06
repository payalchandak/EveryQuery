import hashlib
import re
from typing import Any


def values_as_list(**kwargs) -> list[Any]:
    return list(kwargs.values())


def code_slug(code: str, n_hash: int = 10, prefix_len: int = 24) -> str:
    h = hashlib.sha1(code.encode("utf-8")).hexdigest()[:n_hash]
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", code).strip("_")[:prefix_len]
    return f"{prefix}__{h}" if prefix else h
