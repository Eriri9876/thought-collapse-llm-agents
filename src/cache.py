"""
Disk-backed cache for LLM responses (and other deterministic API calls).

Wraps `diskcache.Cache` so repeated runs of the same (model, messages,
temperature, seed) tuple don't re-spend API credits. Build the key with
`make_key(...)`, then read/write through the `Cache` returned by
`get_cache()`:

    from src.cache import get_cache, make_key
    cache = get_cache()
    key = make_key(model, messages, temperature=0.0, seed=42)
    cached = cache.get(key)
    if cached is not None:
        return cached
    response = api_call(...)
    cache.set(key, response)

Windows note: `Cache(timeout=60)` is **mandatory** here. SQLite's
default lock timeout is short and the cache lives behind a SQLite
file; under concurrent writes (async runners) it raises
`OperationalError: database is locked`. Do not lower the timeout
without re-testing under async load.

Cache directory defaults to `.cache/llm/` in the project root and is
gitignored. Override with the `LLM_CACHE_DIR` env var.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from diskcache import Cache

_DEFAULT_DIR = Path(os.getenv("LLM_CACHE_DIR", ".cache/llm"))
_DEFAULT_TIMEOUT = 60  # seconds — mandatory on Windows; see module docstring.

_cache: Cache | None = None


def get_cache(
    directory: Path | str | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Cache:
    """Return a process-singleton ``diskcache.Cache``.

    The first call wins: later calls ignore ``directory`` / ``timeout`` and
    return the existing instance. Use :func:`close` to reset.
    """
    global _cache
    if _cache is None:
        path = Path(directory) if directory is not None else _DEFAULT_DIR
        path.mkdir(parents=True, exist_ok=True)
        _cache = Cache(str(path), timeout=timeout)
    return _cache


def make_key(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    seed: int | None = None,
    extra: dict | None = None,
) -> str:
    """Stable SHA-256 key over (model, messages, temperature, seed, extra).

    JSON dump uses ``sort_keys=True`` so dict ordering doesn't change the key.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "extra": extra or {},
    }
    blob = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def stats() -> dict[str, Any]:
    """Return ``{'size_bytes': int, 'count': int}`` for the singleton cache."""
    c = get_cache()
    return {"size_bytes": c.volume(), "count": len(c)}


def close() -> None:
    """Close the singleton. The next :func:`get_cache` call reopens it."""
    global _cache
    if _cache is not None:
        _cache.close()
        _cache = None
