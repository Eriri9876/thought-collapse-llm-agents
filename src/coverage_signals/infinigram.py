"""
InfiniGram n-gram frequency signal.

Hits the public InfiniGram API (https://api.infini-gram.io/) to count how
many times a query string occurs in a fixed pretraining-corpus index. Used
as a *direct* coverage signal — unlike pageview popularity (a proxy for
exposure) this counts actual occurrences in the corpus.

Default index ``v4_dolma-v1_7_llama`` (Dolma v1.7, ~3T tokens, Llama
tokenizer) — the largest available corpus on the public API. The user's
backup choice ``v4_rpj_c4_train`` is *not* a valid index; the closest
matches are ``v4_rpj_llama_s4`` (RedPajama, 1.4T) and ``v4_c4train_llama``
(C4, 200B). All 13 valid indexes are listed in
https://infini-gram.readthedocs.io/en/latest/api.html.

Caching: a separate ``diskcache`` instance under ``.cache/infinigram/`` so
InfiniGram lookups don't share key-space with LLM responses. Same Windows
``timeout=60`` rule as ``src/cache.py``.

The API has no published rate limit but advises retry-on-failure; we
exponential-backoff up to 3 attempts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import requests
from diskcache import Cache

API_URL = "https://api.infini-gram.io/"
DEFAULT_INDEX = "v4_dolma-v1_7_llama"
BACKUP_INDEX = "v4_rpj_llama_s4"
HEADERS = {"Content-Type": "application/json"}

_CACHE_DIR = Path(os.getenv("INFINIGRAM_CACHE_DIR", ".cache/infinigram"))
_cache: Cache | None = None


def _get_cache() -> Cache:
    global _cache
    if _cache is None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache = Cache(str(_CACHE_DIR), timeout=60)
    return _cache


def _key(index: str, query: str) -> str:
    blob = json.dumps({"i": index, "q": query},
                      sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def count(
    query: str,
    *,
    index: str = DEFAULT_INDEX,
    retries: int = 3,
    timeout: float = 30.0,
    use_cache: bool = True,
) -> dict | None:
    """Return the raw API response for a count query, or ``None`` on failure.

    Response shape: ``{count, approx, latency, token_ids, tokens}``.
    """
    if not query or not query.strip():
        return None
    cache = _get_cache()
    cache_key = _key(index, query)
    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    body = {"index": index, "query_type": "count", "query": query}
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, json=body, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            # API may surface "error" key on bad query; treat as miss
            if "error" in data and "count" not in data:
                return None
            if use_cache:
                cache.set(cache_key, data)
            return data
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def frequency_signal(
    query: str,
    *,
    index: str = DEFAULT_INDEX,
) -> dict:
    """High-level wrapper: count the query and return ``log10(1+count)``.

    Keys returned regardless of success / failure:
        index        which InfiniGram corpus index was queried
        query        the input string verbatim
        count        int, occurrence count (None on API failure)
        log10_count  log10(1 + count) — the headline signal (None on miss)
        approx       whether the count is approximate (per API)
        n_tokens     token count of the query in the index's tokenizer
    """
    res = count(query, index=index)
    if res is None or "count" not in res:
        return {"index": index, "query": query, "count": None,
                "log10_count": None, "approx": None, "n_tokens": None}
    c = int(res["count"])
    return {
        "index":       index,
        "query":       query,
        "count":       c,
        "log10_count": math.log10(1 + c),
        "approx":      bool(res.get("approx", False)),
        "n_tokens":    len(res.get("tokens", [])) or None,
    }
