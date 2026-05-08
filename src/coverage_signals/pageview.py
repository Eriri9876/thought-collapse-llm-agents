"""
Wikipedia pageview popularity signal.

For a query string (typically a head entity from
:mod:`src.coverage_signals.ner`), resolves the best-matching Wikipedia
article via the standard search API, then queries the Wikimedia REST
``pageviews`` endpoint for monthly view counts in a fixed window.

The signal returned is :math:`\\log_{10}(1 + \\overline{V})` where
:math:`\\overline{V}` is the arithmetic mean of monthly views over the
window. Log-transform compresses the heavy tail (popular articles get
millions/month, obscure ones get tens).

API:
    - search:    https://en.wikipedia.org/w/api.php
    - pageviews: https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article

Free, no auth. Rate limit: ~200 req/sec per IP — sequential calls are
fine for our volume (≤2000 questions per experiment).

Default window: the 12 most recent COMPLETE calendar months. Override
with the ``window=(start_yyyymmdd, end_yyyymmdd)`` argument.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import requests
from diskcache import Cache

PAGEVIEW_BASE = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
)
WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "thought-collapse-research/1.0 (academic research)"}

_CACHE_DIR = Path(os.getenv("PAGEVIEW_CACHE_DIR", ".cache/pageview"))
_cache_obj: Cache | None = None


def _get_cache() -> Cache:
    """Process-singleton diskcache for popularity_signal results.

    Cached value is the full ``popularity_signal`` dict, keyed by
    ``(query, window)``. Windows-mandatory ``timeout=60`` (see
    ``src/cache.py``).
    """
    global _cache_obj
    if _cache_obj is None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_obj = Cache(str(_CACHE_DIR), timeout=60)
    return _cache_obj


def _signal_key(query: str, window: tuple[str, str] | None) -> str:
    payload = json.dumps(
        {"q": query, "w": list(window) if window else None},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _default_window() -> tuple[str, str]:
    """Return (start, end) ``YYYYMMDD`` for the last 12 complete months."""
    today = date.today()
    # End = last day of the previous calendar month.
    end = today.replace(day=1) - timedelta(days=1)
    # Start = first day of (end - 11 months).
    start_year = end.year
    start_month = end.month - 11
    while start_month <= 0:
        start_month += 12
        start_year -= 1
    start = date(start_year, start_month, 1)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def resolve_title(query: str, retries: int = 3) -> str | None:
    """Best-matching Wikipedia article title for ``query``, or ``None``."""
    params = {
        "action": "query", "format": "json",
        "list": "search", "srsearch": query, "srlimit": 1,
    }
    for attempt in range(retries):
        try:
            r = requests.get(WIKI_SEARCH, params=params, headers=HEADERS,
                             timeout=15)
            results = r.json().get("query", {}).get("search", [])
            return results[0]["title"] if results else None
        except Exception:
            if attempt < retries - 1:
                time.sleep(2)
    return None


def fetch_pageviews(
    title: str,
    *,
    project: str = "en.wikipedia",
    access: str = "all-access",
    agent: str = "all-agents",
    granularity: str = "monthly",
    window: tuple[str, str] | None = None,
    retries: int = 3,
) -> list[int] | None:
    """Return monthly view counts for ``title`` over ``window``; None on miss."""
    start, end = window or _default_window()
    article = quote(title.replace(" ", "_"), safe="")
    url = (f"{PAGEVIEW_BASE}/{project}/{access}/{agent}/{article}/"
           f"{granularity}/{start}/{end}")
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 404:
                return None
            data = r.json()
            return [item["views"] for item in data.get("items", [])]
        except Exception:
            if attempt < retries - 1:
                time.sleep(2)
    return None


def popularity_signal(
    query: str,
    *,
    window: tuple[str, str] | None = None,
    use_cache: bool = True,
) -> dict:
    """Compute the full popularity signal for one query.

    Returns a dict with keys:
        title         resolved Wikipedia title (None if unresolved)
        monthly_views list of monthly counts (empty if missing)
        mean_views    arithmetic mean (None if missing)
        log10_views   log10(1 + mean_views) — the headline signal

    Cached on ``(query, window)`` under ``.cache/pageview/`` so repeated
    runs (e.g. multi-entity extractor visiting the same head twice)
    don't re-hit the Wikimedia API.
    """
    cache = _get_cache()
    key = _signal_key(query, window)
    if use_cache:
        cached = cache.get(key)
        if cached is not None:
            return cached

    title = resolve_title(query)
    if title is None:
        result = {"title": None, "monthly_views": [],
                  "mean_views": None, "log10_views": None}
        cache.set(key, result)
        return result
    monthly = fetch_pageviews(title, window=window)
    if not monthly:
        result = {"title": title, "monthly_views": [],
                  "mean_views": None, "log10_views": None}
        cache.set(key, result)
        return result
    mean = sum(monthly) / len(monthly)
    result = {
        "title":         title,
        "monthly_views": monthly,
        "mean_views":    mean,
        "log10_views":   math.log10(1 + mean),
    }
    cache.set(key, result)
    return result
