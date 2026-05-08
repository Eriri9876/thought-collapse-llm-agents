"""
Best-effort cumulative USD tracker for paid LLM API calls.

A process-singleton counter; ``record(model, prompt_tokens, completion_tokens)``
is called from inside the LLM client wrapper after each successful API call.
On crossing $5 / $10 / $20 cumulative spend, prints a WARNING line. Each
threshold fires at most once per process (sticky).

Per-million-token prices below are best-effort as of 2026-05-07. This tracker
exists for budget alerting, **not invoicing** — verify large-budget runs
against the provider console. To update a price, edit ``PRICES_USD_PER_M``;
unknown models tracked tokens-only (USD = 0) and warn once.

Thread-safe (lock-protected) so async runners can record concurrently.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

# (input_usd_per_m, output_usd_per_m). Add new models here as they are wired in.
# SiliconFlow Qwen2.5 series: CNY-priced, converted at 7.25 CNY/USD, symmetric.
# DeepSeek: from api-docs.deepseek.com (V3.2 unified pricing, cache miss).
# Together: from together.ai/pricing (Instruct-Turbo endpoints).
PRICES_USD_PER_M: dict[str, tuple[float, float]] = {
    # SiliconFlow Qwen2.5 (¥/M → $/M @ 7.25)
    "Qwen/Qwen2.5-7B-Instruct":  (0.048, 0.048),  # ~¥0.35/M
    "Qwen/Qwen2.5-14B-Instruct": (0.097, 0.097),  # ~¥0.70/M
    "Qwen/Qwen2.5-32B-Instruct": (0.174, 0.174),  # ~¥1.26/M
    "Qwen/Qwen2.5-72B-Instruct": (0.570, 0.570),  # ~¥4.13/M

    # DeepSeek
    "deepseek-chat":     (0.28, 0.42),
    "deepseek-reasoner": (0.28, 0.42),

    # Together AI Llama-3.1 Instruct-Turbo
    "meta-llama/Llama-3.1-8B-Instruct-Turbo":  (0.18, 0.18),
    "meta-llama/Llama-3.1-70B-Instruct-Turbo": (0.88, 0.88),
}

WARNING_THRESHOLDS_USD: tuple[float, ...] = (5.0, 10.0, 20.0)


@dataclass
class _State:
    cost_usd: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    fired: set[float] = field(default_factory=set)
    unknown_models: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)


_state = _State()


def record(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Add a single API call to the running totals."""
    prices = PRICES_USD_PER_M.get(model)
    delta = 0.0
    if prices is not None:
        in_p, out_p = prices
        delta = (prompt_tokens / 1e6) * in_p + (completion_tokens / 1e6) * out_p

    with _state.lock:
        _state.in_tokens += prompt_tokens
        _state.out_tokens += completion_tokens

        if prices is None:
            if model not in _state.unknown_models:
                _state.unknown_models.add(model)
                print(f"[cost_tracker] WARNING: no price entry for {model!r}; "
                      f"tokens are tracked but USD is not")
            return

        prev = _state.cost_usd
        _state.cost_usd += delta
        new = _state.cost_usd
        for thr in WARNING_THRESHOLDS_USD:
            if prev < thr <= new and thr not in _state.fired:
                _state.fired.add(thr)
                print(f"[cost_tracker] WARNING: cumulative spend crossed "
                      f"${thr:.0f} (now ${new:.4f}; "
                      f"in={_state.in_tokens:,} out={_state.out_tokens:,})")


def summary() -> dict:
    with _state.lock:
        return {
            "cost_usd":         round(_state.cost_usd, 6),
            "in_tokens":        _state.in_tokens,
            "out_tokens":       _state.out_tokens,
            "fired_thresholds": sorted(_state.fired),
            "unknown_models":   sorted(_state.unknown_models),
        }


def print_summary() -> None:
    s = summary()
    print(f"[cost_tracker] final: ${s['cost_usd']:.4f}  "
          f"in={s['in_tokens']:,} out={s['out_tokens']:,}")


def reset() -> None:
    """Reset all counters. Useful for tests; rarely needed in production."""
    global _state
    _state = _State()
