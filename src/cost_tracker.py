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

    # OpenRouter Llama-3.1 (DeepInfra bf16 endpoint, verified 2026-05-09).
    # 8B  : deepinfra/bf16  → $0.02 in / $0.05 out
    # 70B : deepinfra/base  → $0.40 in / $0.40 out  (bf16; Turbo is fp8 — do not use)
    "meta-llama/llama-3.1-8b-instruct":  (0.02, 0.05),
    "meta-llama/llama-3.1-70b-instruct": (0.40, 0.40),
}

WARNING_THRESHOLDS_USD: tuple[float, ...] = (5.0, 10.0, 20.0)


class BudgetExceeded(RuntimeError):
    """Raised by :func:`record` when cumulative spend crosses the hard cap."""


@dataclass
class _State:
    cost_usd: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    fired: set[float] = field(default_factory=set)
    unknown_models: set[str] = field(default_factory=set)
    hard_cap_usd: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_state = _State()

# Per-trace completion-token counter, used by ReAct to enforce
# max_total_output_tokens. Thread-local so concurrent traces do not interfere.
_run_counter = threading.local()


def reset_run_counter() -> None:
    """Zero the per-trace completion-token counter for the current thread."""
    _run_counter.completion_tokens = 0


def add_run_completion_tokens(n: int) -> None:
    """Add to the per-trace completion-token counter (called from llm.chat)."""
    _run_counter.completion_tokens = getattr(_run_counter, "completion_tokens", 0) + n


def get_run_completion_tokens() -> int:
    """Return the per-trace completion-token total for the current thread."""
    return getattr(_run_counter, "completion_tokens", 0)


def set_hard_cap(usd: float | None) -> None:
    """Set a cumulative-USD ceiling. Crossing it makes :func:`record` raise
    :class:`BudgetExceeded`. Pass ``None`` to disable."""
    with _state.lock:
        _state.hard_cap_usd = usd


def record(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Add a single API call to the running totals. Returns the USD delta."""
    # Always update per-trace completion-token counter, even for unknown models.
    add_run_completion_tokens(completion_tokens)

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
            return 0.0

        prev = _state.cost_usd
        _state.cost_usd += delta
        new = _state.cost_usd
        for thr in WARNING_THRESHOLDS_USD:
            if prev < thr <= new and thr not in _state.fired:
                _state.fired.add(thr)
                print(f"[cost_tracker] WARNING: cumulative spend crossed "
                      f"${thr:.0f} (now ${new:.4f}; "
                      f"in={_state.in_tokens:,} out={_state.out_tokens:,})")

        cap = _state.hard_cap_usd
        if cap is not None and new >= cap:
            raise BudgetExceeded(
                f"cumulative spend ${new:.4f} reached hard cap ${cap:.2f} "
                f"(in={_state.in_tokens:,} out={_state.out_tokens:,})"
            )

    return delta


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
