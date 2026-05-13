import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError
from dotenv import load_dotenv

from src import cache, cost_tracker

load_dotenv()

_clients = {}
_or_first_call_logged: set[str] = set()  # models we've already printed actual-provider for


# OpenRouter provider-pinning config per Llama-3.1 model.
# Verified 2026-05-09 against https://openrouter.ai/api/v1/models/.../endpoints :
#   8B  has only one DeepInfra endpoint, slug "deepinfra/bf16".
#   70B has two DeepInfra endpoints — bf16 is "deepinfra/base", fp8 is
#   "deepinfra/turbo". We pin to base, ignore turbo, and require bf16
#   quantization (belt + suspenders) so a future endpoint rename can't silently
#   route to fp8.
_OPENROUTER_PROVIDER_CONFIG: dict[str, dict] = {
    "meta-llama/llama-3.1-8b-instruct": {
        "order": ["deepinfra/bf16"],
        "allow_fallbacks": False,
        "quantizations": ["bf16"],
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "order": ["deepinfra/base"],
        "allow_fallbacks": False,
        "quantizations": ["bf16"],
        "ignore": ["deepinfra/turbo"],
    },
}

# Endpoint UUIDs captured from a control routing test on 2026-05-10. The
# chat-completion response only exposes the parent provider name
# ("DeepInfra"), so substring checks on the response cannot distinguish bf16
# from fp8/turbo — *but* OpenRouter's generation-lookup API
# (/api/v1/generation?id=<gen_id>) returns a stable per-endpoint UUID
# (provider_responses[0].endpoint_id). The control test forced
# order=["deepinfra/turbo"] vs the normal bf16 config and confirmed the UUIDs
# differ. Stage-1 post-hoc verification (src/verify_endpoints.py) samples
# audit-log gen_ids and asserts the endpoint UUID equals the expected bf16
# UUID below; mismatches abort.
_EXPECTED_ENDPOINT_IDS: dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct":  "858e9b98-fa86-433e-8299-17c3c4d6c24f",  # deepinfra/bf16
    "meta-llama/llama-3.1-70b-instruct": "59c87462-40b6-4231-91cf-6d0f8f25e8b9",  # deepinfra/base (bf16)
}
_FORBIDDEN_ENDPOINT_IDS: dict[str, set[str]] = {
    "meta-llama/llama-3.1-70b-instruct": {
        "036dfa0a-ebbb-4a51-aa90-091a75c2cadb",  # deepinfra/turbo (fp8)
    },
}

OPENROUTER_AUDIT_LOG = Path("logs/openrouter_call_audit.jsonl")


def _backend(model: str) -> str:
    if model.startswith("Qwen/") or model.startswith("qwen"):
        return "siliconflow"
    if model.startswith("meta-llama/"):
        return "openrouter"
    return "deepseek"


def _get_client(model: str) -> OpenAI:
    backend = _backend(model)
    if backend not in _clients:
        if backend == "siliconflow":
            _clients[backend] = OpenAI(
                api_key=os.getenv("SILICONFLOW_API_KEY"),
                base_url="https://api.siliconflow.cn/v1",
            )
        elif backend == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY not set in environment / .env"
                )
            _clients[backend] = OpenAI(
                api_key=key,
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            _clients[backend] = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
    return _clients[backend]


def _extract_response_provider(response) -> str | None:
    """OpenRouter attaches the resolved provider name (parent slug, e.g.
    ``"DeepInfra"`` — *not* the endpoint variant) to the response. The OpenAI
    SDK keeps unknown fields under ``model_extra`` (Pydantic). Return as
    lowercase string or ``None``."""
    extra = getattr(response, "model_extra", None) or {}
    prov = extra.get("provider")
    if isinstance(prov, str) and prov:
        return prov.lower()
    if isinstance(prov, dict):
        for k in ("slug", "name", "id"):
            v = prov.get(k)
            if isinstance(v, str) and v:
                return v.lower()
    direct = getattr(response, "provider", None)
    if isinstance(direct, str) and direct:
        return direct.lower()
    return None


def _audit_openrouter_call(model: str, response) -> None:
    """Append one line to logs/openrouter_call_audit.jsonl for every successful
    OpenRouter call. The endpoint UUID is *not* in the chat-completion response
    (only in the generation-lookup API, which has 15-30s ingest delay), so
    this audit only records what the SDK already exposed: gen_id, parent
    provider name, token counts, OpenRouter-side cost. Post-hoc, run
    ``src.verify_endpoints`` on the audit log to sample N gen_ids, fetch the
    generation API, and assert each ``endpoint_id`` matches the expected
    bf16 UUID hardcoded in ``_EXPECTED_ENDPOINT_IDS``."""
    parent = _extract_response_provider(response)
    if model not in _or_first_call_logged:
        expected = (_OPENROUTER_PROVIDER_CONFIG.get(model, {}).get("order") or [""])[0]
        print(f"[llm] OpenRouter first call for {model}: parent provider="
              f"{parent!r}, requested endpoint slug={expected!r}. "
              f"Endpoint UUID will be verified post-hoc via "
              f"src.verify_endpoints against the audit log.")
        _or_first_call_logged.add(model)

    usage = getattr(response, "usage", None)
    in_tok = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
    out_tok = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
    or_cost = None
    if usage is not None:
        d = usage.model_dump() if hasattr(usage, "model_dump") else {}
        or_cost = d.get("cost")

    record = {
        "ts":          datetime.now(timezone.utc).isoformat(),
        "model":       model,
        "gen_id":      getattr(response, "id", None),
        "provider":    parent,
        "in_tok":      in_tok,
        "out_tok":     out_tok,
        "or_cost_usd": or_cost,
    }
    try:
        OPENROUTER_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(OPENROUTER_AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        # audit failure must never kill the experiment
        print(f"[llm] WARNING: could not write audit log: {e}")


def chat(
    messages: list[dict],
    model: str = "deepseek-chat",
    temperature: float = 0.0,
    timeout: float = 240.0,
    *,
    use_cache: bool = True,
    seed: int | None = None,
    max_tokens: int | None = None,
) -> str:
    """OpenAI-compatible chat call with disk cache + USD cost tracking.

    Cache hits skip the API call and the cost tracker entirely. Set
    ``use_cache=False`` to force a fresh call (success still writes the
    response back to cache for future hits). ``seed`` is folded into the
    cache key so seeded reruns don't share entries with unseeded ones.

    For OpenRouter (``meta-llama/`` models), the provider config is also
    folded into the cache key so a bf16 cache entry can never satisfy a
    request that should hit fp8 (or vice versa).
    """
    backend = _backend(model)
    extra_body: dict | None = None
    if backend == "openrouter":
        prov = _OPENROUTER_PROVIDER_CONFIG.get(model)
        if prov is None:
            raise RuntimeError(
                f"No OpenRouter provider config for {model!r}; refusing to "
                f"call without an explicit bf16 pin."
            )
        extra_body = {"provider": prov}

    cache_key = cache.make_key(
        model, messages, temperature=temperature, seed=seed,
        extra={"max_tokens": max_tokens, "extra_body": extra_body},
    )
    if use_cache:
        cached = cache.get_cache().get(cache_key)
        if cached is not None:
            return cached

    client = _get_client(model)
    create_kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
    }
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens
    if extra_body is not None:
        create_kwargs["extra_body"] = extra_body

    max_retries = 7
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(**create_kwargs)
            if backend == "openrouter":
                _audit_openrouter_call(model, response)
            content = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            if usage is not None:
                cost_tracker.record(
                    model,
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0,
                )
            if content:
                cache.get_cache().set(cache_key, content)
            return content
        except (APITimeoutError, APIConnectionError, APIStatusError) as e:
            if attempt == max_retries:
                raise
            wait = min(300, 15 * (2 ** attempt))
            print(f"    [retry {attempt+1}/{max_retries} after {wait}s: {type(e).__name__}]")
            time.sleep(wait)
