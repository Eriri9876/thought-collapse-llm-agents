import os
import time
from openai import OpenAI, APITimeoutError, APIConnectionError
from dotenv import load_dotenv

from src import cache, cost_tracker

load_dotenv()

_clients = {}


def _get_client(model: str) -> OpenAI:
    if model.startswith("Qwen/") or model.startswith("qwen"):
        key = "siliconflow"
        if key not in _clients:
            _clients[key] = OpenAI(
                api_key=os.getenv("SILICONFLOW_API_KEY"),
                base_url="https://api.siliconflow.cn/v1",
            )
        return _clients[key]
    else:
        key = "deepseek"
        if key not in _clients:
            _clients[key] = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
        return _clients[key]


def chat(
    messages: list[dict],
    model: str = "deepseek-chat",
    temperature: float = 0.0,
    timeout: float = 240.0,
    *,
    use_cache: bool = True,
    seed: int | None = None,
) -> str:
    """OpenAI-compatible chat call with disk cache + USD cost tracking.

    Cache hits skip the API call and the cost tracker entirely. Set
    ``use_cache=False`` to force a fresh call (success still writes the
    response back to cache for future hits). ``seed`` is folded into the
    cache key so seeded reruns don't share entries with unseeded ones.
    """
    cache_key = cache.make_key(
        model, messages, temperature=temperature, seed=seed,
    )
    if use_cache:
        cached = cache.get_cache().get(cache_key)
        if cached is not None:
            return cached

    client = _get_client(model)
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
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
        except (APITimeoutError, APIConnectionError) as e:
            if attempt == 3:
                raise
            wait = 10 * (attempt + 1)
            print(f"    [retry {attempt+1}/3 after {wait}s: {type(e).__name__}]")
            time.sleep(wait)
