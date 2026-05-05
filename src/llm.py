import os
import time
from openai import OpenAI, APITimeoutError, APIConnectionError
from dotenv import load_dotenv

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


def chat(messages: list[dict], model: str = "deepseek-chat", temperature: float = 0.0, timeout: float = 240.0) -> str:
    client = _get_client(model)
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            return response.choices[0].message.content
        except (APITimeoutError, APIConnectionError) as e:
            if attempt == 3:
                raise
            wait = 10 * (attempt + 1)
            print(f"    [retry {attempt+1}/3 after {wait}s: {type(e).__name__}]")
            time.sleep(wait)
