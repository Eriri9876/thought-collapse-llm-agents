import time
import requests

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "thought-collapse-research/1.0 (academic research)"}


def _get(params: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=15)
            return r.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(2)
    raise ConnectionError("Wikipedia API unreachable after retries.")


def _resolve_title(query: str) -> str | None:
    data = _get({"action": "query", "format": "json", "list": "search", "srsearch": query, "srlimit": 1})
    results = data["query"]["search"]
    return results[0]["title"] if results else None


def search_wikipedia(query: str, sentences: int = 3) -> str:
    try:
        title = _resolve_title(query)
        if title is None:
            return f"No Wikipedia article found for '{query}'."

        data = _get({
            "action": "query", "format": "json", "prop": "extracts",
            "exintro": True, "explaintext": True, "exsentences": sentences,
            "redirects": True, "titles": title,
        })
        page = next(iter(data["query"]["pages"].values()))
        if "missing" in page:
            return f"No Wikipedia article found for '{query}'."
        return f"[{title}] {page['extract'].strip()}"
    except Exception as e:
        return f"Search failed: {e}"


def calculate(expression: str) -> str:
    try:
        allowed = {k: v for k, v in vars(__builtins__).items()
                   if k in ("abs", "round", "int", "float", "max", "min", "sum", "pow")} \
                  if isinstance(vars(__builtins__), dict) else \
                  {k: getattr(__builtins__, k) for k in
                   ("abs", "round", "int", "float", "max", "min", "sum", "pow")
                   if hasattr(__builtins__, k)}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


TOOLS = {
    "search": search_wikipedia,
    "calculate": calculate,
}
