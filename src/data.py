from datasets import load_dataset
import hashlib
import random
import re

_cache = {}


def _math_id(problem: str) -> str:
    """Stable cross-process ID for a MATH-hard problem.

    Earlier versions used ``f"math_{hash(problem) & 0xFFFFFF}"`` which
    relies on Python's string ``hash()`` — randomised per process unless
    ``PYTHONHASHSEED`` is fixed. That made cross-process joins (pageview
    extractor vs probe_direct logs) silently fail with 0 matched rows.
    SHA-256 is deterministic and yields the same ID in every Python run.
    """
    h = hashlib.sha256(problem.encode("utf-8")).hexdigest()
    return f"math_{h[:8]}"


def _load_hotpotqa():
    if "hotpotqa" not in _cache:
        _cache["hotpotqa"] = load_dataset("hotpot_qa", "fullwiki", split="validation")
    return _cache["hotpotqa"]


def _extract_boxed(solution: str) -> str | None:
    """Extract answer from LaTeX \\boxed{...} in MATH solutions."""
    m = re.search(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    return m.group(1).strip() if m else None


def _boxed_to_number(boxed: str) -> str | None:
    """Try to convert a boxed answer to a plain number string."""
    s = boxed.strip()
    # simple integer or decimal
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return s
    # negative with parens: (-3) -> -3
    m = re.fullmatch(r"\((-?\d+(\.\d+)?)\)", s)
    if m:
        return m.group(1)
    # fraction: \frac{a}{b} — keep as string "a/b"
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None


def get_samples(n: int, seed: int = 42, dataset: str = "hotpotqa", hotpotqa_type: str = None) -> list[dict]:
    if dataset == "hotpotqa":
        ds = _load_hotpotqa()
        indices = list(range(len(ds)))
        if hotpotqa_type:
            indices = [i for i in indices if ds[i]["type"] == hotpotqa_type]
        random.seed(seed)
        random.shuffle(indices)
        return [
            {
                "id": ds[i]["id"],
                "question": ds[i]["question"],
                "answer": ds[i]["answer"],
                "aliases": [ds[i]["answer"]],
            }
            for i in indices[:n]
        ]
    elif dataset == "webquestions":
        if "webquestions" not in _cache:
            _cache["webquestions"] = load_dataset("web_questions", split="test")
        ds = _cache["webquestions"]
        indices = list(range(len(ds)))
        random.seed(seed)
        random.shuffle(indices)
        return [
            {
                "id": ds[i]["url"],
                "question": ds[i]["question"],
                "answer": ds[i]["answers"][0],
                "aliases": [a.lower() for a in ds[i]["answers"]],
            }
            for i in indices[:n]
        ]
    elif dataset == "gsm8k":
        if "gsm8k" not in _cache:
            _cache["gsm8k"] = load_dataset("gsm8k", "main", split="test")
        ds = _cache["gsm8k"]
        # "hard" = solution requires >=5 reasoning lines before the #### answer
        hard_indices = [
            i for i in range(len(ds))
            if len([l for l in ds[i]["answer"].split("\n") if l.strip() and not l.startswith("####")]) >= 5
        ]
        random.seed(seed)
        random.shuffle(hard_indices)
        return [
            {
                "id": f"gsm8k_{i}",
                "question": ds[i]["question"],
                "answer": ds[i]["answer"].split("####")[-1].strip(),
                "aliases": [ds[i]["answer"].split("####")[-1].strip()],
            }
            for i in hard_indices[:n]
        ]
    elif dataset == "triviaqa":
        if "triviaqa" not in _cache:
            _cache["triviaqa"] = load_dataset(
                "trivia_qa", "rc.wikipedia", split="validation"
            )
        ds = _cache["triviaqa"]
        indices = list(range(len(ds)))
        random.seed(seed)
        random.shuffle(indices)
        samples = []
        for i in indices:
            if len(samples) >= n:
                break
            item = ds[i]
            answer_val = item["answer"]["value"]
            aliases = list({a.lower() for a in item["answer"].get("aliases", [])}
                           | {answer_val.lower()})
            samples.append({
                "id":       item["question_id"],
                "question": item["question"],
                "answer":   answer_val,
                "aliases":  aliases,
            })
        return samples

    elif dataset == "math_hard":
        # MATH Level 4-5, numeric-answer problems only (avoids LaTeX-heavy evaluation)
        if "math_hard" not in _cache:
            MATH_SUBSETS = [
                "algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
            ]
            raw_all = []
            for subset in MATH_SUBSETS:
                try:
                    raw_all.extend(load_dataset(
                        "EleutherAI/hendrycks_math", subset, split="test"
                    ))
                except Exception:
                    pass
            raw = raw_all
            hard = []
            for item in raw_all:
                if item["level"] not in ("Level 4", "Level 5"):
                    continue
                boxed = _extract_boxed(item["solution"])
                if boxed is None:
                    continue
                plain = _boxed_to_number(boxed)
                if plain is None:
                    continue
                hard.append({
                    "id":       _math_id(item["problem"]),
                    "question": item["problem"],
                    "answer":   plain,
                    "aliases":  [plain],
                    "type":     item["type"],
                })
            _cache["math_hard"] = hard
        hard = _cache["math_hard"]
        random.seed(seed)
        indices = list(range(len(hard)))
        random.shuffle(indices)
        return [hard[i] for i in indices[:n]]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
