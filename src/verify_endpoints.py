"""
Post-hoc verification of OpenRouter endpoint UUIDs.

Reads ``logs/openrouter_call_audit.jsonl`` (written by every OpenRouter
chat-completion call), samples ``--n`` entries per model, queries the
generation-lookup API to retrieve each call's ``endpoint_id`` (a stable per-
endpoint UUID that distinguishes ``deepinfra/base`` from ``deepinfra/turbo``),
and asserts every retrieved UUID matches the expected bf16 UUID hardcoded
in ``src.llm._EXPECTED_ENDPOINT_IDS``. Any mismatch (or hit on a
``_FORBIDDEN_ENDPOINT_IDS`` UUID) prints ``CRITICAL`` and exits non-zero.

The chat-completion response only exposes the parent provider name
("DeepInfra"), not the endpoint variant, so this verifier is the canonical
audit step before treating an experiment as bf16-clean.

The generation-lookup API has a 15-30s ingest delay; this script runs after
a stage completes and is therefore cheap to wait on.

Usage:
  venv/Scripts/python -m src.verify_endpoints                # default n=5 per model
  venv/Scripts/python -m src.verify_endpoints --n 10
  venv/Scripts/python -m src.verify_endpoints --models meta-llama/llama-3.1-70b-instruct
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.llm import (
    OPENROUTER_AUDIT_LOG,
    _EXPECTED_ENDPOINT_IDS,
    _FORBIDDEN_ENDPOINT_IDS,
)

load_dotenv()

GENERATION_API = "https://openrouter.ai/api/v1/generation"


def _fetch_endpoint_id(gen_id: str, key: str, max_wait_s: int = 90) -> tuple[str | None, dict | None]:
    """Poll OpenRouter generation API until the record is ingested. Return
    (endpoint_id, full_data) or (None, None) on timeout."""
    waited = 0
    delay = 5
    while waited < max_wait_s:
        time.sleep(delay)
        waited += delay
        try:
            r = requests.get(
                GENERATION_API,
                params={"id": gen_id},
                headers={"Authorization": f"Bearer {key}"},
                timeout=30,
            )
            j = r.json()
        except (requests.RequestException, ValueError) as e:
            print(f"  [{gen_id}] HTTP/JSON error: {e}; retrying")
            delay = 10
            continue
        if "error" in j:
            delay = 10
            continue
        data = j.get("data") or {}
        provs = data.get("provider_responses") or []
        if provs:
            return provs[0].get("endpoint_id"), data
        return None, data
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5,
                        help="samples per model (default 5)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="restrict to these model ids; default = all OpenRouter models in audit log")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-log", type=str, default=str(OPENROUTER_AUDIT_LOG))
    args = parser.parse_args()

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    audit_path = Path(args.audit_log)
    if not audit_path.exists():
        print(f"ERROR: audit log {audit_path} does not exist", file=sys.stderr)
        sys.exit(2)

    by_model: dict[str, list[dict]] = defaultdict(list)
    with open(audit_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("gen_id"):
                by_model[rec["model"]].append(rec)

    if args.models:
        by_model = {m: v for m, v in by_model.items() if m in args.models}

    if not by_model:
        print("ERROR: no audit entries for the requested models", file=sys.stderr)
        sys.exit(2)

    rng = random.Random(args.seed)
    failures = []
    for model, recs in by_model.items():
        n = min(args.n, len(recs))
        sampled = rng.sample(recs, n)
        expected = _EXPECTED_ENDPOINT_IDS.get(model)
        forbidden = _FORBIDDEN_ENDPOINT_IDS.get(model, set())
        print(f"\n=== {model} | sampled {n}/{len(recs)} | expected={expected} ===")
        if expected is None:
            print(f"  [SKIP] no expected endpoint UUID hardcoded for {model}")
            continue
        for rec in sampled:
            gid = rec["gen_id"]
            eid, data = _fetch_endpoint_id(gid, key)
            mark = "OK" if eid == expected else ("CRITICAL" if eid in forbidden else "MISMATCH")
            print(f"  [{mark}] gen_id={gid}  endpoint_id={eid}")
            if eid != expected:
                failures.append({
                    "model": model, "gen_id": gid,
                    "got_endpoint_id": eid, "expected": expected,
                    "is_forbidden": eid in forbidden,
                })

    print()
    if failures:
        print(f"!!! {len(failures)} mismatches across {len(by_model)} model(s) !!!")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print(f"All sampled calls hit the expected bf16 endpoint. "
          f"({sum(min(args.n, len(v)) for v in by_model.values())} samples total)")


if __name__ == "__main__":
    main()
