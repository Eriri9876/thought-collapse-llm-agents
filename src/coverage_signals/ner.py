"""
Named-entity head extractor for question text.

Loads spaCy ``en_core_web_sm`` lazily on first use and exposes
``extract_head_entity(text)``, which returns the first salient named entity
in the text or ``None`` if none is found. Used by the popularity (pageview)
and n-gram (InfiniGram) signals to pick the entity to query.

Salient entity types (any of these counts; first hit wins, in document order):
    PERSON, ORG, GPE, LOC, FAC, WORK_OF_ART, EVENT, NORP, PRODUCT, LAW

Filters (applied to both NER hits and the noun-chunk fallback):
  - reject pure numbers (``^[\\d.,]+$``) — math word problems leak digits
    that resolve to spurious Wikipedia / corpus matches
  - reject token sequences shorter than 3 characters
  - reject spaCy-defined English stop words (``the``, ``a``, ``of``, …)

Run once before first use::

    python -m spacy download en_core_web_sm
"""
from __future__ import annotations

import re
from functools import lru_cache

import spacy

_SALIENT_TYPES = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "FAC",
    "WORK_OF_ART", "EVENT", "NORP", "PRODUCT", "LAW",
})

_PURE_NUMBER = re.compile(r"^[\d.,]+$")


@lru_cache(maxsize=1)
def _nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError as e:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Run:\n"
            "    python -m spacy download en_core_web_sm"
        ) from e


@lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    """spaCy's built-in English stop-word list, plus a couple of extras
    the math word problems lean on."""
    return frozenset(
        {*_nlp().Defaults.stop_words, "ms", "mr", "mrs", "dr"}
    )


def _is_garbage(text: str) -> bool:
    """Return True if `text` should be rejected as a useless NER hit."""
    if text is None:
        return True
    s = text.strip()
    if len(s) < 3:
        return True
    if _PURE_NUMBER.match(s):
        return True
    # All tokens are stop words (e.g. "the", "of the", "an") → reject
    tokens = s.lower().split()
    if tokens and all(t in _stop_words() for t in tokens):
        return True
    return False


def extract_head_entity(text: str) -> str | None:
    """Return first salient named entity that survives filters, else
    longest non-pronoun noun chunk that survives filters, else None."""
    doc = _nlp()(text)
    for ent in doc.ents:
        if ent.label_ not in _SALIENT_TYPES:
            continue
        s = ent.text.strip()
        if _is_garbage(s):
            continue
        return s
    # noun-chunk fallback (also filtered)
    chunks = [c for c in doc.noun_chunks
              if not any(t.pos_ == "PRON" for t in c)
              and not _is_garbage(c.text.strip())]
    if chunks:
        return max(chunks, key=lambda c: len(c.text)).text.strip()
    return None


def extract_all_entities(text: str) -> list[dict]:
    """Return every entity that survives filters, with type and char span."""
    doc = _nlp()(text)
    return [
        {"text": ent.text.strip(), "label": ent.label_,
         "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
        if not _is_garbage(ent.text.strip())
    ]


def extract_all_entities_filtered(text: str) -> list[str]:
    """Return ordered, deduped list of candidate entity strings.

    Selection mirrors :func:`extract_head_entity`'s precedence:
      1. salient NER entities (PERSON, ORG, GPE, …) that survive filters
      2. fallback to noun chunks (no pronouns, filtered) only if no
         salient entity survives

    Used by ``extract_pageview_min`` for the multi-entity pageview signal:
    head is ``cands[0]`` if salient hits exist; otherwise the longest
    chunk (matching legacy ``extract_head_entity`` semantics). The min /
    max pageview are computed across this whole list.
    """
    doc = _nlp()(text)
    out: list[str] = []
    seen: set[str] = set()

    for ent in doc.ents:
        if ent.label_ not in _SALIENT_TYPES:
            continue
        s = ent.text.strip()
        if _is_garbage(s):
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)

    if out:
        return out

    for c in doc.noun_chunks:
        if any(t.pos_ == "PRON" for t in c):
            continue
        s = c.text.strip()
        if _is_garbage(s):
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out
