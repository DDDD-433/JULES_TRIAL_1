from __future__ import annotations

import re
from typing import Dict, List

from shared_utils import MessageFormatter

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def clean_sentence(text: str) -> str:
    if not text:
        return ""
    cleaned = MessageFormatter.clean_markdown_text(text).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def ensure_sentence(text: str, fallback: str, min_words: int) -> str:
    candidate = MessageFormatter.clean_markdown_text(text or "").strip()
    if candidate.lower() == fallback.lower() or len(candidate.split()) < min_words:
        candidate = fallback
    return clean_sentence(candidate)


def ensure_paragraph(text: str, fallback: str, min_words: int) -> str:
    candidate = MessageFormatter.clean_markdown_text(text or "").strip()
    if len(candidate.split()) < min_words or candidate.lower() == fallback.lower():
        candidate = fallback
    sentences = SENTENCE_END.split(candidate)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [fallback]
    if len(sentences) == 1:
        sentences.append(fallback)
    paragraph = ". ".join(sentences)
    return clean_sentence(paragraph)


def sanitize_facts(facts: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        title = MessageFormatter.clean_markdown_text(fact.get("title", "")).strip()
        detail = MessageFormatter.clean_markdown_text(fact.get("detail") or fact.get("value", "")).strip()
        if title and detail:
            cleaned.append({"title": title[:60], "detail": clean_sentence(detail)[:240]})
        if len(cleaned) >= 5:
            break
    return cleaned
