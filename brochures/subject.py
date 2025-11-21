import re
from typing import Optional

TRIM_RE = re.compile(r"\s+")

PHRASES = [
    "create a brochure about",
    "create a brochure on",
    "create a brocure on",
    "create a brcohure on",
    "create brochure on",
    "make a brochure about",
    "make a brochure on",
    "plan a brochure about",
    "brochure about",
    "brochure on",
    "about",
    "on",
]

STOP_WORDS = {"with", "featuring", "including", "using", "for", "around", "at", "in"}


def derive_subject(user_prompt: str, fallback: str = "Cultural Showcase") -> str:
    text = (user_prompt or "").strip()
    if not text:
        return fallback

    lowered = text.lower()
    subject: Optional[str] = None
    for phrase in PHRASES:
        idx = lowered.find(phrase)
        if idx != -1:
            subject = text[idx + len(phrase):].strip()
            break

    if not subject:
        subject = text

    subject = subject.split("\n", 1)[0]
    subject = re.split(r"[.?!]", subject)[0]

    tokens = subject.split()
    clean_tokens = []
    for token in tokens:
        if clean_tokens and token.lower() in STOP_WORDS:
            break
        clean_tokens.append(token)
    subject = " ".join(clean_tokens).strip(" ,;:-")

    if not subject:
        return fallback

    return subject.title()
