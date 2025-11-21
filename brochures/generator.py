import json
from dataclasses import dataclass
from typing import Any, Dict, List

from shared_utils import logger

from .prompts import build_messages
from .subject import derive_subject
from .validators import clean_sentence, ensure_paragraph, ensure_sentence, sanitize_facts


@dataclass
class BrochureGenerationRequest:
    user_prompt: str


@dataclass
class BrochureGenerationResult:
    subject: str
    title: str
    short_description: str
    detailed_description: str
    image_caption: str
    image_queries: List[str]
    facts: List[Dict[str, str]]
    more_info_title: str
    more_info_url: str
    raw_response: Dict[str, Any]


class UniversalBrochureGenerator:
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, llm_service) -> None:
        self.llm = llm_service

    def generate(self, request: BrochureGenerationRequest) -> BrochureGenerationResult:
        subject = derive_subject(request.user_prompt)
        messages = build_messages(subject, request.user_prompt)
        raw_text = self.llm.generate_text(
            messages=messages,
            model=self.MODEL,
            max_tokens=950,
            temperature=0.6,
            response_format={"type": "json_object"},
            timeout=45,
        )
        payload = self._parse_response(raw_text)
        validated = self._validate(payload, subject, request)
        return BrochureGenerationResult(**validated)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        try:
            return json.loads(response_text)
        except Exception as exc:
            logger.warning(f"Brochure JSON parse error: {exc}")
            raise

    def _validate(self, payload: Dict[str, Any], subject: str, request: BrochureGenerationRequest) -> Dict[str, Any]:
        title = payload.get("title") or subject
        short_description = payload.get("short_description") or subject
        detailed_description = payload.get("detailed_description") or short_description
        image_caption = payload.get("image_caption") or subject

        fallback_short = (
            f"Celebrate {subject} with a thoughtfully curated experience that brings people together."
        )
        short_description = ensure_sentence(short_description, fallback_short, min_words=8)

        fallback_detail = (
            f"{subject} welcomes guests with warmth and cultural richness. "
            "Share the story, highlights, and reasons this gathering matters to your audience."
        )
        detailed_description = ensure_paragraph(detailed_description, fallback_detail, min_words=16)

        image_caption = clean_sentence(image_caption) or f"{subject} cultural experience"

        image_queries = payload.get("image_queries")
        if not isinstance(image_queries, list):
            image_queries = []
        image_queries = [q.strip() for q in image_queries if isinstance(q, str) and q.strip()][:3]
        if not image_queries:
            image_queries = [subject, request.user_prompt.strip()[:80]]

        facts = payload.get("facts")
        if not isinstance(facts, list):
            facts = []
        facts = sanitize_facts(facts)
        if not facts:
            facts = [
                {
                    "title": "Highlight",
                    "detail": clean_sentence(
                        f"{subject} offers memorable experiences that connect people through culture"
                    ),
                }
            ]

        more_info = payload.get("more_info") or {}
        if not isinstance(more_info, dict):
            more_info = {}
        more_info_title = more_info.get("title") or "Learn More"
        more_info_url = more_info.get("url") or "https://www.moc.gov.sa"
        if not str(more_info_url).lower().startswith(("http://", "https://")):
            more_info_url = "https://www.moc.gov.sa"

        return {
            "subject": subject,
            "title": title[:120],
            "short_description": short_description[:200],
            "detailed_description": detailed_description[:500],
            "image_caption": image_caption[:100],
            "image_queries": image_queries,
            "facts": facts,
            "more_info_title": more_info_title[:50],
            "more_info_url": more_info_url,
            "raw_response": payload,
        }
