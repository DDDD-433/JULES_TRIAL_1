import json
import logging
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

try:
    import google.generativeai as genai

    _GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore[assignment]
    _GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

ENHANCED_SYSTEM_PROMPT = """You are an elite Adaptive Card designer creating STUNNING, professional Microsoft-quality cards.

🎯 CRITICAL REQUIREMENTS:

1. **STRUCTURED IMAGE REQUIREMENTS** (MANDATORY):
   Return image_queries as an array of objects:
   ```json
   "image_queries": [
     {"query": "professional business dashboard analytics modern", "purpose": "hero"},
     {"query": "team collaboration meeting office", "purpose": "content"},
     {"query": "revenue chart icon", "purpose": "icon"}
   ]
   ```

2. **Image Placeholders**:
   - Use: `"url": "https://placeholder.com/hero.jpg"` for ALL images
   - NEVER use adaptivecards.io URLs
   - Images will be auto-replaced with real URLs

3. **CRITICAL SCHEMA RULES**:
   ❌ NEVER put Actions inside Container items
   ❌ NEVER put Actions inside ColumnSets
   ✅ ALWAYS put Actions at card root level only
   ✅ OR inside Action.ShowCard cards only
   
4. **Valid Action Placement**:
   ```json
   {
     "card": {
       "body": [...],
       "actions": [
         {"type": "Action.OpenUrl", "title": "View Report", "url": "https://example.com"},
         {"type": "Action.Submit", "title": "Submit", "data": {"action": "submit"}}
       ]
     }
   }
   ```

5. **VISUAL HIERARCHY**:
   - Hero image (Large) at top
   - Emphasis container for title
   - ColumnSets for side-by-side content
   - FactSets for data
   - Separators between sections

6. **EXAMPLE: Dashboard Card (FOLLOW THIS STRUCTURE)**:
```json
{
  "card": {
    "$schema": "https://adaptivecards.io/schemas/adaptive-card.json",
    "type": "AdaptiveCard",
    "version": "1.5",
    "body": [
      {
        "type": "Image",
        "url": "https://placeholder.com/hero.jpg",
        "size": "Large",
        "horizontalAlignment": "Center",
        "altText": "Dashboard overview"
      },
      {
        "type": "Container",
        "style": "emphasis",
        "items": [
          {
            "type": "TextBlock",
            "text": "Q3 Performance Dashboard",
            "size": "ExtraLarge",
            "weight": "Bolder",
            "color": "Accent",
            "wrap": true
          },
          {
            "type": "TextBlock",
            "text": "Real-time insights and key metrics",
            "isSubtle": true,
            "wrap": true,
            "spacing": "None"
          }
        ]
      },
      {
        "type": "Container",
        "separator": true,
        "spacing": "Large",
        "items": [
          {
            "type": "ColumnSet",
            "columns": [
              {
                "type": "Column",
                "width": "auto",
                "items": [
                  {
                    "type": "Image",
                    "url": "https://placeholder.com/icon1.jpg",
                    "size": "Small",
                    "altText": "Revenue icon"
                  }
                ]
              },
              {
                "type": "Column",
                "width": "stretch",
                "items": [
                  {
                    "type": "TextBlock",
                    "text": "Revenue Growth",
                    "size": "Large",
                    "weight": "Bolder",
                    "wrap": true
                  },
                  {
                    "type": "FactSet",
                    "facts": [
                      {"title": "Current Quarter", "value": "$2.4M"},
                      {"title": "Growth Rate", "value": "+18.5%"},
                      {"title": "vs Target", "value": "112%"}
                    ]
                  }
                ]
              }
            ]
          }
        ]
      },
      {
        "type": "Container",
        "spacing": "Large",
        "items": [
          {
            "type": "ColumnSet",
            "columns": [
              {
                "type": "Column",
                "width": "stretch",
                "items": [
                  {
                    "type": "Image",
                    "url": "https://placeholder.com/content1.jpg",
                    "size": "Medium",
                    "altText": "Team performance"
                  },
                  {
                    "type": "TextBlock",
                    "text": "Top Performing Team",
                    "weight": "Bolder",
                    "spacing": "Small",
                    "wrap": true
                  }
                ]
              },
              {
                "type": "Column",
                "width": "stretch",
                "items": [
                  {
                    "type": "Image",
                    "url": "https://placeholder.com/content2.jpg",
                    "size": "Medium",
                    "altText": "Customer satisfaction"
                  },
                  {
                    "type": "TextBlock",
                    "text": "Customer Success",
                    "weight": "Bolder",
                    "spacing": "Small",
                    "wrap": true
                  }
                ]
              }
            ]
          }
        ]
      }
    ],
    "actions": [
      {
        "type": "Action.OpenUrl",
        "title": "View Full Report",
        "url": "https://example.com/report",
        "style": "positive"
      },
      {
        "type": "Action.Submit",
        "title": "Export Data",
        "data": {"action": "export"}
      }
    ]
  },
  "data": {},
  "image_queries": [
    {"query": "modern business analytics dashboard charts blue professional", "purpose": "hero"},
    {"query": "corporate team collaboration meeting", "purpose": "content"},
    {"query": "customer satisfaction business", "purpose": "content"},
    {"query": "revenue growth icon chart", "purpose": "icon"}
  ],
  "warnings": []
}
```

⚠️ **CRITICAL RULES - MUST FOLLOW**:
1. NEVER put Actions inside Container/ColumnSet items
2. ALL Actions go in card.actions array at root level
3. Use placeholder.com URLs for images (will be auto-replaced)
4. Always include 4-7 image_queries with purposes
5. Always set wrap: true for TextBlocks
6. Use FactSets for data, not tables
7. Return ONLY valid JSON

Generate a PRODUCTION-READY card following these rules exactly."""


def auto_detect_tone(brief: str) -> str:
    lowered = brief.lower()
    tone_keywords = {
        "Professional": [
            "business",
            "corporate",
            "enterprise",
            "professional",
            "executive",
            "formal",
            "report",
            "analytics",
            "dashboard",
            "metrics",
            "performance",
        ],
        "Friendly": ["welcome", "hello", "hi", "greet", "casual", "friendly", "community"],
        "Exciting": ["amazing", "exciting", "launch", "announcement", "celebrate", "breakthrough"],
        "Playful": ["fun", "game", "playful", "enjoy", "family", "kid", "adventure"],
        "Confident": ["achieve", "milestone", "growth", "confident", "leader", "success"],
        "Reassuring": ["safe", "health", "medical", "doctor", "care", "support", "wellness"],
        "Energetic": ["workshop", "event", "conference", "training", "dynamic", "interactive"],
        "Urgent": ["alert", "critical", "urgent", "immediate", "warning", "emergency"],
    }
    scores = {tone: 0 for tone in tone_keywords}
    for tone, keywords in tone_keywords.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[tone] += 3 if " " in keyword else 1
    detected = max(scores, key=scores.get)
    if scores[detected] == 0:
        detected = "Professional"
    return detected


def auto_detect_theme(brief: str) -> str:
    lowered = brief.lower()
    theme_keywords = {
        "enterprise": ["dashboard", "analytics", "report", "metrics", "business", "corporate"],
        "magazine": ["article", "news", "blog", "story", "feature", "showcase", "gallery"],
        "clean": ["simple", "minimal", "clean", "basic", "form", "registration"],
        "compact": ["quick", "summary", "brief", "compact", "notification", "alert"],
    }
    scores = {theme: 0 for theme in theme_keywords}
    for theme, keywords in theme_keywords.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[theme] += 1
    detected = max(scores, key=scores.get)
    if scores[detected] == 0:
        detected = "magazine" if len(brief.split()) > 15 else "clean"
    return detected


def _resolve_gemini_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if key:
        return key
    alt = os.getenv("GOOGLE_GEMINI_API_KEY", "").strip()
    if alt:
        os.environ["GOOGLE_API_KEY"] = alt
        return alt
    raise RuntimeError("Missing GOOGLE_API_KEY or GOOGLE_GEMINI_API_KEY for Gemini.")


def _get_gemini_model():
    if not _GENAI_AVAILABLE:
        raise RuntimeError(
            "google-generativeai is not installed. Add it to your environment to enable Gemini."
        )
    api_key = _resolve_gemini_api_key()
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    except ValueError:
        temperature = 0.7
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
    }
    return genai.GenerativeModel(model_name=model_name, generation_config=generation_config)


def _clean_markdown_json(raw: str) -> str:
    content = raw.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def _extract_gemini_text(response: Any) -> str:
    if response is None:
        raise RuntimeError("Gemini returned an empty response.")
    if isinstance(response, str):
        return response
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        parts = []
        if content is not None:
            parts = getattr(content, "parts", []) or []
        elif hasattr(candidate, "parts"):
            parts = getattr(candidate, "parts") or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                return part_text
    raise RuntimeError("Gemini response did not contain text content.")


def generate_card_structured(brief: str, tone: str) -> Dict[str, Any]:
    model = _get_gemini_model()
    form_keywords = [
        "form",
        "register",
        "registration",
        "submit",
        "input",
        "sign up",
        "signup",
        "enroll",
        "apply",
        "application",
        "survey",
        "feedback",
    ]
    needs_inputs = any(keyword in brief.lower() for keyword in form_keywords)
    input_reminder = ""
    if needs_inputs:
        input_reminder = "\n\n🔥 FORM DETECTED: Include Input.* elements and a Submit action!"

    prompt_body = f"""Brief: {brief}
Tone: {tone}{input_reminder}

Create a stunning, Microsoft-quality Adaptive Card following ALL requirements above. Focus on:
1. Exceptional visual hierarchy
2. 5-7 detailed image queries with purpose tags (hero/content/icon)
3. Professional interactivity (2-4 actions)
4. Real, specific content

Return ONLY valid JSON."""

    prompt = f"{ENHANCED_SYSTEM_PROMPT}\n\n{prompt_body}"

    response = model.generate_content(prompt)
    content = _clean_markdown_json(_extract_gemini_text(response))
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Gemini response: %s", exc)
        raise RuntimeError("Adaptive card generation failed to parse JSON.") from exc
    return parsed


def ensure_card_core_fields(card: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = deepcopy(card)
    cleaned["$schema"] = "https://adaptivecards.io/schemas/adaptive-card.json"
    cleaned["type"] = "AdaptiveCard"
    cleaned.setdefault("version", "1.5")
    cleaned.setdefault("body", [])
    return cleaned


def google_image_search_enhanced(
    queries: List[Dict[str, str]],
    max_per_purpose: Optional[Dict[str, int]] = None,
) -> List[Dict[str, str]]:

    def _env_value(*names: str) -> str:
        for candidate in names:
            value = os.getenv(candidate, "").strip()
            if value:
                return value
        return ""

    if max_per_purpose is None:
        max_per_purpose = {"hero": 2, "content": 4, "icon": 3}
    key = _env_value("GOOGLE_CSE_KEY", "GOOGLE_CSE_API_KEY", "GOOGLE_CUSTOM_SEARCH_API_KEY")
    cx = _env_value("GOOGLE_CSE_CX", "GOOGLE_CSE_ENGINE_ID", "GOOGLE_CUSTOM_SEARCH_CX")
    if not key or not cx:
        logger.warning("Google CSE credentials not configured; skipping image enrichment.")
        return []

    results: List[Dict[str, str]] = []
    seen: set = set()
    counts = {"hero": 0, "content": 0, "icon": 0}

    for item in queries:
        query = item.get("query", "").strip()
        purpose = (item.get("purpose") or "content").lower()
        if not query:
            continue
        if counts.get(purpose, 0) >= max_per_purpose.get(purpose, 0):
            continue

        img_size = "large" if purpose == "hero" else "medium" if purpose == "content" else "icon"
        try:
            resp = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": key,
                    "cx": cx,
                    "searchType": "image",
                    "q": query,
                    "safe": "active",
                    "num": 6,
                    "imgSize": img_size,
                    "imgType": "photo" if purpose in {"hero", "content"} else "clipart",
                },
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            for img in payload.get("items", []):
                link = img.get("link")
                if (
                    link
                    and link.lower().startswith("http")
                    and link.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                    and link not in seen
                ):
                    results.append({"url": link, "context": query, "purpose": purpose})
                    seen.add(link)
                    counts[purpose] += 1
                    if counts[purpose] >= max_per_purpose.get(purpose, 0):
                        break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Image search failed for '%s': %s", query, exc)

    return results


def _get_image_by_purpose(
    images: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    heroes = [img for img in images if img.get("purpose") == "hero"]
    content = [img for img in images if img.get("purpose") == "content"]
    icons = [img for img in images if img.get("purpose") == "icon"]
    if not heroes and not content and not icons and images:
        heroes = images[:2]
        content = images[2:6]
        icons = images[6:]
    return heroes, content, icons


def replace_placeholder_images(card: Dict[str, Any], images: List[Dict[str, str]]) -> Dict[str, Any]:
    if not images:
        return card

    heroes, content_imgs, icon_imgs = _get_image_by_purpose(images)
    hero_idx = content_idx = icon_idx = 0

    def pick_image(size: Optional[str]) -> Optional[Dict[str, str]]:
        nonlocal hero_idx, content_idx, icon_idx
        size = (size or "").lower()
        if size in {"large"} and hero_idx < len(heroes):
            img = heroes[hero_idx]
            hero_idx += 1
            return img
        if size in {"medium"} and content_idx < len(content_imgs):
            img = content_imgs[content_idx]
            content_idx += 1
            return img
        if size in {"small"} and icon_idx < len(icon_imgs):
            img = icon_imgs[icon_idx]
            icon_idx += 1
            return img
        pools = [content_imgs, heroes, icon_imgs]
        indices = [content_idx, hero_idx, icon_idx]
        for pool, idx_name in zip(pools, ["content", "hero", "icon"]):
            if idx_name == "content" and content_idx < len(content_imgs):
                img = content_imgs[content_idx]
                content_idx += 1
                return img
            if idx_name == "hero" and hero_idx < len(heroes):
                img = heroes[hero_idx]
                hero_idx += 1
                return img
            if idx_name == "icon" and icon_idx < len(icon_imgs):
                img = icon_imgs[icon_idx]
                icon_idx += 1
                return img
        return None

    def traverse(node: Any) -> Any:
        if isinstance(node, dict):
            if node.get("type") == "Image" and node.get("url"):
                url = node["url"]
                is_placeholder = (
                    "placeholder" in url
                    or "adaptivecards.io" in url
                    or not url.lower().startswith("http")
                )
                if is_placeholder:
                    selected = pick_image(node.get("size"))
                    if selected:
                        node["url"] = selected["url"]
                        node.setdefault("altText", selected.get("context") or "Image")
            for key, value in list(node.items()):
                node[key] = traverse(value)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                node[idx] = traverse(item)
        return node

    replaced = traverse(deepcopy(card))
    if replaced.get("body") and images and hero_idx == 0 and content_idx == 0 and icon_idx == 0:
        hero = images[0]
        replaced["body"].insert(
            0,
            {
                "type": "Image",
                "url": hero["url"],
                "size": "Large",
                "horizontalAlignment": "Center",
                "altText": hero.get("context") or "Hero image",
            },
        )
    return replaced


def log_card_event(sender_id: str, brief: str, card_title: Optional[str], host_theme: str) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": "adaptive_card_generated",
        "sender_id": sender_id,
        "brief": brief[:500],
        "card_title": card_title,
        "host_theme": host_theme,
    }
    try:
        with open("chatbot.log", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to log adaptive card event: %s", exc)


class ActionGenerateAdaptiveCard(Action):
    def name(self) -> str:
        return "action_generate_adaptive_card"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        brief = tracker.latest_message.get("text", "").strip()
        if not brief:
            dispatcher.utter_message(text="I need a description of the card you want me to design.")
            return []

        try:
            tone = auto_detect_tone(brief)
            host_theme = auto_detect_theme(brief)
            generation = generate_card_structured(brief, tone)
            card = ensure_card_core_fields(generation.get("card", {}))
            data = generation.get("data") or {}
            warnings = generation.get("warnings") or []
            image_queries = generation.get("image_queries") or []
            images = google_image_search_enhanced(image_queries)
            enriched_card = replace_placeholder_images(card, images)

            payload = {
                "card": enriched_card,
                "data": data,
                "hostTheme": host_theme,
                "warnings": warnings,
                "resolvedImages": images,
            }

            dispatcher.utter_message(json_message={"payload": "adaptiveCard", "data": payload})
            if warnings:
                dispatcher.utter_message(text="Heads-up: " + " | ".join(warnings[:2]))

            log_card_event(
                sender_id=tracker.sender_id,
                brief=brief,
                card_title=data.get("title") if isinstance(data, dict) else None,
                host_theme=host_theme,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Adaptive card generation failed: %s", exc)
            dispatcher.utter_message(
                text="I wasn't able to generate the adaptive card right now. Please double-check that the Gemini API key and Google Custom Search credentials are configured."
            )
        return []
