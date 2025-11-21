import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa_sdk import Action, Tracker
from rasa_sdk.events import EventType, SlotSet
from rasa_sdk.executor import CollectingDispatcher

from shared_utils import MessageFormatter, get_service_manager, logger
from brochures.generator import UniversalBrochureGenerator, BrochureGenerationRequest, BrochureGenerationResult

SECURE_LOG_PATH = Path("secure_rag.log")
DEFAULT_IMAGE_URL = "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=1400&q=80"
DEFAULT_IMAGE_ALT = "Scenic cultural gathering"


def _recent_user_messages(tracker: Tracker, limit: int = 5) -> List[str]:
    collected: List[str] = []
    for event in reversed(tracker.events):
        if event.get("event") == "user":
            text = (event.get("text") or "").strip()
            if text:
                collected.append(text)
            if len(collected) >= limit:
                break
    return list(reversed(collected))


class ActionGenerateBrochure(Action):
    def __init__(self) -> None:
        self.services = get_service_manager()
        self.generator = UniversalBrochureGenerator(self.services.get_llm_service())

    def name(self) -> Text:
        return "action_generate_brochure"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        latest_text = tracker.latest_message.get("text", "").strip()
        if not latest_text:
            dispatcher.utter_message(
                text="I need a little more detail to craft the brochure. What should it cover?"
            )
            return []

        recent_messages = _recent_user_messages(tracker)
        if recent_messages:
            # Ensure latest request is clearly highlighted so LLM doesn't reuse older topics
            latest = recent_messages[-1]
            prior_requests = [msg for msg in recent_messages[:-1] if msg.strip()]
            if prior_requests:
                contextual_prompt = "Earlier user requests (for context only):\n"
                contextual_prompt += "\n".join(f"- {msg}" for msg in prior_requests)
                contextual_prompt += (
                    "\n\nIgnore earlier topics if they conflict with the latest instruction below."
                )
                combined_prompt = f"{contextual_prompt}\n\nLatest brochure request:\n{latest}"
            else:
                combined_prompt = latest
        else:
            combined_prompt = latest_text

        request = BrochureGenerationRequest(user_prompt=combined_prompt)

        try:
            result = self.generator.generate(request)
            image_url, image_alt, attribution = self._select_image(result)
            card_payload = self._build_adaptive_card(result, image_url, image_alt)
            summary_text = self._build_summary(result)

            dispatcher.utter_message(text=summary_text)
            dispatcher.utter_message(
                json_message={
                    "payload": "adaptiveCard",
                    "data": card_payload,
                    "meta": {
                        "template_id": "universal_card",
                        "language": "en",
                        "accent": "bespoke",
                        "image_attribution": attribution,
                    },
                }
            )

            plan_payload = {
                "subject": result.subject,
                "title": result.title,
                "short_description": result.short_description,
                "detailed_description": result.detailed_description,
                "image_caption": result.image_caption,
                "image_url": image_url,
                "facts": result.facts,
                "more_info_title": result.more_info_title,
                "more_info_url": result.more_info_url,
                "raw_response": result.raw_response,
            }
            self._log_generation(tracker, plan_payload)

            return [SlotSet("brochure_plan", json.dumps(plan_payload, ensure_ascii=False))]
        except Exception as exc:
            logger.error(f"Brochure generation failed: {exc}")
            dispatcher.utter_message(
                text=(
                    "I couldn’t complete that brochure draft right now. "
                    "Please try again with a little more guidance."
                )
            )
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_image(self, result) -> Tuple[str, str, Optional[str]]:
        image_service = None
        try:
            image_service = self.services.get_image_service()
        except Exception as exc:
            logger.warning(f"Unable to access image service: {exc}")

        if image_service and getattr(image_service, "is_enabled", lambda: False)():
            for query in result.image_queries:
                try:
                    search_result = image_service.fetch_image(query)
                except Exception as exc:
                    logger.warning(f"Image search failed for '{query}': {exc}")
                    continue
                if search_result and search_result.get("image_url"):
                    image_url = search_result.get("image_url")
                    alt_text = (
                        search_result.get("alt")
                        or search_result.get("attribution")
                        or result.image_caption
                    )
                    attribution = search_result.get("context_url") or search_result.get("attribution")
                    return image_url, alt_text, attribution

        return DEFAULT_IMAGE_URL, result.image_caption or DEFAULT_IMAGE_ALT, None

    def _build_adaptive_card(self, result, image_url: str, image_alt: str) -> Dict[str, Any]:
        title = MessageFormatter.clean_markdown_text(result.title)
        short_desc = MessageFormatter.clean_markdown_text(result.short_description)
        detailed_desc = MessageFormatter.clean_markdown_text(result.detailed_description)
        alt_text = MessageFormatter.clean_markdown_text(image_alt or result.image_caption)

        body: List[Dict[str, Any]] = [
            {
                "type": "Image",
                "url": image_url,
                "altText": alt_text,
            },
            {
                "type": "TextBlock",
                "text": title,
                "weight": "Bolder",
                "size": "Large",
                "wrap": True,
                "spacing": "Medium",
            },
            {
                "type": "TextBlock",
                "text": short_desc,
                "wrap": True,
                "spacing": "Medium",
            },
            {
                "type": "TextBlock",
                "text": detailed_desc,
                "wrap": True,
            },
        ]

        if result.facts:
            body.append(
                {
                    "type": "TextBlock",
                    "text": "Highlights",
                    "wrap": True,
                    "size": "Medium",
                    "weight": "Bolder",
                    "spacing": "Medium",
                }
            )
            body.append(
                {
                    "type": "FactSet",
                    "facts": [
                        {
                            "title": MessageFormatter.clean_markdown_text(fact["title"]),
                            "value": MessageFormatter.clean_markdown_text(fact["detail"]),
                        }
                        for fact in result.facts
                    ],
                }
            )

        more_info_title = MessageFormatter.clean_markdown_text(result.more_info_title or "Learn More")

        actions = [
            {
                "type": "Action.OpenUrl",
                "title": more_info_title or "Learn More",
                "url": result.more_info_url,
                "fallbackUrl": result.more_info_url,
                "tooltip": "Open related page in a new tab",
            }
        ]

        return {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.3",
            "body": body,
            "actions": actions,
        }

    def _build_summary(self, result) -> str:
        title = MessageFormatter.clean_markdown_text(result.title)
        summary = MessageFormatter.clean_markdown_text(result.short_description)

        sentences = [f"Here’s a brochure concept titled **{title}**."]
        if summary:
            sentences.append(summary)
        sentences.append("Feel free to adjust any details or ask for revisions.")
        return "\n\n".join(sentences)

    def _log_generation(self, tracker: Tracker, payload: Dict[str, Any]) -> None:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "brochure_generated",
            "sender_id": tracker.sender_id,
            "content_title": payload.get("title"),
            "more_info_url": payload.get("more_info_url"),
        }
        try:
            with open(SECURE_LOG_PATH, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to append brochure log: {exc}")
