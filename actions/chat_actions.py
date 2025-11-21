import time
from typing import Any, Dict, List, Optional, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import FollowupAction
from shared_utils import get_service_manager, MessageFormatter, logger, ModelSelector, Config
import re
from pathlib import Path
import json
from datetime import datetime
 
class ActionLiveAgent(Action):
    _RECENT_REQUEST_CACHE: Dict[str, float] = {}
    _RECENT_CACHE_WINDOW = 5.0  # seconds

    def __init__(self):
        self.services = get_service_manager()
    def name(self) -> Text:
        return "action_live_agent"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_question = tracker.latest_message.get("text", "")
        if len(user_question.strip()) < 2:
            dispatcher.utter_message(text="I'm here to help! What would you like to ask?")
            return [FollowupAction("action_listen")]

        lowered = user_question.lower()
        brochure_triggers = [
            "brochure",
            "coffee table book",
            "coffee-table book",
            "coffee table card",
            "adaptive card invite",
            "birthday invitation",
            "event invitation brochure",
            "invite brochure",
        ]
        if any(trigger in lowered for trigger in brochure_triggers):
            return [FollowupAction("action_generate_brochure")]

        if self._is_duplicate_request(tracker.sender_id, user_question):
            return [FollowupAction("action_listen")]
        try:
            user_metadata = tracker.latest_message.get("metadata") or {}
            preferred_model = user_metadata.get("preferred_model")

            # Only attempt MCP tools when user hasn't explicitly chosen a model
            # (i.e., Auto-Select mode). This prevents accidental web searches for
            # simple questions like "what is 4+4".
            if not preferred_model or str(preferred_model).lower() == 'auto':
                mcp_response = self.services.get_mcp_service().try_all_tools(user_question)
                if mcp_response:
                    dispatcher.utter_message(text=mcp_response)
                    return []

            context = self._build_context(tracker)
            chosen_model = ModelSelector.choose_model(
                user_text=user_question,
                preferred_model=preferred_model,
                available_models=Config.AVAILABLE_MODELS,
                default_model=Config.DEFAULT_TEXT_MODEL,
            )

            # Append model_used event to chatbot.log (even when Auto)
            try:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "event": "model_used",
                    "sender_id": tracker.sender_id,
                    "auto": (not preferred_model or str(preferred_model).lower() == 'auto'),
                    "model": chosen_model,
                    "user_text": (user_question or "")[:500],
                    "role": (user_metadata.get("role") or "user")
                }
                Path("chatbot.log").write_text("", encoding="utf-8") if not Path("chatbot.log").exists() else None
                with open("chatbot.log", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

            # SECURE RAG: map role and call secure wrapper, preserving model selection
            role = (user_metadata.get("role") or "user").strip().lower()
            clean_response = self.services.get_llm_service().generate_text_secure(
                user_text=user_question,
                user_role=role,
                model=chosen_model,
                context_tail=context,
                user_security_groups=user_metadata.get("security_groups"),
                sender_id=tracker.sender_id,
                max_tokens=450,
                timeout=30,
            )

            clean_response = MessageFormatter.clean_markdown_text(clean_response)
            if re.search(r"\b(two|2)\s+sentences\b", user_question.lower()):
                clean_response = MessageFormatter.truncate_sentences(clean_response, 2)

            visual_payload = None
            try:
                if self._should_attach_visual_panel(user_question, clean_response):
                    image_service = self.services.get_image_service()
                    if image_service.is_enabled():
                        query = self._choose_visual_query(user_question, clean_response)
                        image_data = image_service.fetch_image(query)
                        if image_data:
                            visual_payload = self._build_visual_payload(user_question, clean_response, image_data)
                            self._log_visual_panel(tracker, user_question, image_data)
            except Exception as viz_err:
                logger.warning(f"Visual enrichment failed: {viz_err}")

            if visual_payload:
                dispatcher.utter_message(json_message={"payload": "visual_panel", "data": visual_payload})

            dispatcher.utter_message(text=clean_response, metadata={"model_used": chosen_model})

            related_items = self._generate_related_questions(
                tracker=tracker,
                user_question=user_question,
                assistant_response=clean_response,
                chosen_model=chosen_model,
                context=context,
            )
            if related_items:
                dispatcher.utter_message(json_message={
                    "payload": "related_links",
                    "data": {"items": related_items}
                })
                self._log_related_questions(tracker, user_question, related_items)
        except Exception as e:
            logger.error(f"Error in ActionLiveAgent: {e}")
            error_msg = "I'm having trouble processing your request. Please try again."
            if "timeout" in str(e).lower():
                error_msg = "The request is taking longer than expected. Please try a simpler question."
            elif "connection" in str(e).lower():
                error_msg = "I'm having connection issues. Please try again in a moment."
            dispatcher.utter_message(text=error_msg)
        return [FollowupAction("action_listen")]
    def _build_context(self, tracker: Tracker) -> str:
        events = tracker.events_after_latest_restart()
        messages = []
        for event in events[-6:]:
            if event.get('event') == 'user':
                messages.append(f"User: {event.get('text')}")
            elif event.get('event') == 'bot' and event.get('text'):
                messages.append(f"Assistant: {event.get('text')}")
        return "\n".join(messages)

    def _should_attach_visual_panel(self, user_question: str, response_text: str) -> bool:
        text = (user_question or "").strip().lower()
        if not text or len(response_text or "") < 40:
            return False
        trigger_phrases = [
            "what is", "who is", "where is", "tell me about", "show me",
            "describe", "picture", "image", "look like", "landmark", "museum",
        ]
        if any(phrase in text for phrase in trigger_phrases):
            return True
        # Allow fallback when the question asks about places/things broadly.
        keywords = ["building", "monument", "city", "country", "animal", "plant", "food", "art"]
        return any(word in text.split() for word in keywords)

    def _choose_visual_query(self, user_question: str, response_text: str) -> str:
        question = (user_question or "").strip()
        if question:
            sanitized = question.lower()
            prefixes = [
                "what is", "what's", "who is", "who's", "where is", "where's",
                "describe", "tell me about", "show me", "give me", "what does", "how does", "how do", "find",
            ]
            for prefix in prefixes:
                if sanitized.startswith(prefix):
                    question = question[len(prefix):].strip()
                    sanitized = question.lower()
                    break
            question = question.strip(" ?!.") or (user_question or "")
        if question:
            return question[:120]
        return response_text.split(".")[0][:80]

    def _build_visual_payload(
        self,
        user_question: str,
        response_text: str,
        image_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary = MessageFormatter.truncate_sentences(response_text, 2)
        return {
            "title": user_question.strip()[:120] if user_question else "",
            "summary": summary,
            "image_url": image_data.get("image_url"),
            "thumbnail_url": image_data.get("thumbnail_url"),
            "alt": image_data.get("alt") or summary,
            "attribution": image_data.get("attribution"),
            "source_url": image_data.get("context_url") or image_data.get("image_url"),
        }

    def _log_visual_panel(self, tracker: Tracker, question: str, image_data: Dict[str, Any]) -> None:
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "event": "visual_panel",
                "sender_id": tracker.sender_id,
                "question": (question or "")[:200],
                "image_url": image_data.get("image_url"),
                "attribution": image_data.get("attribution"),
            }
            Path("chatbot.log").write_text("", encoding="utf-8") if not Path("chatbot.log").exists() else None
            with open("chatbot.log", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _is_duplicate_request(self, sender_id: Text, question: str) -> bool:
        key = f"{sender_id}:{(question or '').strip().lower()}"
        if not key.strip(":"):
            return False
        now = time.time()
        # prune stale entries
        stale_keys = [k for k, ts in self._RECENT_REQUEST_CACHE.items() if now - ts > self._RECENT_CACHE_WINDOW]
        for k in stale_keys:
            self._RECENT_REQUEST_CACHE.pop(k, None)
        last = self._RECENT_REQUEST_CACHE.get(key)
        self._RECENT_REQUEST_CACHE[key] = now
        return last is not None and (now - last) < 1.5

    def _generate_related_questions(
        self,
        tracker: Tracker,
        user_question: str,
        assistant_response: str,
        chosen_model: str,
        context: str,
    ) -> List[Dict[str, str]]:
        """Use LLM to suggest follow-up prompts tied to the current conversation."""
        try:
            normalized_question = (user_question or "").strip()
            if not normalized_question or len(normalized_question.split()) < 3:
                return []

            recent_context = context or self._build_context(tracker)

            system_prompt = (
                "You generate short follow-up suggestions for a chat assistant. "
                "They must stay tightly related to the latest user request and discussion. "
                "Respond ONLY with JSON matching the schema."
            )
            user_prompt = (
                "Conversation summary:\n"
                f"{recent_context or 'No additional context provided.'}\n\n"
                f"Latest user request: {normalized_question}\n"
                f"Assistant reply: {assistant_response}\n\n"
                "Produce 2 to 3 concise follow-up prompts the user might ask next. "
                "Each prompt must be self-contained and clearly reference the assistant reply so the assistant can act immediately without asking for additional input. "
                "For example, say \"Expand the previous 200-word overview of India into 300 words while keeping the same tone\" instead of simply \"Expand to 300 words\". "
                "Avoid duplicates or generic advice. Keep titles <= 80 characters and prompts <= 140 characters. "
                "Return JSON: {\"related_questions\": [{\"title\": \"...\", \"prompt\": \"...\"} ...]}. "
                "If you cannot find meaningful follow-ups, return {\"related_questions\": []}."
            )

            related_model = chosen_model or ModelSelector.choose_model(
                user_text=normalized_question,
                preferred_model=None,
                available_models=Config.AVAILABLE_MODELS,
                default_model=Config.DEFAULT_TEXT_MODEL,
            )

            llm_response = self.services.get_llm_service().generate_text(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=related_model,
                max_tokens=300,
                temperature=0.2,
                timeout=12,
                response_format={"type": "json_object"},
            )

            parsed = json.loads(llm_response) if isinstance(llm_response, str) else llm_response
            items = parsed.get("related_questions") if isinstance(parsed, dict) else None
            results: List[Dict[str, str]] = []
            if isinstance(items, list):
                for entry in items:
                    title = (entry.get("title") or "").strip()
                    prompt = (entry.get("prompt") or "").strip()
                    if title and prompt:
                        results.append({
                            "title": title[:120],
                            "prompt": prompt[:200],
                        })
            return results[:3]
        except Exception as exc:
            logger.debug(f"Related question generation skipped: {exc}")
            return []

    def _log_related_questions(self, tracker: Tracker, question: str, items: List[Dict[str, str]]) -> None:
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "event": "related_questions",
                "sender_id": tracker.sender_id,
                "question": (question or "")[:300],
                "items": items,
            }
            path = Path("chatbot.log")
            path.write_text("", encoding="utf-8") if not path.exists() else None
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
 
class ActionHandleContactCardSubmission(Action):
    def __init__(self):
        self.services = get_service_manager()
    def name(self) -> Text:
        return "action_handle_contact_card_submission"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        metadata = tracker.latest_message.get('metadata')
        if not metadata:
            dispatcher.utter_message(text="Sorry, I couldn't find the submitted data. Please try again.")
            return []
        try:
            data = {
                "name": metadata.get("person_name") or metadata.get("name") or metadata.get("fullName"),
                "phone": metadata.get("phone_number") or metadata.get("phone") or metadata.get("phoneNumber"),
                "address": metadata.get("address")
            }
            clean_data = {k: str(v).strip() if v else None for k, v in data.items()}
            if not any(clean_data.values()):
                dispatcher.utter_message(text="The form appears to be empty. Please fill in at least one field.")
                return []
            self.services.get_storage_service().store_user_info(tracker.sender_id, clean_data)
            dispatcher.utter_message(text="Information saved successfully! Your details have been recorded.")
        except Exception as e:
            logger.error(f"Error in contact card submission: {e}")
            dispatcher.utter_message(text="An error occurred while processing the form. Please try again.")
        return []
