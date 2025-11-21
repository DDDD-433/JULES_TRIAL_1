"""Rasa custom action to call the external retrieval service."""
from __future__ import annotations

import json
import logging
import os
import textwrap
import threading
import uuid
from typing import Any, Dict, List

import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.events import EventType
from rasa_sdk.executor import CollectingDispatcher

# Load environment variables from a local .env if present so VECTOR_API_KEY/VECTOR_BASE_URL are picked up
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

LOGGER = logging.getLogger("action_vector_search")
if not LOGGER.handlers:
    logfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chatbot.log")
    logfile = os.path.abspath(logfile)
    handler = logging.FileHandler(logfile, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)


class ActionVectorSearch(Action):
    """Call the retrieval service and surface raw snippets to the user."""

    name = "action_vector_search"

    _seed_lock = threading.Lock()
    _seeded = False

    def __init__(self) -> None:
        self.base_url = os.getenv("VECTOR_BASE_URL", "http://localhost:8001").rstrip("/")
        self.api_key = os.getenv("VECTOR_API_KEY", "")
        self.timeout = float(os.getenv("VECTOR_TIMEOUT", "6.0"))
        self.default_namespace = os.getenv("VECTOR_NAMESPACE", "demo").strip() or "demo"
        self.auto_seed = os.getenv("VECTOR_AUTO_SEED", "1") != "0"
        self.seed_backend = os.getenv("VECTOR_SEED_BACKEND", os.getenv("DEFAULT_BACKEND", "auto"))

    def _ensure_seeded(self) -> None:
        if not self.auto_seed or type(self)._seeded:
            return
        with type(self)._seed_lock:
            if type(self)._seeded:
                return
            payload = {
                "namespace": self.default_namespace,
                "backend": self.seed_backend,
                "items": [
                    {
                        "id": "doc-qdrant",
                        "text": "Qdrant is a vector database that provides high-performance similarity search with HNSW indexing and REST/GRPC APIs.",
                        "metadata": {"tag": "vector", "topic": "qdrant"},
                    },
                    {
                        "id": "doc-vector-intro",
                        "text": "Vector search systems store embeddings to retrieve semantically related documents for retrieval-augmented generation workflows.",
                        "metadata": {"tag": "vector", "topic": "retrieval"},
                    },
                    {
                        "id": "doc-secure-policy",
                        "text": "Security guidelines emphasize confidentiality, mandatory incident reporting, respectful conduct, and controlled access to sensitive HR data.",
                        "metadata": {"tag": "policy", "topic": "security"},
                    },
                ],
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            try:
                response = requests.post(
                    f"{self.base_url}/upsert",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                type(self)._seeded = True
                LOGGER.info("vector demo docs upserted | namespace=%s", self.default_namespace)
            except Exception as exc:  # pragma: no cover - seeding best effort
                LOGGER.warning("vector demo seed failed: %s", exc)
                type(self)._seeded = True  # Prevent repeated attempts

    def name(self) -> str:  # type: ignore[override]
        return "action_vector_search"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_hits(hits: List[Dict[str, Any]], limit: int = 3) -> str:
        lines: List[str] = ["Here's what I found:"]
        for hit in hits[:limit]:
            text = (hit.get("text") or "").strip()
            text = " ".join(text.split())
            if len(text) > 200:
                text = textwrap.shorten(text, width=200, placeholder="...")
            lines.append(f"- {text}")
        return "\n".join(lines)

    def _call_service(self, payload: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/retrieve"
        headers = {
            "Content-Type": "application/json",
            "X-Trace-Id": trace_id,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Rasa Action API
    # ------------------------------------------------------------------
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[EventType]:
        user_message = tracker.latest_message.get("text", "").strip()
        namespace_slot = tracker.get_slot("namespace")
        if isinstance(namespace_slot, str):
            namespace = namespace_slot.strip()
        else:
            namespace = ""
        if not namespace:
            namespace = self.default_namespace
        top_k_slot = tracker.get_slot("top_k")
        top_k = 5
        if isinstance(top_k_slot, int) and 1 <= top_k_slot <= 20:
            top_k = top_k_slot

        trace_id = uuid.uuid4().hex
        payload = {
            "namespace": namespace,
            "query": user_message,
            "top_k": top_k,
            "backend": os.getenv("DEFAULT_BACKEND", "auto"),
        }

        metadata = tracker.latest_message.get("metadata") or {}
        groups = metadata.get("security_groups") or metadata.get("oracle_groups")
        if isinstance(groups, (list, tuple)):
            filters = {}
            cleaned_groups = [str(group).strip().lower() for group in groups if str(group).strip()]
            if cleaned_groups:
                filters["security_groups"] = cleaned_groups
            if filters:
                payload["filters"] = filters

        LOGGER.info(
            "vector search start | trace_id=%s namespace=%s top_k=%s message=%s",
            trace_id,
            namespace,
            top_k,
            user_message,
        )

        self._ensure_seeded()

        if not user_message:
            dispatcher.utter_message("I need some text to search. Could you rephrase your request?")
            LOGGER.info("vector search empty message | trace_id=%s", trace_id)
            return []

        try:
            result = self._call_service(payload, trace_id)
            hits = result.get("hits") or []
            if hits:
                reply = self._format_hits(hits, limit=min(len(hits), top_k))
                dispatcher.utter_message(reply)
                LOGGER.info(
                    "vector search success | trace_id=%s hits=%s backend=%s",
                    trace_id,
                    len(hits),
                    result.get("backend"),
                )
            else:
                dispatcher.utter_message(
                    "I couldn't find anything for that. Try rephrasing or asking about a different topic."
                )
                LOGGER.info("vector search no hits | trace_id=%s", trace_id)
        except requests.Timeout:
            dispatcher.utter_message(
                "The search service took too long to respond. Please try again in a moment."
            )
            LOGGER.warning("vector search timeout | trace_id=%s", trace_id)
        except requests.HTTPError as exc:
            dispatcher.utter_message(
                "The search service returned an error. Please try again shortly."
            )
            LOGGER.error(
                "vector search HTTP error | trace_id=%s status=%s body=%s",
                trace_id,
                exc.response.status_code if exc.response else "unknown",
                exc.response.text if exc.response else "",
            )
        except Exception as exc:  # pragma: no cover - defensive
            dispatcher.utter_message(
                "I ran into a problem while searching. Please try again later."
            )
            LOGGER.exception("vector search exception | trace_id=%s error=%s", trace_id, exc)
        return []



