import logging
from typing import Any, Dict, List, Text
import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import logging
import json
 
class ActionOrchestratorQuery(Action):
    def name(self) -> Text:
        return "action_orchestrator_query"
 
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        logger = logging.getLogger(__name__)
        try:
            user_query = tracker.latest_message.get("text", "")
            if not user_query:
                dispatcher.utter_message(text="Please provide a question.")
                return []
 
            # Call the orchestrator endpoint
            resp = requests.post(
                "http://127.0.0.1:8001/llm",
                json={"query": user_query},
                timeout=180,
            )
 
            if resp.status_code != 200:
                logger.error(f"Orchestrator error: {resp.text}")
                dispatcher.utter_message(
                    text="I'm having trouble getting an answer right now. Please try again later."
                )
                return []
 
            data = resp.json()
            answer = data.get("message", "Sorry, I couldn't find an answer.")
            chart_urls = data.get("chart_urls", [])
 
            # Always send the text answer
            dispatcher.utter_message(text=answer)
            logger.info(f"API answered: {answer}")
            logger.info(f"API chart_urls: {chart_urls}")
 
            # Send each chart as a native image message (renders in common Rasa web UIs)
            for chart_url in chart_urls:
                logger.info(f"Sending image: {chart_url}")
                dispatcher.utter_message(image=chart_url)
 
            return []
 
        except Exception as e:
            logger.error(f"Exception in ActionOrchestratorQuery: {e}", exc_info=True)
            dispatcher.utter_message(text="An error occurred while processing the request.")
            return []