import time
from typing import Any, Dict, List, Optional, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from shared_utils import get_service_manager, MessageFormatter, logger

class ActionMCPUnified(Action):
    """
    Unified MCP action that intelligently routes to appropriate MCP tools
    No need to create separate actions for each new MCP tool!
    """
    def __init__(self):
        self.services = get_service_manager()
   
    def name(self) -> Text:
        return "action_mcp_unified"
   
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            query = tracker.latest_message.get("text", "")
            intent = tracker.latest_message.get("intent", {}).get("name", "")
           
            if len(query.strip()) < 2:
                dispatcher.utter_message(text="Please provide more details for your request.")
                return []

            # Build context
            context: Dict[str, Any] = {
                "sender_id": tracker.sender_id,
                "intent": intent,
            }
            metadata = tracker.latest_message.get("metadata")
            if isinstance(metadata, dict):
                context.update(metadata)

            # Extract employee_id from multiple sources
            slots = getattr(tracker, "slots", {})
            potential_ids: List[Any] = []
            if isinstance(metadata, dict):
                potential_ids.extend([
                    metadata.get("employee_id"),
                    metadata.get("employeeId"),
                ])
            if isinstance(slots, dict):
                potential_ids.append(slots.get("employee_id"))
            potential_ids.append(tracker.sender_id)

            sanitized_id = None
            for candidate in potential_ids:
                candidate_id = sanitize_employee_id(candidate)
                if candidate_id:
                    sanitized_id = candidate_id
                    break

            if sanitized_id:
                context["employee_id"] = sanitized_id
            else:
                context.pop("employee_id", None)

            # Let MCP service intelligently route to the right tool
            result = self.services.get_mcp_service().try_all_tools(query, context=context)
           
            if result:
                if isinstance(result, dict):
                    result_type = result.get("type")
                    if result_type == "card" and "payload" in result:
                        # Merge template into metadata so frontend can access it
                        metadata = result.get("metadata", {})
                        if "template" in result:
                            metadata["template"] = result["template"]
                       
                        dispatcher.utter_message(
                            json_message={
                                "payload": "adaptiveCard",
                                "data": result["payload"],
                                "metadata": metadata,
                            }
                        )
                        return []
                    text_payload = result.get("text")
                    if text_payload:
                        dispatcher.utter_message(text=text_payload)
                        return []
                    dispatcher.utter_message(text=str(result))
                else:
                    dispatcher.utter_message(text=result)
            else:
                # Fallback based on detected intent
                fallback_message = self._get_fallback_message(intent)
                dispatcher.utter_message(text=fallback_message)
               
        except Exception as e:
            logger.error(f"Error in MCP unified action: {e}")
            dispatcher.utter_message(text="I'm having trouble processing your request right now. Please try again later.")
       
        return []
   
    def _get_fallback_message(self, intent: str) -> str:
        """Provide intent-specific fallback messages"""
        fallback_messages = {
            "get_weather": "I couldn't get weather information. Please specify a city name.",
            "get_news": "I couldn't fetch news right now. Please try again later.",
            "web_search": "I couldn't search the web right now. Please try again later.",
            "search_news": "I couldn't search for specific news. Please try again later.",
            "leave_request": "I couldn't process your leave request. Please provide valid dates and leave type."
        }
        return fallback_messages.get(intent, "I couldn't process your request. Please try rephrasing.")


class ActionProcessLeaveSubmission(Action):
    """Process submitted leave form data and validate via API."""
   
    def __init__(self):
        self.services = get_service_manager()
   
    def name(self) -> Text:
        return "action_process_leave_submission"
   
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            metadata = tracker.latest_message.get("metadata", {})
           
            # Extract data from metadata (submitted from adaptive card)
            employee_id = metadata.get("employee_id")
            start_date = metadata.get("start_date")
            end_date = metadata.get("end_date")
            leave_type = metadata.get("leave_type")
           
            # CRITICAL FIX: Use tracker.sender_id to get the correct employee ID
            # The sender_id format is "user-{employee_id}", extract the number
            sender_id = tracker.sender_id
            if sender_id and sender_id.startswith("user-"):
                extracted_employee_id = sender_id.replace("user-", "")
                if extracted_employee_id.isdigit():
                    logger.info(f"Overriding employee_id from {employee_id} to {extracted_employee_id} (from sender_id)")
                    employee_id = extracted_employee_id  # Use the correct employee ID
           
            # Validate required fields
            if not all([employee_id, start_date, end_date, leave_type]):
                dispatcher.utter_message(text="Missing required information. Please fill all fields in the form.")
                return []
           
            # Call the validation service
            result = self.services.get_mcp_service().leave_validate({
                "employee_id": employee_id,
                "start_date": start_date,
                "end_date": end_date,
                "leave_type": leave_type,
            })
           
            # Send response card
            if isinstance(result, dict) and result.get("type") == "card":
                # Merge template into metadata so frontend can access it
                metadata = result.get("metadata", {})
                if "template" in result:
                    metadata["template"] = result["template"]
               
                dispatcher.utter_message(
                    json_message={
                        "payload": "adaptiveCard",
                        "data": result["payload"],
                        "metadata": metadata,
                    }
                )
            else:
                dispatcher.utter_message(text="Leave request processed successfully!")
               
        except Exception as e:
            logger.error(f"Error processing leave submission: {e}")
            dispatcher.utter_message(text="Unable to process your leave request. Please try again.")
       
        return []
    
def sanitize_employee_id(value: Any) -> Optional[str]:
    """Extract only numeric characters from a potential employee identifier."""
    if value is None:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits or None
