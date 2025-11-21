import json
import os
import random
from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, FollowupAction
from rasa_sdk.executor import CollectingDispatcher
from shared_utils import get_service_manager, logger
 
class ActionBotCapabilities(Action):
    def name(self) -> Text:
        return "action_bot_capabilities"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text=(
            "🤖 Hi! I'm Alizha, your advanced AI assistant. Here's what I can help you with:\n\n"
            "📋 Information Collection\n• Collect and store your personal details\n• Create adaptive contact forms\n\n"
            "📊 Survey Generation\n• Generate dynamic surveys based on conversations\n\n"
            "🖼️ File Analysis\n• Analyze and describe images\n• Process and summarize documents\n\n"
            "💬 General Assistance\n• Answer questions on various topics\n\n"
            "Just ask me anything or say 'collect my info', 'generate survey', 'adaptive card', or upload files!"
        ))
        return []
 
class ActionOutOfScope(Action):
    def name(self) -> Text:
        return "action_out_of_scope"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get("text", "").lower()
        intent_name = tracker.latest_message['intent'].get('name')
        confidence = tracker.latest_message['intent'].get('confidence', 0)
        # Reduced noisy info log
        return [FollowupAction("action_live_agent")]
 
class ActionResetSlots(Action):
    def name(self) -> Text:
        return "action_reset_slots"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        slots_to_reset = [
            "person_name", "phone_number", "address", "survey_question_1", "survey_question_2",
            "survey_question_3", "current_survey_question_1", "current_survey_question_2",
            "current_survey_question_3", "dynamic_form_schema"
        ]
        return [SlotSet(slot, None) for slot in slots_to_reset]
 
class ActionExit(Action):
    def name(self) -> Text:
        return "action_exit"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="👋 Thank you for using our service! Have a great day! Feel free to return anytime if you need assistance.")
        return []
 
class ActionCorrectInfo(Action):
    def name(self) -> Text:
        return "action_correct_info"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Let's correct your information. I'll start a new form for you.")
        return [SlotSet("person_name", None), SlotSet("phone_number", None), SlotSet("address", None), FollowupAction("info_form")]
 
class ActionViewStoredInfo(Action):
    def __init__(self):
        self.services = get_service_manager()
    def name(self) -> Text:
        return "action_view_stored_info"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            user_info = self.services.get_storage_service().get_user_info(tracker.sender_id)
            if not user_info:
                dispatcher.utter_message(text="No stored information found yet. Say 'collect my info' to start the form.")
                return []
            # Otherwise show a read-only summary card
            card = self._create_display_card(user_info)
            dispatcher.utter_message(json_message={"payload": "adaptiveCard", "data": card})
        except Exception as e:
            logger.error(f"Error viewing stored info: {e}")
            dispatcher.utter_message(text="I'm having trouble displaying your stored information right now.")
        return []
    def _create_display_card(self, user_info: Dict) -> Dict:
        return {
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {"type": "TextBlock", "text": "Contact Information Form", "weight": "Bolder", "size": "Large", "horizontalAlignment": "Center"},
                {"type": "Image", "url": "/static/images/nch-hero-bg.jpg", "size": "Large", "horizontalAlignment": "Center", "width": "75%"},
                {"type": "TextBlock", "text": f"Name - {user_info.get('name', 'Not provided')}", "wrap": True, "spacing": "Small"},
                {"type": "TextBlock", "text": f"Phone Number - {user_info.get('phone', 'Not provided')}", "wrap": True, "spacing": "Small"},
                {"type": "TextBlock", "text": f"Address - {user_info.get('address', 'Not provided')}", "wrap": True, "spacing": "Small"}
            ]
        }

    # Input card intentionally removed per requirements (read-only only)
 
class ActionContinueOrExit(Action):
    def name(self) -> Text:
        return "action_continue_or_exit"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(
            text="What would you like to do next?",
            buttons=[
                {"title": "Collect Information", "payload": "/collect_info"},
                {"title": "Generate a Survey", "payload": "/generate_survey"},
                {"title": "Ask a Question", "payload": "/ask_anything"},
                {"title": "Exit", "payload": "/goodbye"}
            ]
        )
        return []


