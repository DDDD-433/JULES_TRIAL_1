from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, FollowupAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.types import DomainDict
from shared_utils import get_service_manager, MessageFormatter, logger
 
class ActionSendToAgent(Action):
    def __init__(self):
        self.services = get_service_manager()
    def name(self) -> Text:
        return "action_send_to_agent"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        slots = {
            "name": tracker.get_slot("person_name"),
            "phone": tracker.get_slot("phone_number"),
            "address": tracker.get_slot("address")
        }
        missing = [k for k, v in slots.items() if not v]
        if missing:
            dispatcher.utter_message(text=f"Missing: {', '.join(missing)}. Please provide all details.")
            return [FollowupAction("info_form")]
        try:
            self.services.get_storage_service().store_user_info(tracker.sender_id, slots)
            dispatcher.utter_message(text=(
                f"✅ Information Saved!\n"
                f"• Name: {slots['name']}\n"
                f"• Phone: {slots['phone']}\n"
                f"• Address: {slots['address']}"
            ))
        except Exception as e:
            logger.error(f"Error saving info: {e}")
            dispatcher.utter_message(text="Information collected but there was an issue saving it.")
        return [SlotSet(k, None) for k in ["person_name", "phone_number", "address"]]
 
class ValidateInfoForm(FormValidationAction):
    def __init__(self):
        self.services = get_service_manager()
    def name(self) -> Text:
        return "validate_info_form"
    def validate_person_name(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        value = str(slot_value or "").strip()
        # Suppress early warnings when user hasn't answered yet
        if not value:
            return {"person_name": None}
        if self.services.get_validation_service().validate_name(value):
            return {"person_name": value}
        dispatcher.utter_message(text="Please provide a valid name (at least 2 characters).")
        return {"person_name": None}
    def validate_phone_number(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        value = str(slot_value or "").strip()
        if not value:
            return {"phone_number": None}
        validator = self.services.get_validation_service()
        normalized = validator.normalize_phone(value)
        is_valid = validator.validate_phone(normalized)
        logger.debug("validate_phone_number | raw=%s normalized=%s valid=%s", value, normalized, is_valid)
        if is_valid:
            return {"phone_number": normalized}
        dispatcher.utter_message(text="Please provide a valid phone number with at least 10 digits.")
        return {"phone_number": None}
    def validate_address(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        value = str(slot_value or "").strip()
        if not value:
            return {"address": None}
        if self.services.get_validation_service().validate_address(value):
            return {"address": value}
        dispatcher.utter_message(text="Please provide a more complete address.")
        return {"address": None}
 
class ActionContactInfoAdaptiveCard(Action):
    def __init__(self):
        self.services = get_service_manager()
    def name(self) -> Text:
        return "action_contact_info_adaptive_card"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_info = self.services.get_storage_service().get_user_info(tracker.sender_id)
        if user_info:
            card = self._create_display_card(user_info)
            dispatcher.utter_message(json_message={"payload": "adaptiveCard", "data": card})
        else:
            dispatcher.utter_message(text="No stored information found yet. Say 'collect my info' to start the form.")
        return []

    def _create_display_card(self, user_info: Dict) -> Dict:
        return {
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {"type": "TextBlock", "text": "Contact Information", "weight": "Bolder", "size": "Large", "horizontalAlignment": "Center"},
                {"type": "Image", "url": "/static/images/nch-hero-bg.jpg", "size": "Large", "horizontalAlignment": "Center", "width": "75%"},
                {"type": "TextBlock", "text": f"Name - {user_info.get('name', 'Not provided')}", "wrap": True, "spacing": "Small"},
                {"type": "TextBlock", "text": f"Phone Number - {user_info.get('phone', 'Not provided')}", "wrap": True, "spacing": "Small"},
                {"type": "TextBlock", "text": f"Address - {user_info.get('address', 'Not provided')}", "wrap": True, "spacing": "Small"}
            ]
        }

    def _create_input_card(self, user_info: Dict) -> Dict:
        """Interactive Adaptive Card with inputs and submit."""
        return {
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {"type": "TextBlock", "text": "Contact Information Form", "weight": "Bolder", "size": "Large", "horizontalAlignment": "Center"},
                {"type": "Image", "url": "/static/images/nch-hero-bg.jpg", "size": "Large", "horizontalAlignment": "Center", "width": "75%"},
                {"type": "TextBlock", "text": "Please provide your details below:", "wrap": True, "spacing": "Small"},
                {
                    "type": "Container",
                    "spacing": "Small",
                    "items": [
                        {"type": "TextBlock", "text": "Name -", "weight": "Bolder", "spacing": "None"},
                        {"type": "Input.Text", "id": "person_name", "placeholder": "Full name", "value": user_info.get("name", "")}
                    ]
                },
                {
                    "type": "Container",
                    "spacing": "Small",
                    "items": [
                        {"type": "TextBlock", "text": "Phone Number -", "weight": "Bolder", "spacing": "None"},
                        {"type": "Input.Text", "id": "phone_number", "placeholder": "Phone number", "value": user_info.get("phone", "")}
                    ]
                },
                {
                    "type": "Container",
                    "spacing": "Small",
                    "items": [
                        {"type": "TextBlock", "text": "Address -", "weight": "Bolder", "spacing": "None"},
                        {"type": "Input.Text", "id": "address", "isMultiline": True, "placeholder": "Complete address", "value": user_info.get("address", "")}
                    ]
                }
            ],
            "actions": [
                {"type": "Action.Submit", "title": "Submit", "data": {"submit": "contact_info_submit"}}
            ]
        }





