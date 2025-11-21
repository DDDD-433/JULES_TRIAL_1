import json
from datetime import datetime
from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, FollowupAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.types import DomainDict
from shared_utils import get_service_manager, logger, ModelSelector, Config
 
class ActionGenerateDynamicSurvey(Action):
    def __init__(self):
        self.services = get_service_manager()
    def name(self) -> Text:
        return "action_generate_dynamic_survey"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            conversation = self._get_conversation(tracker)
            form_schema = self._generate_survey(conversation)
            survey_title = form_schema.get('form_title', 'Survey')
            questions = form_schema.get('questions', [])
            dispatcher.utter_message(text=f"{survey_title} Generated - {len(questions)} questions created based on your conversation.")
            slots_to_set = [SlotSet("dynamic_form_schema", json.dumps(form_schema))]
            for i, question in enumerate(questions[:3], 1):
                slot_name = f"current_survey_question_{i}"
                question_text = question.get('text', f'Question {i}')
                slots_to_set.append(SlotSet(slot_name, question_text))
            dispatcher.utter_message(text="Survey ready! Say 'start survey' to begin answering the questions.")
            return slots_to_set
        except Exception as e:
            logger.error(f"Error generating survey: {e}")
            dispatcher.utter_message(text="Error generating survey. I can still help you with information collection or other tasks!")
            return []
    def _get_conversation(self, tracker: Tracker) -> str:
        try:
            events = tracker.events_after_latest_restart()
        except Exception:
            events = tracker.events
        # Focus survey generation strictly on USER messages to avoid bleeding in system/tool chatter
        user_only = []
        for event in events[-200:]:
            if isinstance(event, dict) and event.get('event') == 'user' and event.get('text'):
                txt = str(event.get('text') or '').strip()
                if txt:
                    user_only.append(txt)
        convo = '\n'.join(user_only)
        # hard trim to 4000 chars to stay safe
        return convo[-4000:]
    def _generate_survey(self, conversation: str) -> Dict[str, Any]:
        if not conversation or len(conversation.strip()) < 10:
            return self._get_fallback_survey()
        prompt = f"""You are an expert survey creator.
Use only the USER's messages below to infer topics they actually asked about.
Do not ask about meta-topics like web search, tools, MCP, forms, or contact collection.
Create exactly 3 specific, concise questions about those user topics (no generic UX or feedback questions).

USER MESSAGES:
{conversation}

Return only valid JSON (no commentary):
{{
  \"form_title\": \"Conversation-Based Survey\",
  \"questions\": [
    {{\"text\": \"[Q1]\", \"type\": \"text\"}},
    {{\"text\": \"[Q2]\", \"type\": \"text\"}},
    {{\"text\": \"[Q3]\", \"type\": \"text\"}}
  ]
}}"""
        messages = [
            {"role": "system", "content": "You are an expert survey creator. Always return valid JSON, no explanations."},
            {"role": "user", "content": prompt}
        ]
        try:
            # Force a JSON-capable, larger-context model for surveys
            chosen_model = "llama-3.3-70b-versatile"
            response = self.services.get_llm_service().generate_text(
                messages,
                model=chosen_model,
                max_tokens=600,
                temperature=0.2,
                response_format={"type": "json_object"},
                timeout=30,
            )
            clean_response = response.strip()
            if clean_response.startswith('```'):
                clean_response = clean_response.split('```')[1]
                if clean_response.startswith('json'):
                    clean_response = clean_response[4:]
            survey_data = json.loads(clean_response)
            if "questions" in survey_data and len(survey_data["questions"]) >= 3:
                return survey_data
        except Exception as e:
            logger.error(f"LLM survey generation failed: {e}")
            # one retry with stricter instruction
            try:
                # Retry without forcing JSON mode and parse the first JSON object
                response = self.services.get_llm_service().generate_text(
                    messages,
                    model=chosen_model,
                    max_tokens=600,
                    temperature=0.1,
                    timeout=30,
                )
                clean = response.strip()
                if clean.startswith('```'):
                    clean = clean.split('```')[1]
                    if clean.startswith('json'):
                        clean = clean[4:]
                survey_data = json.loads(clean)
                if "questions" in survey_data and len(survey_data["questions"]) >= 3:
                    return survey_data
            except Exception as e2:
                logger.error(f"Survey retry failed: {e2}")
            return self._get_fallback_survey()
    def _get_fallback_survey(self) -> Dict[str, Any]:
        return {
            "form_title": "Conversation-Based Survey",
            "questions": [
                {"text": "What aspects of our conversation interested you the most?", "type": "text"},
                {"text": "What additional information would be helpful for you?", "type": "text"},
                {"text": "How can we better assist you with your questions?", "type": "text"}
            ]
        }
 
class ActionStartSurveyCollection(Action):
    def name(self) -> Text:
        return "action_start_survey_collection"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            form_schema_json = tracker.get_slot("dynamic_form_schema")
            if not form_schema_json:
                last_text = (tracker.latest_message.get('text') or '').lower()
                if 'survey' not in last_text:
                    dispatcher.utter_message(text="I don't see an active survey yet, so I'll answer your question directly.")
                    return [FollowupAction("action_live_agent")]
                dispatcher.utter_message(text="No survey found. Please generate a survey first.")
                return []
            form_schema = json.loads(form_schema_json)
            questions = form_schema.get('questions', [])
            events_to_return = [
                SlotSet("survey_question_1", None),
                SlotSet("survey_question_2", None),
                SlotSet("survey_question_3", None)
            ]
            dispatcher.utter_message(text=f"Starting Survey - {len(questions)} questions ready. Let's begin!")
            events_to_return.append(FollowupAction("survey_form"))
            return events_to_return
        except Exception as e:
            logger.error(f"Error starting survey collection: {e}")
            dispatcher.utter_message(text="Error starting survey collection.")
            return []
 
class ValidateSurveyForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_survey_form"
    def validate_survey_question_1(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        if slot_value and len(slot_value.strip()) > 0:
            return {"survey_question_1": slot_value.strip()}
        return {"survey_question_1": None}
    def validate_survey_question_2(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        if slot_value and len(slot_value.strip()) > 0:
            return {"survey_question_2": slot_value.strip()}
        return {"survey_question_2": None}
    def validate_survey_question_3(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        if slot_value and len(slot_value.strip()) > 0:
            return {"survey_question_3": slot_value.strip()}
        return {"survey_question_3": None}
 
class ActionSubmitSurveyResponses(Action):
    def name(self) -> Text:
        return "action_submit_survey_responses"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            responses = {}
            for i in range(1, 3+1):
                slot_name = f"survey_question_{i}"
                question_slot = f"current_survey_question_{i}"
                response = tracker.get_slot(slot_name)
                question = tracker.get_slot(question_slot)
                if response and question:
                    responses[f"Q{i}"] = {"question": question, "response": response}
            if responses:
                result_text = self._format_survey_results(responses)
                dispatcher.utter_message(text=result_text)
                return [
                    SlotSet("survey_question_1", None),
                    SlotSet("survey_question_2", None),
                    SlotSet("survey_question_3", None),
                    SlotSet("current_survey_question_1", None),
                    SlotSet("current_survey_question_2", None),
                    SlotSet("current_survey_question_3", None),
                    SlotSet("dynamic_form_schema", None),
                ]
            else:
                dispatcher.utter_message(text="No survey responses found to submit.")
                return []
        except Exception as e:
            logger.error(f"Error in ActionSubmitSurveyResponses: {e}")
            dispatcher.utter_message(text="Error submitting survey responses.")
            return []
    def _format_survey_results(self, responses: Dict) -> str:
        result = "Survey Responses Submitted Successfully!\n\nYour Responses:\n"
        for q_num, data in responses.items():
            result += f"\nQ: {data['question']}\nA: {data['response']}\n"
        result += f"\nTotal Questions Answered: {len(responses)}"
        result += f"\nSubmitted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        result += "\n\nThank you for your feedback!"
        return result
