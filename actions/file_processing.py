from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from shared_utils import get_service_manager, MessageFormatter, logger, ModelSelector, Config
 
class ActionUploadFile(Action):
    def name(self) -> Text:
        return "action_upload_file"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text=(
            "📁 File Upload Instructions\n\n"
            "I can analyze images (JPG, PNG, GIF, WebP) and documents (PDF, DOCX, TXT).\n\n"
            "Steps:\n1. Upload your file\n2. Say 'analyze image' or 'process document'\n3. I'll analyze it for you!"
        ))
        return []
 
class ActionAnalyzeImage(Action):
    def __init__(self):
        self.services = get_service_manager()
    def name(self) -> Text:
        return "action_analyze_image"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            image_files = self.services.get_file_service().find_recent_files('image')
            if not image_files:
                dispatcher.utter_message(text="🖼️ No recent images found. Please upload an image first, then say 'analyze image'.")
                return []
            result = self.services.get_file_service().process_image(image_files[0])
            if result:
                clean_result = MessageFormatter.clean_markdown_text(result)
                dispatcher.utter_message(text=f"🔍 Image Analysis Result:\n\n{clean_result}")
            else:
                dispatcher.utter_message(text="❌ I couldn't analyze the image. Please ensure it's a valid image file.")
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            dispatcher.utter_message(text="Error analyzing image. Please try again.")
        return []
 
class ActionProcessDocument(Action):
    def __init__(self):
        self.services = get_service_manager()
    def name(self) -> Text:
        return "action_process_document"
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            doc_files = self.services.get_file_service().find_recent_files('document')
            if not doc_files:
                dispatcher.utter_message(text="📄 No recent documents found. Please upload a document first, then say 'process document'.")
                return []
            # Choose a model if needed in future expansions; currently FileService handles LLM
            result = self.services.get_file_service().process_document(doc_files[0])
            if result:
                clean_result = MessageFormatter.clean_markdown_text(result)
                dispatcher.utter_message(text=f"🔍 Document Analysis Result:\n\n{clean_result}")
            else:
                dispatcher.utter_message(text="❌ I couldn't process the document. Please ensure it's a valid PDF, Word, or text file.")
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            dispatcher.utter_message(text="Error processing document. Please try again.")
        return []