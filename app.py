import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional
 
import requests
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, login_required, current_user
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
 
from shared_utils import get_service_manager, MessageFormatter, logger
from models.user import db, User
from auth.routes import auth_bp
from auth.rbac import role_required
 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
# Keep werkzeug INFO level so Flask prints clickable URLs
 
try:
    from dotenv import load_dotenv
    load_dotenv()
    # logger.info("✅ Flask: Environment variables loaded from .env file")
except ImportError:
    logger.warning("⚠️ Flask: python-dotenv not installed, using system environment variables only")
except Exception as e:
    logger.error(f"⚠️ Flask: Error loading .env file: {e}")
 
# Create Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-this-in-production'))
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Database configuration
if os.getenv('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
else:
    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{data_dir}/app.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

# Flask-Login configuration
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    try:
        # SQLAlchemy 2.0 style to avoid LegacyAPIWarning
        return db.session.get(User, int(user_id))
    except Exception:
        return None

CORS(app)

# Register blueprints
app.register_blueprint(auth_bp)
 
RASA_SERVER_URL = "http://localhost:5005"
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

MAX_UPLOAD_AGE_HOURS = int(os.getenv('UPLOAD_MAX_AGE_HOURS', '24'))
MAX_UPLOAD_FILES = int(os.getenv('UPLOAD_MAX_FILES', '200'))


def cleanup_upload_folder(max_age_hours: int = MAX_UPLOAD_AGE_HOURS, max_files: int = MAX_UPLOAD_FILES) -> None:
    """Remove old or excess files from the upload directory."""
    try:
        if not UPLOAD_FOLDER.exists():
            return

        files = [p for p in UPLOAD_FOLDER.iterdir() if p.is_file()]
        if max_age_hours > 0:
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            for file_path in files:
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff:
                    file_path.unlink(missing_ok=True)

        if max_files > 0:
            files = [p for p in UPLOAD_FOLDER.iterdir() if p.is_file()]
            if len(files) > max_files:
                for file_path in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[max_files:]:
                    file_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning(f"Upload cleanup warning: {exc}")


ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}
ALLOWED_DOCUMENT_EXTENSIONS = {'pdf', 'docx', 'txt'}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_DOCUMENT_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
def is_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
 
def send_message_to_rasa(message: str, sender_id: str = "flask_user", metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    try:
        url = f"{RASA_SERVER_URL}/webhooks/rest/webhook"
        payload: Dict[str, Any] = {"sender": sender_id, "message": message}
        if metadata:
            payload["metadata"] = metadata
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            responses = response.json()
            # logger.info(f"Rasa response: {responses}")
            return responses
        else:
            error_msg = f"Error: Rasa server responded with status {response.status_code}"
            logger.error(error_msg)
            return [{"text": error_msg}]
    except requests.exceptions.ConnectionError as e:
        error_msg = "❌ Cannot connect to Rasa server. Make sure it's running on http://localhost:5005"
        logger.error(f"Connection error: {e}")
        return [{"text": error_msg}]
    except requests.exceptions.Timeout as e:
        error_msg = "❌ Request timed out. Please try again."
        logger.error(f"Timeout error: {e}")
        return [{"text": error_msg}]
    except Exception as e:
        error_msg = f"❌ Error communicating with Rasa: {str(e)}"
        logger.error(error_msg)
        return [{"text": error_msg}]
 
 
def _append_json_log(entry: Dict[str, Any], log_file: Path = Path("chatbot.log")) -> None:
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True) if hasattr(log_file, "parent") else None
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to write to chatbot.log: {e}")
 
 
@app.route("/chat", methods=["POST"])
@login_required
def chat_proxy():
    try:
        data = request.get_json(force=True) or {}
        message: str = data.get("message", "")
        # Scope sender_id to the authenticated user to avoid cross-user leakage
        try:
            user_id = getattr(current_user, 'id', None)
            sender: str = f"user-{int(user_id)}" if user_id is not None else "flask_user"
        except Exception:
            sender = "flask_user"
        metadata: Optional[Dict[str, Any]] = data.get("metadata", {})
        # Inject authenticated user's role for Secure RAG if available
        try:
            user_role = getattr(current_user, 'role', None)
            if user_role:
                metadata["role"] = str(user_role)
        except Exception:
            pass

        # Attach security groups derived from the SSO stub
        try:
            oracle_groups = session.get('oracle_groups') or []
            if oracle_groups and isinstance(oracle_groups, list):
                metadata["security_groups"] = oracle_groups
        except Exception:
            pass

        # NEW: Language detection and input translation
        translation_service = get_service_manager().get_translation_service()
        detected_language, confidence = translation_service.detect_language(message)
       
        # Store original language and message for response translation
        metadata["original_language"] = detected_language
        metadata["original_message"] = message
        metadata["language_confidence"] = confidence
        direction = 'rtl' if detected_language == 'ar' else 'ltr'
        metadata["text_direction"] = direction
       
        # Translate Arabic input to English for system processing
        processed_message = message
        if detected_language == "ar":
            processed_message = translation_service.translate_arabic_to_english(message)
            logger.info(f"🔄 Translated AR→EN: {message[:50]}... → {processed_message[:50]}...")
 
        # Send translated/original message to Rasa
        responses = send_message_to_rasa(
            message=processed_message,
            sender_id=sender,
            metadata=metadata
        )

        if isinstance(responses, list):
            for item in responses:
                if isinstance(item, dict):
                    meta = item.get("metadata")
                    if not isinstance(meta, dict):
                        meta = {} if meta is None else {"value": meta}
                    meta.setdefault("language", detected_language)
                    meta["direction"] = direction
                    item["metadata"] = meta
 
        # NEW: Translate responses back to user's original language
        if detected_language == "ar":
            for response in responses:
                if "text" in response and response["text"]:
                    original_text = response["text"]
                    translated_text = translation_service.translate_english_to_arabic(original_text)
                    response["text"] = translated_text
                    logger.info(f"🔄 Translated EN→AR: {original_text[:50]}... → {translated_text[:50]}...")
 
        # Log each bot response as a JSON line with translation info
        timestamp = datetime.now().isoformat()
        for item in responses:
            _append_json_log({
                "timestamp": timestamp,
                "event": "bot_uttered",
                "source": "rasa",
                "sender_id": sender,
                "request": message,  # Original user message
                "processed_request": processed_message,  # Translated message sent to Rasa
                "original_language": detected_language,
                "language_confidence": confidence,
                "metadata": metadata,
                "response": item
            })
       
        # Log model preference per request
        if metadata and "preferred_model" in metadata:
            _append_json_log({
                "timestamp": timestamp,
                "event": "model_preference",
                "sender_id": sender,
                "preferred_model": metadata.get("preferred_model")
            })
 
        return jsonify(responses)
    except Exception as e:
        logger.error(f"/chat proxy error: {e}")
        return jsonify({"error": "Chat proxy failed"}), 500
 
def check_rasa_server() -> bool:
    try:
        response = requests.get(f"{RASA_SERVER_URL}/version", timeout=5)
        return response.status_code == 200
    except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
        logger.warning(f"Rasa server check failed: {e}")
        return False
 
def process_image_with_vision(image_file, question: str = "Describe this image") -> str:
    try:
        services = get_service_manager()
        file_service = services.get_file_service()
        llm_service = services.get_llm_service()
       
        if hasattr(image_file, 'read'):
            image_bytes = image_file.read()
            image_file.seek(0)
        else:
            with open(image_file, 'rb') as f:
                image_bytes = f.read()
       
        image_path = Path("temp_image.jpg")
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
       
        result = file_service.process_image(image_path)
        image_path.unlink(missing_ok=True)
       
        return MessageFormatter.clean_markdown_text(result)
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return f"Error processing image: {str(e)}"
 
def analyze_document_with_llm(document_text: str, question: str = "Summarize this document") -> str:
    try:
        services = get_service_manager()
        messages = [
            {"role": "system", "content": "Analyze and summarize the document content."},
            {"role": "user", "content": f"Document: {document_text[:3000]}...\n\nQuestion: {question}"}
        ]
        result = services.get_llm_service().generate_text(messages)
        return MessageFormatter.clean_markdown_text(result)
    except Exception as e:
        logger.error(f"Document analysis error: {e}")
        return f"Error analyzing document: {str(e)}"
 
def initialize_session():
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'processed_files' not in session:
        session['processed_files'] = []
 
@app.route('/')
@login_required
def index():
    initialize_session()
    return render_template('components/index.html', chat_history=session.get('chat_history', []))

# Convenience: /login and /logout paths
@app.route('/login')
def login_redirect():
    return redirect(url_for('auth.login'))

@app.route('/logout')
def logout_redirect():
    return redirect(url_for('auth.logout'))
 
@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    initialize_session()
   
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
       
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
       
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
       
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)
       
        file_type = request.form.get('file_type', 'auto')
        question = request.form.get('question', '')
       
        if file_type == 'auto':
            file_type = 'image' if is_image_file(file.filename) else 'document'
       
        if not question:
            question = 'Describe this image in detail' if file_type == 'image' else 'Summarize the main points of this document'
       
        if file_type == 'image':
            with open(filepath, 'rb') as f:
                result = process_image_with_vision(f, question)
        else:
            try:
                services = get_service_manager()
                result = services.get_file_service().process_document(filepath)
            except Exception as e:
                result = f"❌ Error processing document: {str(e)}"
       
        file_info = {
            'type': file_type,
            'name': file.filename,
            'question': question[:100] + '...' if len(question) > 100 else question,
            'result': result[:500] + '...' if len(result) > 500 else result,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
       
        if 'processed_files' not in session:
            session['processed_files'] = []
       
        session['processed_files'].append(file_info)

        if len(session['processed_files']) > 3:
            session['processed_files'] = session['processed_files'][-3:]

        session.modified = True

        # Perform periodic cleanup to avoid unbounded storage growth
        cleanup_upload_folder()
       
        # Log non-Rasa bot-style response from file processing
        _append_json_log({
            "timestamp": datetime.now().isoformat(),
            "event": "bot_uttered",
            "sender_id": session.get('session_id', 'flask_user'),
            "request": f"/upload {file_type}",
            "metadata": {"filename": file.filename},
            "response": {"text": result, "file_type": file_type, "filename": file.filename},
            "source": "flask_upload"
        })
 
        return jsonify({
            'success': True,
            'result': result,
            'file_type': file_type,
            'filename': file.filename
        })
       
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': f'File processing failed: {str(e)}'}), 500
 
@app.route('/health')
def health():
    rasa_status = check_rasa_server()
   
    try:
        services = get_service_manager()
        services.get_llm_service()._get_api_key()
        api_key_status = True
    except ValueError:
        api_key_status = False
   
    return jsonify({
        'status': 'healthy',
        'rasa_server': 'connected' if rasa_status else 'disconnected',
        'groq_api': 'configured' if api_key_status else 'missing',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/healthz')
def healthz():
    return jsonify({'ok': True}), 200
 
@app.route('/clear_chat', methods=['POST'])
@login_required
def clear_chat():
    session['chat_history'] = []
    session['processed_files'] = []
    session.modified = True
    return jsonify({'success': True, 'message': 'Chat history cleared'})

@app.route('/admin')
@role_required('admin')
def admin_dashboard():
    """Admin dashboard"""
    users = User.query.all()
    return render_template('admin/dashboard.html', users=users)


@app.route('/admin/api/users', methods=['GET'])
@role_required('admin')
def admin_list_users():
    """API endpoint to list all users for admin"""
    users = User.query.all()
    users_data = [
        {
            'id': user.id,
            'email': user.email,
            'role': user.role,
            'created_at': user.created_at.isoformat()
        }
        for user in users
    ]
    return jsonify({'users': users_data})

# Role change endpoint removed per requirements (UI is view-only)
 
@app.route('/api/models', methods=['GET'])
@login_required
def get_available_models():
    try:
        from shared_utils import Config
        return jsonify({
            'success': True,
            'models': Config.AVAILABLE_MODELS
        })
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': 'Failed to retrieve models'}), 500


@app.route('/api/model_preference', methods=['POST'])
@login_required
def set_model_preference():
    try:
        data = request.get_json()
        preferred_model = data.get('model')
       
        if not preferred_model:
            return jsonify({'error': 'Model not specified'}), 400
       
        from shared_utils import Config
        if preferred_model not in Config.AVAILABLE_MODELS and preferred_model != 'auto':
            return jsonify({'error': 'Invalid model'}), 400
       
        if 'model_preferences' not in session:
            session['model_preferences'] = {}
       
        session['model_preferences']['preferred_model'] = preferred_model
        session.modified = True
       
        # logger.info(f"Set user model preference: {preferred_model}")
       
        return jsonify({
            'success': True,
            'message': f'Model preference set to: {preferred_model}',
            'preferred_model': preferred_model
        })
       
    except Exception as e:
        logger.error(f"Error setting model preference: {e}")
        return jsonify({'error': 'Failed to set model preference'}), 500
 
@app.route('/api/model_preference', methods=['GET'])
@login_required
def get_model_preference():
    try:
        preferences = session.get('model_preferences', {})
        preferred_model = preferences.get('preferred_model', 'auto')
       
        return jsonify({
            'success': True,
            'preferred_model': preferred_model
        })
       
    except Exception as e:
        logger.error(f"Error getting model preference: {e}")
        return jsonify({'error': 'Failed to get model preference'}), 500
 
@app.route('/api/test_model', methods=['POST'])
@login_required
def test_model():
    try:
        data = request.get_json()
        model = data.get('model', 'llama-3.1-8b-instant')
        test_query = data.get('query', 'Hello! Please respond with a short greeting.')
       
        from shared_utils import Config
        if model not in Config.AVAILABLE_MODELS:
            return jsonify({'error': 'Invalid model'}), 400
       
        services = get_service_manager()
        messages = [{"role": "user", "content": test_query}]
       
        start_time = datetime.now()
        response = services.get_llm_service().generate_text(
            messages=messages,
            model=model,
            max_tokens=100,
            temperature=0.7,
            timeout=10
        )
        end_time = datetime.now()
       
        response_time = (end_time - start_time).total_seconds()
       
        # Log LLM direct test as a bot response for audit
        _append_json_log({
            "timestamp": datetime.now().isoformat(),
            "event": "bot_uttered",
            "sender_id": session.get('session_id', 'flask_user'),
            "request": f"/api/test_model: {test_query}",
            "metadata": {"model": model},
            "response": {"text": response},
            "source": "llm_test"
        })
 
        return jsonify({
            'success': True,
            'model': model,
            'response': response,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        })
       
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'model': data.get('model', 'unknown') if 'data' in locals() else 'unknown'
        }), 500
 
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
 
@app.errorhandler(404)
def not_found(e):
    return render_template('components/index.html'), 404
 
@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

# Serve a favicon to avoid 404 spam in logs
from flask import send_from_directory
@app.route('/favicon.ico')
def favicon():
    try:
        return send_from_directory('static/images', 'aliza-icon.jpg')
    except Exception:
        # Fallback: no favicon
        return ('', 204)

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
        logger.info("✅ Database tables created")
    
    # logger.info("🚀 Starting Flask application...")
   
    rasa_status = "✅ Connected" if check_rasa_server() else "❌ Disconnected"
    # logger.info(f"Rasa Server: {rasa_status}")
   
    try:
        services = get_service_manager()
        services.get_llm_service()._get_api_key()
        api_key_status = "✅ Set"
    except ValueError:
        api_key_status = "❌ Missing"
    # logger.info(f"GROQ API Key: {api_key_status}")
   
    if rasa_status == "❌ Disconnected":
        logger.warning("🚨 Start Rasa server: `rasa run --enable-api --cors '*'`")
    if api_key_status == "❌ Missing":
        logger.warning("🔑 Set GROQ_API_KEY for AI features")
   
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )
