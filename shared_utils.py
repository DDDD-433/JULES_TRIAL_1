import base64
import io
import json
import logging
import os
import re
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import PyPDF2
from docx import Document
 
 
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# Import langdetect for language detection
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed. Language detection will use LLM only.")
 
class Config:
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TIMEOUT = 30
    VISION_TIMEOUT = 60
    MAX_IMAGE_SIZE_MB = 10
    MAX_IMAGE_DIMENSION = 2048
    JPEG_QUALITY = 85
    MAX_STORAGE_SIZE = 1000
    UPLOAD_FOLDER = "uploads"
    MIN_PHONE_LENGTH = 10
    MAX_PHONE_LENGTH = 15
    MIN_NAME_LENGTH = 2
    MIN_ADDRESS_LENGTH = 3
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    VISION_MODELS = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    ]
    DEFAULT_TEXT_MODEL = "llama-3.1-8b-instant"
    AVAILABLE_MODELS = [
        'llama-3.1-8b-instant',
        'llama-3.3-70b-versatile',
        'meta-llama/llama-guard-4-12b',
        'openai/gpt-oss-120b',
        'openai/gpt-oss-20b',
        'moonshotai/kimi-k2-instruct',
        'qwen/qwen3-32b',
    ]
   
    # Translation Configuration
    TRANSLATION_MODEL = "llama-3.3-70b-versatile"
    TRANSLATION_TIMEOUT = 30
    TRANSLATION_MAX_TOKENS = 1000
    TRANSLATION_TEMPERATURE = 0.3
    SUPPORTED_LANGUAGES = ["en", "ar"]
    LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD = 0.7
 
# Secure RAG JSONL logger helper
def _sr_log(entry: Dict[str, Any]) -> None:
    try:
        path = Path("secure_rag.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
 
class ServiceManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._services = {}
            cls._instance._storage = []
        return cls._instance
    def get_llm_service(self):
        if 'llm' not in self._services:
            self._services['llm'] = LLMService()
        return self._services['llm']
    def get_storage_service(self):
        if 'storage' not in self._services:
            self._services['storage'] = StorageService(self._storage)
        return self._services['storage']
    def get_file_service(self):
        if 'file' not in self._services:
            self._services['file'] = FileService()
        return self._services['file']
    def get_validation_service(self):
        if 'validation' not in self._services:
            self._services['validation'] = ValidationService()
        return self._services['validation']
    def get_mcp_service(self):
        if 'mcp' not in self._services:
            from mcp_utilities.client import (
                run_web_search,
                run_weather_search,
                run_news_search,
                run_leave_analysis,
                run_leave_validation,
                mcp_client,
            )

            # Create the MCP service with core tools
            mcp_service = MCPService(
                run_web_search,
                run_weather_search,
                run_news_search,
                run_leave_analysis,
                run_leave_validation,
                mcp_client,
            )

            self._services['mcp'] = mcp_service
        return self._services['mcp']
    def get_translation_service(self):
        if 'translation' not in self._services:
            self._services['translation'] = LLMTranslationService()
        return self._services['translation']
 
def get_service_manager():
    return ServiceManager()
 
class LLMService:
    def __init__(self):
        # Try to load API key, but don't hard-fail: enable graceful fallback
        try:
            self._api_key = self._get_api_key()
        except Exception as e:
            logger.warning(f"GROQ_API_KEY not configured; enabling fallback mode ({e})")
            self._api_key = ""
 
        # Build headers (omit Authorization if no key)
        self._headers = {"Content-Type": "application/json"}
        if self._api_key:
            self._headers["Authorization"] = f"Bearer {self._api_key}"
 
        # Lazy init for Secure RAG components
        self._input_guard = None
        self._output_guard = None
        self._rbac = None
    def _get_api_key(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or len(api_key.strip()) < 20:
            raise ValueError("GROQ_API_KEY not configured properly")
        return api_key.strip()
    def generate_text(self, messages, model=None, **kwargs):
        """Generate text via Groq HTTP API; fallback to secure_rag GroqService if unavailable.
 
        messages: list of {role, content}. We will extract the latest user prompt and any system
        context for fallback mode.
        """
        # If API key is missing, go straight to fallback
        if not self._api_key:
            return self._generate_text_fallback(messages, model=model, **kwargs)
 
        payload = {
            "model": model or Config.DEFAULT_TEXT_MODEL,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', Config.DEFAULT_MAX_TOKENS),
            "temperature": kwargs.get('temperature', Config.DEFAULT_TEMPERATURE)
        }
        if 'response_format' in kwargs and kwargs['response_format']:
            payload["response_format"] = kwargs['response_format']
 
        try:
            response = requests.post(
                Config.GROQ_API_URL,
                json=payload,
                headers=self._headers,
                timeout=kwargs.get('timeout', Config.DEFAULT_TIMEOUT)
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Groq HTTP call failed; using fallback ({e})")
            return self._generate_text_fallback(messages, model=model, **kwargs)
 
    def _generate_text_fallback(self, messages, model=None, **kwargs) -> str:
        """Use secure_rag GroqService intelligent fallback to produce a best-effort answer."""
        try:
            from secure_rag.groq_service import GroqService
            # Extract prompt and context from messages
            user_prompt = ""
            for m in reversed(messages or []):
                if isinstance(m, dict) and m.get("role") == "user":
                    user_prompt = m.get("content", "")
                    break
            context_parts = [m.get("content", "") for m in (messages or []) if isinstance(m, dict) and m.get("role") == "system"]
            context = "\n\n".join(context_parts).strip()
 
            svc = GroqService(verbose=False, enable_fallback=True)
            resp = svc.generate_text(
                prompt=user_prompt,
                model_id=model or Config.DEFAULT_TEXT_MODEL,
                temperature=kwargs.get('temperature', Config.DEFAULT_TEMPERATURE),
                max_tokens=kwargs.get('max_tokens', Config.DEFAULT_MAX_TOKENS),
                context=context
            )
            return resp.content or ""
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return "I'm sorry, I'm unable to generate a response right now."
 
    # --- Secure RAG integration wrappers ---
    def _init_secure_services(self):
        if self._input_guard is None or self._output_guard is None or self._rbac is None:
            try:
                from secure_rag.input_guard_service import InputGuardService
                from secure_rag.output_guard_service import OutputGuardService
                from secure_rag.rbac_service import RBACService, AccessLevel
                self._InputGuardServiceCls = InputGuardService
                self._OutputGuardServiceCls = OutputGuardService
                self._RBACServiceCls = RBACService
                self._AccessLevelCls = AccessLevel
                # Create instances (verbose False to reduce console noise in prod)
                self._input_guard = self._InputGuardServiceCls(verbose=False)
                self._output_guard = self._OutputGuardServiceCls(verbose=False)
                self._rbac = self._RBACServiceCls(verbose=False)
 
                # Seed RBAC with a minimal document set if empty, so context building is useful
                try:
                    if hasattr(self._rbac, 'documents') and not self._rbac.documents:
                        sample_docs = [
                            {
                                "owner_ids": ["hr_admin", "hr_executive"],
                                "shared": True,
                                "category": "employee_data",
                                "sensitivity_level": "medium",
                                "created_by": "hr_admin",
                                "department": "Engineering",
                                "text": "Employee Details: Akash works at ITC Infotech as IS2 level engineer in Engineering department"
                            },
                            {
                                "owner_ids": ["hr_admin", "emp2"],
                                "shared": True,
                                "category": "employee_data",
                                "sensitivity_level": "medium",
                                "created_by": "hr_admin",
                                "department": "Data Science",
                                "text": "Employee Details: Arpan works at ITC Infotech as IS1 level Data Scientist in MOC Innovation Team"
                            },
                            {
                                "owner_ids": [],
                                "shared": True,
                                "category": "public",
                                "sensitivity_level": "low",
                                "created_by": "system",
                                "department": "General",
                                "text": "Company Policies: Security guidelines, work hours, and professional conduct standards"
                            },
                            {
                                "owner_ids": ["hr_admin"],
                                "shared": True,
                                "category": "general",
                                "sensitivity_level": "low",
                                "created_by": "hr_admin",
                                "department": "General",
                                "text": "ITC Infotech provides digital transformation and IT services to clients worldwide"
                            }
                        ]
                        try:
                            self._rbac.ingest_documents(sample_docs)
                        except Exception as ie:
                            logger.warning(f"RBAC document seeding failed: {ie}")
                except Exception as seed_err:
                    logger.warning(f"RBAC initialization warning: {seed_err}")
            except Exception as e:
                logger.error(f"Secure RAG init failed: {e}")
                self._input_guard = None
                self._output_guard = None
                self._rbac = None
 
    def _map_role_to_user_id(self, user_role: Optional[str]) -> str:
        role = (user_role or "user").strip().lower()
        return "hr_admin" if role == "admin" else "hr_common"
 
    def _should_include_rbac_context(self, prompt: str) -> bool:
        """Heuristic to include RBAC document context only for document-relevant queries.
        Reduces leakage of internal names into unrelated answers (e.g., jokes).
        """
        text = (prompt or "").lower()
        triggers = [
            "employee", "employees", "hr", "confidential", "policy", "policies",
            "document", "documents", "docs", "company policy", "company policies",
            "details for", "itc infotech", "akash", "arpan"
        ]
        return any(t in text for t in triggers)
 
    def generate_text_secure(
        self,
        user_text: str,
        user_role: Optional[str] = None,
        model: Optional[str] = None,
        context_tail: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Secure RAG wrapper around generate_text with input/output guards and RBAC.
 
        - user_role: "admin" or "user" (defaults to user)
        - model: keep your adapter's chosen model (auto/llama/qwen/gptoss)
        - context_tail: optional recent conversation to include for tone/continuity
        """
        # Initialize Secure RAG services (lazy)
        self._init_secure_services()
 
        # If Secure RAG unavailable, fall back to vanilla generation
        if not self._input_guard or not self._output_guard or not self._rbac:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_text},
            ]
            return self.generate_text(messages, model=model, **kwargs)
 
        # 1) Input guard
        scan = self._input_guard.scan_input(user_text or "")
        scan_results = getattr(scan, "scanner_results", {}) if scan else {}
        sanitized_prompt = getattr(scan, "sanitized_prompt", user_text)
        sanitized_changed = sanitized_prompt != user_text
 
        invalid_scanners = [name for name, ok in scan_results.items() if not ok]
        admin_override = False
        if not getattr(scan, "is_valid", True):
            if (user_role or "").strip().lower() == "admin" and invalid_scanners and all(name.lower().startswith("anonymize") for name in invalid_scanners):
                admin_override = True
            else:
                try:
                    _sr_log({
                        "timestamp": datetime.now().isoformat(),
                        "stage": "input_guard",
                        "result": "blocked",
                        "scanner_results": scan_results,
                        "scanner_scores": getattr(scan, 'scanner_scores', {}),
                        "warnings": getattr(scan, 'warnings', []),
                        "errors": getattr(scan, 'errors', [])
                    })
                except Exception:
                    pass
                return "Your input was blocked by security policy. Please rephrase."
 
        if admin_override:
            sanitized_prompt = user_text
            sanitized_changed = False
 
        if (user_role or "").strip().lower() == "admin" and sanitized_changed:
            # Preserve full context for administrators to enable RBAC evaluation.
            sanitized_prompt = user_text
            sanitized_changed = False
        try:
            _sr_log({
                "timestamp": datetime.now().isoformat(),
                "stage": "input_guard",
                "result": "ok",
                "scanner_results": scan_results,
                "scanner_scores": getattr(scan, 'scanner_scores', {}),
                "warnings": getattr(scan, 'warnings', []),
                "errors": getattr(scan, 'errors', []),
                "sanitized": sanitized_changed
            })
        except Exception:
            pass
 
        # 2) RBAC context (admin gets FULL, user LIMITED)
        user_id_for_rbac = self._map_role_to_user_id(user_role)
        access = self._rbac.check_access(user_id_for_rbac, sanitized_prompt)
        # Build context from accessible docs (only when relevant)
        include_docs = self._should_include_rbac_context(sanitized_prompt)
        rbac_context = ""
        if include_docs and access and getattr(access, "filtered_documents", None):
            ctx_parts = []
            for doc in access.filtered_documents:
                part = f"Document: {doc.page_content}"
                cat = doc.metadata.get('category') if isinstance(doc.metadata, dict) else None
                if cat:
                    part += f" (Category: {cat})"
                ctx_parts.append(part)
            rbac_context = "\n\n".join(ctx_parts)
 
        # Optionally include conversation tail
        composed_context = rbac_context
        if context_tail and isinstance(context_tail, str) and len(context_tail.strip()) > 0:
            # Only append conversation tail; do not inject default doc text when none
            composed_context = (composed_context + ("\n\nRecent conversation:\n" + context_tail.strip() if composed_context else f"Recent conversation:\n{context_tail.strip()}"))
 
        # 3) Call existing adapter generate_text (preserve your model selection)
        # Add explicit role-based policy instructions to avoid contradictory answers
        AccessLevel = self._AccessLevelCls  # type: ignore[attr-defined]
        level = AccessLevel.FULL if (user_role or "").strip().lower() == "admin" else AccessLevel.LIMITED
        base_system = "You are a helpful AI assistant."
        if level.name == "LIMITED":
            policy = (
                " Do not disclose any sensitive or employee-specific personal details. "
                "If asked for confidential information, politely refuse and offer only general, public information."
            )
        else:
            policy = (
                " You may answer using the provided document context when the question is about policies, employees, or company documents. "
                "Do not claim you are unauthorized when the context allows answering. Do not introduce internal names in unrelated topics."
            )
        if composed_context:
            sys_content = f"{base_system}{policy} Use this context if relevant: {composed_context}"
        else:
            sys_content = f"{base_system}{policy}"
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": sanitized_prompt},
        ]
        raw = self.generate_text(messages, model=model, **kwargs)
        try:
            _sr_log({
                "timestamp": datetime.now().isoformat(),
                "stage": "llm_generation",
                "model": model or Config.DEFAULT_TEXT_MODEL,
                "prompt_len": len(sanitized_prompt or ""),
                "context_len": len(composed_context or "")
            })
        except Exception:
            pass
 
        # 4) Output guard with role-based enforcement
        try:
            out = self._output_guard.scan_output(sanitized_prompt, raw, access_level=level)
            if not getattr(out, "is_valid", True):
                try:
                    _sr_log({
                        "timestamp": datetime.now().isoformat(),
                        "stage": "output_guard",
                        "result": "blocked",
                        "warnings": getattr(out, 'warnings', []),
                        "errors": getattr(out, 'errors', []),
                        "quality": getattr(out, 'quality_score', None)
                    })
                except Exception:
                    pass
                return "The response was blocked by security policy. Please try a different question."
            final = getattr(out, "sanitized_response", raw) or raw
            try:
                _sr_log({
                    "timestamp": datetime.now().isoformat(),
                    "stage": "output_guard",
                    "result": "ok",
                    "warnings": getattr(out, 'warnings', []),
                    "errors": getattr(out, 'errors', []),
                    "quality": getattr(out, 'quality_score', None),
                    "final_len": len(final or "")
                })
            except Exception:
                pass
            return MessageFormatter.clean_markdown_text(final)
        except Exception as e:
            logger.error(f"Output guard failed, returning raw: {e}")
            return MessageFormatter.clean_markdown_text(raw)
    def analyze_vision(self, image_base64, query, **kwargs):
        for model in Config.VISION_MODELS:
            try:
                payload = {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }],
                    "max_tokens": kwargs.get('max_tokens', Config.DEFAULT_MAX_TOKENS),
                    "temperature": kwargs.get('temperature', Config.DEFAULT_TEMPERATURE)
                }
                response = requests.post(Config.GROQ_API_URL, json=payload, headers=self._headers, timeout=Config.VISION_TIMEOUT)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except Exception:
                continue
        raise Exception("All vision models failed")
 
class ModelSelector:
    @staticmethod
    def choose_model(user_text: Optional[str], preferred_model: Optional[str], available_models: Optional[List[str]] = None, default_model: Optional[str] = None) -> str:
        # Honor explicit user preference when provided
        if preferred_model and isinstance(preferred_model, str) and preferred_model.strip():
            return preferred_model.strip()
 
        text = (user_text or "").lower()
        available_models = available_models or Config.AVAILABLE_MODELS
        default_model = default_model or Config.DEFAULT_TEXT_MODEL
 
        # Helper to pick best available option among candidates, else default
        def pick(*candidates: str) -> str:
            for cand in candidates:
                if cand in available_models:
                    return cand
            return default_model
 
        reasoning_triggers = [
            "why", "explain", "step by step", "think", "reason", "justify", "proof", "derive",
            "analyze", "compare", "evaluate", "plan", "strategy", "trade-off", "pros and cons",
            "eli5", "explain like i'm", "explain like im"
        ]
        coding_triggers = [
            "code", "python", "javascript", "typescript", "bug", "error", "stack trace", "function", "class", "regex",
            "refactor", "optimize", "snippet"
        ]
        math_triggers = [
            "calculate", "compute", "solve", "equation", "integral", "derivative", "proof"
        ]
        json_triggers = ["json", "return json", "json_object", "valid json", "response_format"]
        creative_triggers = ["essay", "story", "poem", "creative", "blog", "rewrite", "paraphrase"]
        quick_triggers = ["hi", "hello", "hey", "thanks", "thank you", "help", "who are you", "what can you do"]
 
        long_query = len(text.split()) >= 60 or len(text) >= 300
 
        # Heuristics
        if any(k in text for k in reasoning_triggers) or "chain of thought" in text or "cot" in text:
            # Heavy reasoning → prefer Qwen first, then 120B, then 70B
            return pick("qwen/qwen3-32b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile")
 
        if any(k in text for k in coding_triggers) or "explain code" in text or "refactor" in text:
            # Coding → prefer Qwen first for code tasks
            return pick("qwen/qwen3-32b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant")
 
        if any(k in text for k in math_triggers):
            return pick("qwen/qwen3-32b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile")
 
        if any(k in text for k in json_triggers):
            return pick("openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant")
 
        if any(k in text for k in creative_triggers):
            return pick("moonshotai/kimi-k2-instruct", "llama-3.3-70b-versatile", "openai/gpt-oss-20b")
 
        if any(text.strip().startswith(t) or t == text.strip() for t in quick_triggers):
            return pick("llama-3.1-8b-instant", "llama-3.3-70b-versatile")
 
        if long_query or "summarize" in text or "analyze" in text or "write" in text:
            # Long/summary → prefer 70B or Qwen before 120B to reduce cost/noise
            return pick("llama-3.3-70b-versatile", "qwen/qwen3-32b", "openai/gpt-oss-120b")
 
        # Default balanced choice
        return pick("llama-3.3-70b-versatile", "openai/gpt-oss-20b", default_model)
 
class StorageService:
    def __init__(self, storage_list):
        self._storage = storage_list
    def store_user_info(self, user_id, data):
        data.update({"user_id": user_id, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        existing = next((item for item in self._storage if item["user_id"] == user_id), None)
        if existing:
            existing.update(data)
        else:
            self._storage.append(data)
        self._cleanup()
    def get_user_info(self, user_id):
        return next((item for item in self._storage if item["user_id"] == user_id), None)
    def _cleanup(self):
        if len(self._storage) >= Config.MAX_STORAGE_SIZE:
            keep_count = int(Config.MAX_STORAGE_SIZE * 0.8)
            self._storage[:] = self._storage[-keep_count:]
 
class ValidationService:
    def validate_name(self, name):
        return bool(name and len(name.strip()) >= Config.MIN_NAME_LENGTH)
    def normalize_phone(self, phone):
        """Return a normalized representation for phone numbers."""
        if phone is None:
            return ""
        text = str(phone).strip()
        if not text:
            return ""
        digits = ''.join(ch for ch in text if ch.isdigit())
        if not digits:
            return ""
        return digits
 
    def validate_phone(self, phone):
        normalized = self.normalize_phone(phone)
        if not normalized:
            return False
        digits = ''.join(ch for ch in normalized if ch.isdigit())
        return Config.MIN_PHONE_LENGTH <= len(digits) <= Config.MAX_PHONE_LENGTH
    def validate_address(self, address):
        return bool(address and len(address.strip()) >= Config.MIN_ADDRESS_LENGTH)
 
class FileService:
    def __init__(self):
        self.upload_dir = Path(Config.UPLOAD_FOLDER)
        self.upload_dir.mkdir(exist_ok=True)
    def process_image(self, image_path):
        try:
            if not self._is_valid_image(image_path):
                return ""
            base64_image = self._encode_image_to_base64(image_path)
            if not base64_image:
                return ""
            llm_service = ServiceManager().get_llm_service()
            return llm_service.analyze_vision(base64_image, "Please analyze this image and provide a detailed description.")
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return ""
    def process_document(self, doc_path):
        try:
            text = self._extract_document_text(doc_path)
            if not text or len(text.strip()) < 10:
                return ""
            messages = [
                {"role": "system", "content": "Analyze and summarize the document content."},
                {"role": "user", "content": f"Document: {text[:2000]}"}
            ]
            llm_service = ServiceManager().get_llm_service()
            result = llm_service.generate_text(messages, max_tokens=800, temperature=0.3)
            return MessageFormatter.clean_markdown_text(result)
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return ""
    def find_recent_files(self, file_type):
        extensions = {
            'image': {'.jpg', '.jpeg', '.png', '.gif', '.webp'},
            'document': {'.pdf', '.docx', '.txt'}
        }
        if file_type not in extensions:
            return []
        files = []
        if self.upload_dir.exists():
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in extensions[file_type]:
                    files.append(file_path)
        return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
    def _is_valid_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    def _encode_image_to_base64(self, image_path):
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            if len(image_bytes) > Config.MAX_IMAGE_SIZE_MB * 1024 * 1024:
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                if image.width > Config.MAX_IMAGE_DIMENSION or image.height > Config.MAX_IMAGE_DIMENSION:
                    image.thumbnail((Config.MAX_IMAGE_DIMENSION, Config.MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=Config.JPEG_QUALITY)
                image_bytes = img_buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return ""
    def _extract_document_text(self, doc_path):
        if doc_path.suffix.lower() == '.pdf':
            return self._extract_from_pdf(doc_path)
        elif doc_path.suffix.lower() == '.docx':
            return self._extract_from_docx(doc_path)
        elif doc_path.suffix.lower() == '.txt':
            return self._extract_from_txt(doc_path)
        else:
            raise Exception(f"Unsupported file type: {doc_path.suffix}")
    def _extract_from_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    def _extract_from_docx(self, docx_path):
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    def _extract_from_txt(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
 
class MCPService:
    """
    Enhanced MCP Service that can dynamically handle new tools
    without requiring code changes for each new tool
    """

    def __init__(
        self,
        web_search_func,
        weather_search_func,
        news_search_func,
        leave_analyze_func,
        leave_validate_func,
        mcp_client,
    ):
        # Core tools
        self.web_search = web_search_func
        self.weather_search = weather_search_func
        self.news_search = news_search_func
        self._leave_analyze = leave_analyze_func
        self._leave_validate = leave_validate_func
        self.mcp_client = mcp_client

        # Registry for additional tools (for future extensibility)
        self._tool_registry = {}
        self._tool_handlers = {}

        # Register core tools
        self._register_core_tools()
   
    def _register_core_tools(self):
        """Register the core MCP tools with their handlers"""
        self._tool_registry.update({
            'weather': {
                'checker': self.mcp_client.should_use_weather,
                'handler': self._handle_weather,
                'description': 'Weather information for cities'
            },
            'news': {
                'checker': self.mcp_client.should_use_news,
                'handler': self._handle_news,
                'description': 'Latest news and news search'
            },
            'web_search': {
                'checker': self.mcp_client.should_use_web_search,
                'handler': self._handle_web_search,
                'description': 'General web search functionality'
            },
            'leave_calculator': {
                'checker': self.mcp_client.should_use_leave_calculator,
                'handler': self._handle_leave,
                'description': 'Leave calculator form and eligibility checks'
            }
        })
   
    def register_tool(self, tool_name: str, checker_func, handler_func, description: str = ""):
        """
        Register a new MCP tool dynamically
       
        Args:
            tool_name: Unique identifier for the tool
            checker_func: Function that checks if this tool should handle the query
            handler_func: Function that processes the query and returns results
            description: Optional description of the tool
        """
        self._tool_registry[tool_name] = {
            'checker': checker_func,
            'handler': handler_func,
            'description': description
        }
        logger.info(f"Registered new MCP tool: {tool_name} - {description}")
   
    def _handle_weather(self, user_question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Handle weather queries"""
        city = self.mcp_client.extract_city_from_query(user_question)
        return self.weather_search(city) if city else "Please specify a city name."
   
    def _handle_news(self, user_question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Handle news queries"""
        if any(word in user_question.lower() for word in ["about", "search", "find"]):
            search_terms = user_question.lower()
            for word in ["news", "about", "search", "find", "latest", "recent"]:
                search_terms = search_terms.replace(word, "")
            search_terms = search_terms.strip()
            return self.news_search(search_terms, is_headlines=False) if search_terms else self.news_search(is_headlines=True)
        return self.news_search(is_headlines=True)
   
    def _handle_web_search(self, user_question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Handle web search queries"""
        return self.web_search(user_question)

    def _handle_leave(self, user_question: str, context: Optional[Dict[str, Any]] = None):
        """Handle leave calculator queries."""
        employee_id = None
        if context:
            raw_id = context.get("employee_id") or context.get("sender_id")
            if raw_id:
                digits_only = "".join(ch for ch in str(raw_id) if ch.isdigit())
                if digits_only:
                    employee_id = digits_only
        hints = {}
        if context:
            for key in ("start_date", "end_date", "leave_type"):
                value = context.get(key)
                if value:
                    hints[key] = value
        result = self._leave_analyze(user_question, employee_id, hints)
        return result

    def leave_validate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._leave_validate(
                str(payload.get("employee_id")),
                str(payload.get("start_date")),
                str(payload.get("end_date")),
                str(payload.get("leave_type")),
            )
        except Exception as exc:
            logger.error(f"Leave validation pipeline error: {exc}")
            return {"error": "Unable to validate leave request right now."}
   
    def try_all_tools(self, user_question: str, context: Optional[Dict[str, Any]] = None):
        """
        Dynamically try all registered tools to handle the user question
        Returns the result from the first tool that can handle the query
        """
        try:
            # Try each registered tool in priority order
            for tool_name, tool_config in self._tool_registry.items():
                try:
                    checker = tool_config['checker']
                    handler = tool_config['handler']
                   
                    # Check if this tool should handle the query
                    if checker(user_question):
                        logger.info(f"Using MCP tool: {tool_name}")
                        result = handler(user_question, context)
                        if result:  # Only return non-empty results
                            return result
                except Exception as tool_error:
                    logger.warning(f"Tool {tool_name} failed: {tool_error}")
                    continue
           
            return None  # No tool could handle the query
           
        except Exception as e:
            logger.error(f"Error in MCP tools: {e}")
            return None
   
    def list_available_tools(self) -> Dict[str, str]:
        """Return a list of all available tools and their descriptions"""
        return {name: config['description'] for name, config in self._tool_registry.items()}
   
    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        """Get information about a specific tool"""
        return self._tool_registry.get(tool_name)
   
    # Legacy methods for backward compatibility
    def web_search_direct(self, query: str) -> str:
        """Direct web search (legacy method)"""
        return self.web_search(query)
   
    def weather_search_direct(self, city: str) -> str:
        """Direct weather search (legacy method)"""
        return self.weather_search(city)
   
    def news_search_direct(self, query: str = "", is_headlines: bool = True) -> str:
        """Direct news search (legacy method)"""
        return self.news_search(query, is_headlines)
 
class MessageFormatter:
    @staticmethod
    def clean_markdown_text(text):
        if not text:
            return text
        # Strip chain-of-thought style tags and obvious reasoning lead-ins
        try:
            import re
            # Remove <think>...</think> blocks if model leaked internal reasoning
            text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
            # Remove stray HTML-like tags that may break frontend rendering
            text = re.sub(r"<\/?[a-zA-Z][^>]*>", "", text)
            # Trim common meta lead-ins like "Okay," "Let's" when leaking planning
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            pruned = []
            dropping = True
            for ln in lines:
                if re.match(r"^[A-Za-z0-9].*", ln) and not re.match(r"^(Okay|Let\'s|Let us|I will|I cannot|The user|We need to|Reasoning|Analysis|Plan)\b", ln, flags=re.IGNORECASE):
                    dropping = False
                if not dropping:
                    pruned.append(ln)
            text = (" ".join(pruned) if pruned else text).strip()
        except Exception:
            # Best-effort only; continue with other cleanups
            pass
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]*)`', r'\1', text)
        text = re.sub(r'\*\*([^*]*?)\*\*', r'\1', text)
        text = re.sub(r'__([^_]*?)__', r'\1', text)
        text = re.sub(r'\*([^*]*?)\*', r'\1', text)
        text = re.sub(r'_([^_]*?)_', r'\1', text)
        text = re.sub(r'^#{1,6}\s*(.*)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'_+', '', text)
        text = text.replace('**', '').replace('__', '')
        return ' '.join(text.split()).strip()
 
    @staticmethod
    def truncate_sentences(text: str, max_sentences: int = 2) -> str:
        """Return only the first `max_sentences` sentences from text."""
        if not text or max_sentences <= 0:
            return text
        try:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            if not sentences:
                return text.strip()
            truncated = ' '.join(sentences[:max_sentences]).strip()
            return truncated or text.strip()
        except Exception:
            return text.strip()
    @staticmethod
    def get_current_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
 
class LLMTranslationService:
    """Translation service using GROQ Qwen3.3 32B for Arabic ↔ English translation"""
   
    def __init__(self):
        self.llm_service = None  # Will be initialized lazily to avoid circular imports
        self.cache = {}  # In-memory cache as fallback
        self.redis_cache = None  # Redis cache (will be initialized if available)
        self._init_redis_cache()
        self.translation_prompts = {
            "ar_to_en": """You are a professional Arabic to English translator. Translate the following Arabic text to clear, natural English while preserving the exact meaning, tone, and context. Only return the translation, no explanations.
 
Arabic: {text}
 
English:""",
           
            "en_to_ar": """You are a professional English to Arabic translator. Translate the following English text to clear, natural Arabic while preserving the exact meaning, tone, and context. Only return the translation, no explanations.
 
English: {text}
 
Arabic:"""
        }
   
    def _init_redis_cache(self):
        """Initialize Redis cache if available"""
        try:
            import redis
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            redis_db = int(os.getenv('REDIS_DB', '0'))
           
            self.redis_cache = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # Test connection
            self.redis_cache.ping()
            logger.info("✅ Redis cache initialized successfully")
        except ImportError:
            logger.warning("⚠️ Redis not installed. Using in-memory cache only.")
            self.redis_cache = None
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. Using in-memory cache only.")
            self.redis_cache = None
   
    def _get_llm_service(self):
        """Lazy initialization of LLM service to avoid circular imports"""
        if self.llm_service is None:
            self.llm_service = get_service_manager().get_llm_service()
        return self.llm_service
   
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of input text
        Returns: (language_code, confidence_score)
        """
        if not text or len(text.strip()) < 2:
            return "en", 0.5  # Default to English for very short text
       
        text = text.strip()
       
        # Check for Arabic characters
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([char for char in text if char.isalpha()])
       
        if total_chars == 0:
            return "en", 0.5  # Default for non-alphabetic text
       
        arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0
       
        # If more than 30% Arabic characters, likely Arabic
        if arabic_ratio > 0.3:
            confidence = min(0.9, 0.5 + arabic_ratio)
            return "ar", confidence
       
        # Use langdetect if available for better detection
        if LANGDETECT_AVAILABLE:
            try:
                detected_langs = detect_langs(text)
                for lang_obj in detected_langs:
                    if lang_obj.lang == "ar" and lang_obj.prob > Config.LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD:
                        return "ar", lang_obj.prob
                    elif lang_obj.lang == "en" and lang_obj.prob > Config.LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD:
                        return "en", lang_obj.prob
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
       
        # Default to English if uncertain
        return "en", 0.6
   
    def _get_cached_translation(self, text: str, direction: str) -> Optional[str]:
        """Get cached translation from Redis or in-memory cache"""
        cache_key = f"translation:{direction}:{hash(text)}"
       
        # Try Redis first
        if self.redis_cache:
            try:
                cached = self.redis_cache.get(cache_key)
                if cached:
                    logger.info(f"Redis cache hit ({direction.upper()})")
                    return cached
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")
       
        # Fallback to in-memory cache
        if cache_key in self.cache:
            logger.info(f"Memory cache hit ({direction.upper()})")
            return self.cache[cache_key]
       
        return None
   
    def _cache_translation(self, text: str, translation: str, direction: str):
        """Cache translation in Redis and in-memory cache"""
        cache_key = f"translation:{direction}:{hash(text)}"
        ttl = int(os.getenv('TRANSLATION_CACHE_TTL', '3600'))  # 1 hour default
       
        # Cache in Redis
        if self.redis_cache:
            try:
                self.redis_cache.setex(cache_key, ttl, translation)
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
       
        # Cache in memory (with size limit)
        self.cache[cache_key] = translation
        if len(self.cache) > 1000:  # Limit memory cache size
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.cache.keys())[:100]
            for key in oldest_keys:
                del self.cache[key]
 
    def translate_arabic_to_english(self, arabic_text: str) -> str:
        """Translate Arabic text to English using GROQ LLM"""
        if not arabic_text or not arabic_text.strip():
            return arabic_text
       
        # Check cache first
        cached_translation = self._get_cached_translation(arabic_text, "ar_to_en")
        if cached_translation:
            return cached_translation
       
        try:
            prompt = self.translation_prompts["ar_to_en"].format(text=arabic_text)
            messages = [{"role": "user", "content": prompt}]
           
            translation = self._get_llm_service().generate_text(
                messages=messages,
                model=Config.TRANSLATION_MODEL,
                max_tokens=Config.TRANSLATION_MAX_TOKENS,
                temperature=Config.TRANSLATION_TEMPERATURE,
                timeout=Config.TRANSLATION_TIMEOUT
            )
           
            # Clean up the translation
            translation = translation.strip()
            translation = self._strip_reasoning(translation, target_lang="en")
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1]
           
            # Cache the result
            self._cache_translation(arabic_text, translation, "ar_to_en")
            logger.info(f"Translated AR→EN: {arabic_text[:50]}... → {translation[:50]}...")
           
            return translation
           
        except Exception as e:
            logger.error(f"Arabic to English translation failed: {e}")
            return f"[Translation Error] {arabic_text}"
   
    def translate_english_to_arabic(self, english_text: str) -> str:
        """Translate English text to Arabic using GROQ LLM"""
        if not english_text or not english_text.strip():
            return english_text
       
        # Check cache first
        cached_translation = self._get_cached_translation(english_text, "en_to_ar")
        if cached_translation:
            return cached_translation
       
        try:
            prompt = self.translation_prompts["en_to_ar"].format(text=english_text)
            messages = [{"role": "user", "content": prompt}]
           
            translation = self._get_llm_service().generate_text(
                messages=messages,
                model=Config.TRANSLATION_MODEL,
                max_tokens=Config.TRANSLATION_MAX_TOKENS,
                temperature=Config.TRANSLATION_TEMPERATURE,
                timeout=Config.TRANSLATION_TIMEOUT
            )
           
            # Clean up the translation
            translation = translation.strip()
            translation = self._strip_reasoning(translation, target_lang="ar")
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1]
           
            # Cache the result
            self._cache_translation(english_text, translation, "en_to_ar")
            logger.info(f"Translated EN→AR: {english_text[:50]}... → {translation[:50]}...")
           
            return translation
           
        except Exception as e:
            logger.error(f"English to Arabic translation failed: {e}")
            return english_text  # Return original English if translation fails
   
    def translate_with_fallback(self, text: str, direction: str) -> str:
        """Translate with fallback handling"""
        try:
            if direction == "ar_to_en":
                return self.translate_arabic_to_english(text)
            elif direction == "en_to_ar":
                return self.translate_english_to_arabic(text)
            else:
                logger.error(f"Unknown translation direction: {direction}")
                return text
        except Exception as e:
            logger.error(f"Translation failed ({direction}): {e}")
            if direction == "ar_to_en":
                return f"[Translation Error] {text}"
            else:
                return text  # Return English if Arabic translation fails
   
    def is_arabic(self, text: str) -> bool:
        """Quick check if text contains Arabic characters"""
        if not text:
            return False
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        return arabic_chars > 0
   
    def clear_cache(self):
        """Clear translation cache"""
        self.cache.clear()
        logger.info("Translation cache cleared")
 
    def _strip_reasoning(self, text: str, target_lang: str) -> str:
        """Remove common chain-of-thought artifacts and keep the likely translation body.
        For EN->AR, keep content from the first Arabic character onward.
        For AR->EN, remove <think> blocks and leading meta lines like 'Okay,' or 'Let's'.
        """
        if not text:
            return text
        try:
            import re
            # Remove <think>...</think> blocks
            text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
            # Trim whitespace
            text = text.strip()
            if target_lang == "ar":
                # Keep from first Arabic char onward and drop stray non-Arabic symbols
                for idx, ch in enumerate(text):
                    if '\u0600' <= ch <= '\u06FF':
                        ar = text[idx:].strip()
                        # Remove any characters outside Arabic block, digits, whitespace and common punctuation
                        ar = re.sub(r"[^\u0600-\u06FF0-9\s\.,!\?;:\-]+", "", ar)
                        # Collapse whitespace
                        ar = re.sub(r"\s+", " ", ar).strip()
                        return ar
                return text
            else:
                # Remove obvious reasoning lead-ins in English
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                pruned = []
                dropping = True
                for ln in lines:
                    # stop dropping when the line looks like a sentence that could be a translation
                    if re.match(r"^[A-Za-z0-9].*", ln) and not re.match(r"^(Okay|Let\'s|Let us|I need to|I will|The user|We need to|Reasoning|Analysis)\b", ln, flags=re.IGNORECASE):
                        dropping = False
                    if not dropping:
                        pruned.append(ln)
                return (" ".join(pruned) if pruned else text).strip()
        except Exception:
            return text