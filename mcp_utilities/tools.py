import aiohttp
import httpx
import os
import re
import json
import copy
from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple
from mcp_utilities.config import ConfigManager
from mcp_utilities.logger import Logger
 
class Tool(ABC):
    def __init__(self, config: ConfigManager, logger: Logger):
        self.config = config
        self.logger = logger
 
    @abstractmethod
    async def execute(self, **kwargs) -> dict:
        pass
 
    def log_tool_call(self, tool_name: str, parameters: dict, result: dict = None, error: str = None, confidence: float = None):
        # Quiet verbose stdout; keep error logging and success info via logger
        if error:
            self.logger.error(f"Tool '{tool_name}' failed: {error}")
        elif result:
            self.logger.info(f"Tool '{tool_name}' executed successfully")
 
class WebSearchTool(Tool):
    async def execute(self, query: str, search_depth: str = "basic", max_results: int = 5,
                     include_answer: bool = True, include_raw_content: bool = False) -> dict:
        parameters = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content
        }
       
        if not self.config.tavily_key:
            error_msg = "TAVILY_API_KEY not set"
            self.log_tool_call("web_search", parameters, error=error_msg)
            return {"error": error_msg}
           
        try:
            # Validate search_depth parameter
            if search_depth not in ["basic", "advanced"]:
                search_depth = "basic"
           
            # Validate max_results parameter
            if max_results < 0 or max_results > 20:
                max_results = 5
           
            # Build the request payload according to Tavily API spec
            payload = {
                "api_key": self.config.tavily_key,
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
            }
           
            timeout = aiohttp.ClientTimeout(total=6)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                resp = await session.post(
                    "https://api.tavily.com/search",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
               
                # Get response text for debugging
                response_text = await resp.text()
               
                if resp.status != 200:
                    error_msg = f"Web search failed: HTTP {resp.status} - {response_text}"
                    self.log_tool_call("web_search", parameters, error=error_msg)
                    return {"error": error_msg}
               
                result = await resp.json()
                self.log_tool_call("web_search", parameters, result=result)
                return result
               
        except Exception as e:
            error_msg = f"Web search failed: {str(e)}"
            self.log_tool_call("web_search", parameters, error=error_msg)
            return {"error": error_msg}
 
class WeatherTool(Tool):
    async def execute(self, city: str) -> dict:
        # Sanitize city input (strip punctuation, fix common typos)
        def normalize(txt: str) -> str:
            t = (txt or "").strip().strip(" ?!.,;:")
            # Common misspellings
            fixes = {"londonn": "London"}
            low = t.lower()
            if low in fixes:
                return fixes[low]
            return t
        city = normalize(city)
        parameters = {"city": city}
        if not self.config.weather_key:
            error_msg = "WEATHER_API_KEY not set"
            self.log_tool_call("get_weather", parameters, error=error_msg)
            return {"error": error_msg}
        params = {"appid": self.config.weather_key, "q": city, "units": "metric"}
        try:
            timeout = aiohttp.ClientTimeout(total=6)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                resp = await session.get("https://api.openweathermap.org/data/2.5/weather", params=params)
                resp.raise_for_status()
                result = await resp.json()
                self.log_tool_call("get_weather", parameters, result=result)
                return result
        except Exception as e:
            error_msg = f"Failed to get weather: {str(e)}"
            self.log_tool_call("get_weather", parameters, error=error_msg)
            return {"error": error_msg}
 
class TopHeadlinesTool(Tool):
    async def execute(self, page_size: int = 5, country: str = "us") -> dict:
        parameters = {"page_size": page_size, "country": country}
        if not self.config.news_key:
            error_msg = "NEWS_API_KEY not set"
            self.log_tool_call("get_top_headlines", parameters, error=error_msg)
            return {"error": error_msg}
        params = {"apiKey": self.config.news_key, "pageSize": min(page_size, 100), "country": country}
        try:
            timeout = aiohttp.ClientTimeout(total=6)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                resp = await session.get("https://newsapi.org/v2/top-headlines", params=params)
                resp.raise_for_status()
                result = await resp.json()
                self.log_tool_call("get_top_headlines", parameters, result=result)
                return result
        except Exception as e:
            error_msg = f"Failed to fetch headlines: {str(e)}"
            self.log_tool_call("get_top_headlines", parameters, error=error_msg)
            return {"error": error_msg}
 
class SearchNewsTool(Tool):
    async def execute(self, query: str, page_size: int = 5) -> dict:
        parameters = {"query": query, "page_size": page_size}
        if not self.config.news_key:
            error_msg = "NEWS_API_KEY not set"
            self.log_tool_call("search_news", parameters, error=error_msg)
            return {"error": error_msg}
        # Bias NewsAPI Everything endpoint toward relevancy and search within title/description
        params = {
            "apiKey": self.config.news_key,
            "q": query,
            "searchIn": "title,description",
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": min(page_size, 100),
        }
        try:
            timeout = aiohttp.ClientTimeout(total=6)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                resp = await session.get("https://newsapi.org/v2/everything", params=params)
                resp.raise_for_status()
                result = await resp.json()
                # Attach echo of search query for downstream formatting filters
                if isinstance(result, dict):
                    result["_search_query"] = query
                self.log_tool_call("search_news", parameters, result=result)
                return result
        except Exception as e:
            error_msg = f"Failed to search news: {str(e)}"
            self.log_tool_call("search_news", parameters, error=error_msg)
            return {"error": error_msg}


class LeaveCalculatorTool(Tool):
    """Tool that orchestrates leave analysis and validation with hint overrides."""

    def __init__(self, config: ConfigManager, logger: Logger):
        super().__init__(config, logger)
        self._session: Optional[httpx.AsyncClient] = None
        self._request_template_text = self._load_template_text("request_adaptive_card.json")
        self._response_template_json = self._load_template_json("response_adaptive_card.json")
        self._api_base = os.getenv("LEAVE_CALCULATOR_API_URL", "http://localhost:8000")
        self._groq_api_key = os.getenv("GROQ_API_KEY")

    def _load_template_text(self, filename: str) -> str:
        path = Path(__file__).resolve().parent / "templates" / filename
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read()
        except Exception as exc:
            raise FileNotFoundError(f"Unable to load template {filename}: {exc}")

    def _load_template_json(self, filename: str) -> Dict[str, Any]:
        path = Path(__file__).resolve().parent / "templates" / filename
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            raise FileNotFoundError(f"Unable to load template {filename}: {exc}")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._session is None:
            timeout = httpx.Timeout(10.0, connect=5.0)
            self._session = httpx.AsyncClient(timeout=timeout)
        return self._session

    async def execute(
        self,
        mode: str,
        query: Optional[str] = None,
        employee_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        leave_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        parameters = {
            "mode": mode,
            "query": query,
            "employee_id": employee_id,
            "start_date": start_date,
            "end_date": end_date,
            "leave_type": leave_type,
        }

        try:
            if mode == "analyze":
                return await self._analyze_query(
                    query,
                    employee_id,
                    start_hint=start_date,
                    end_hint=end_date,
                    leave_type_hint=leave_type,
                )

            if mode == "validate":
                if not all([employee_id, start_date, end_date, leave_type]):
                    raise ValueError("Missing required parameters for validation")
                return await self._validate_leave(employee_id, start_date, end_date, leave_type)

            raise ValueError("Unknown mode for LeaveCalculatorTool")
        except Exception as exc:
            error_msg = str(exc)
            self.log_tool_call("leave_calculator", parameters, error=error_msg)
            return {"error": error_msg}

    async def _analyze_query(
        self,
        query: Optional[str],
        employee_id: Optional[str],
        *,
        start_hint: Optional[str] = None,
        end_hint: Optional[str] = None,
        leave_type_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not query:
            raise ValueError("Query text is required for analyze mode")

        extracted = await self._extract_with_llm(query)

        overrides: Dict[str, Optional[str]] = {
            "start_date": self._normalize_date_hint(start_hint),
            "end_date": self._normalize_date_hint(end_hint),
            "leave_type": self._normalize_leave_type(leave_type_hint) if leave_type_hint else None,
        }

        merge_order = ["start_date", "end_date", "leave_type"]
        for key in merge_order:
            value = overrides.get(key)
            if value:
                extracted[key] = value

        template = Template(self._request_template_text)
        card_str = template.safe_substitute(
            start_date=extracted.get("start_date", ""),
            end_date=extracted.get("end_date", ""),
            leave_type=extracted.get("leave_type", ""),
        )
        card_payload = json.loads(card_str)

        return {
            "type": "card",
            "template": "leave_request",
            "payload": card_payload,
            "metadata": {
                "employee_id": employee_id,
                "extracted": extracted,
                "hints": {k: v for k, v in overrides.items() if v},
            },
        }

    async def _validate_leave(
        self,
        employee_id: str,
        start_date: str,
        end_date: str,
        leave_type: str,
    ) -> Dict[str, Any]:
        client = await self._get_client()
        url = f"{self._api_base.rstrip('/')}/leave/validate"
        payload = {
            "employee_id": int(employee_id),
            "start_date": start_date,
            "end_date": end_date,
            "leave_type": leave_type,
        }

        self.logger.info(f"Sending leave validation request to {url} with payload: {payload}")
       
        resp = await client.post(url, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            content_type = exc.response.headers.get("content-type", "")
            if content_type.startswith("application/json"):
                detail = exc.response.json()
                self.logger.error(f"Leave validation API error (JSON): {detail}")
            else:
                detail = exc.response.text
                self.logger.error(f"Leave validation API error (Text): {detail}")
           
            # Re-raise the error - NO FALLBACKS!
            raise RuntimeError(f"Leave validation failed: {detail}") from exc

        data = resp.json()
        self.logger.info(f"Leave validation API response: {data}")
        return self._build_validation_card(data)

    def _build_validation_card(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Use ONLY database data - NO defaults or fallbacks
        enriched = copy.deepcopy(data)
        eligible = bool(enriched.get("eligible", False))
       
        # Only set computed values, don't override database values
        enriched.setdefault("employee_name", f"Employee #{enriched.get('employee_id', 'Unknown')}")
        enriched.setdefault("employee_info", f"Requested {enriched.get('leave_type', 'Unknown').title()} leave")
        enriched.setdefault("status_message", enriched.get("message", "Leave status unavailable"))
        enriched.setdefault("status_style", "good" if eligible else "attention")
        enriched.setdefault("remaining_color", "Good" if eligible else "Attention")

        base_card = self._response_template_json.get("leave_calculator_with_dates_card", {})
        card = copy.deepcopy(base_card)
        filled_card = self._replace_placeholders(card, enriched, prefix="{", suffix="}")

        return {
            "type": "card",
            "template": "leave_response",
            "payload": filled_card,
            "metadata": enriched,
        }

    async def _extract_with_llm(self, query: str) -> Dict[str, str]:
        if not self._groq_api_key:
            raise ValueError("GROQ_API_KEY is required for leave extraction - no fallbacks allowed")

        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self._groq_api_key}",
            "Content-Type": "application/json",
        }
        today = datetime.utcnow().date().isoformat()
        system_prompt = (
            "You analyse employee leave requests. Extract the intended start date, end date, and leave type."
            "\n- Use ISO format YYYY-MM-DD.\n"
            f"- Assume unspecified start dates refer to today ({today}).\n"
            "- If only one day is mentioned (e.g. 'tomorrow'), set start and end to that date.\n"
            "- Leave type should be one of: sick leave, annual leave, flexi leave, unpaid leave.\n"
            "- If leave type not mentioned, return an empty string.\n"
            "Return a JSON object with keys start_date, end_date, leave_type, and confidence (0-1)."
        )
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "temperature": 0.1,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }

        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10.0
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        extracted = json.loads(content)
       
        # Validate the response has required fields
        if not all(key in extracted for key in ["start_date", "end_date", "leave_type", "confidence"]):
            raise ValueError("LLM response missing required fields")
           
        return extracted

    def _normalize_date_hint(self, value: Optional[str]) -> Optional[str]:
        if not value or not isinstance(value, str):
            return None
        candidate = value.strip()
        if not candidate:
            return None
        normalized = candidate.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
            return parsed.date().isoformat()
        except ValueError:
            if len(candidate) >= 10:
                return candidate[:10]
        return candidate

    def _fallback_extract(self, query: str) -> Dict[str, str]:
        lowered = query.lower()
        leave_type = ""
        for candidate in [
            "sick leave",
            "annual leave",
            "flexi leave",
            "unpaid leave",
            "sick day",
            "sick",
            "vacation",
            "pto",
        ]:
            if candidate in lowered:
                leave_type = self._normalize_leave_type(candidate)
                break

        start_date, end_date = self._extract_dates_from_text(query)

        return {
            "start_date": start_date,
            "end_date": end_date,
            "leave_type": leave_type,
            "confidence": 0.0,
        }

    def _extract_dates_from_text(self, text: str) -> Tuple[str, str]:
        today = datetime.utcnow().date()
        matches: List[Tuple[int, date]] = []
        lowered = text.lower()

        relative_terms = [
            ("day after tomorrow", 2),
            ("tomorrow", 1),
            ("today", 0),
            ("yesterday", -1),
        ]
        for phrase, delta in relative_terms:
            for match in re.finditer(rf"\b{re.escape(phrase)}\b", lowered):
                matches.append((match.start(), today + timedelta(days=delta)))

        iso_pattern = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")
        for match in iso_pattern.finditer(text):
            parsed = self._try_parse_date(match.group(), today)
            if parsed:
                matches.append((match.start(), parsed))

        numeric_pattern = re.compile(r"\b(\d{1,2})[\/-](\d{1,2})(?:[\/-](\d{2,4}))?\b")
        for match in numeric_pattern.finditer(text):
            parsed = self._parse_numeric_date(match.groups(), today)
            if parsed:
                matches.append((match.start(), parsed))

        month_pattern = re.compile(
            r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
            r"january|february|march|april|june|july|august|september|october|november|december))"
            r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b",
            re.IGNORECASE,
        )
        for match in month_pattern.finditer(text):
            parsed = self._parse_month_text(match.group(), today)
            if parsed:
                matches.append((match.start(), parsed))

        # Handle bare day numbers (e.g., "23" meaning 23rd of current/next month)
        bare_day_pattern = re.compile(r"\bon\s+(\d{1,2})(?:st|nd|rd|th)?\b")
        for match in bare_day_pattern.finditer(lowered):
            day_num = int(match.group(1))
            if 1 <= day_num <= 31:
                # Try current month first
                try:
                    target_date = today.replace(day=day_num)
                    # If date is in the past, use next month
                    if target_date < today:
                        if today.month == 12:
                            target_date = date(today.year + 1, 1, day_num)
                        else:
                            target_date = date(today.year, today.month + 1, day_num)
                    matches.append((match.start(), target_date))
                except ValueError:
                    # Invalid day for current month, try next month
                    try:
                        if today.month == 12:
                            target_date = date(today.year + 1, 1, day_num)
                        else:
                            target_date = date(today.year, today.month + 1, day_num)
                        matches.append((match.start(), target_date))
                    except ValueError:
                        # Skip invalid dates
                        pass

        if not matches:
            return "", ""

        matches.sort(key=lambda entry: entry[0])
        return matches[0][1].isoformat(), matches[-1][1].isoformat()

    def _try_parse_date(self, value: str, today: date) -> Optional[date]:
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None

    def _parse_numeric_date(self, components: Tuple[str, ...], today: date) -> Optional[date]:
        month, day, year = components
        try:
            month_int = int(month)
            day_int = int(day)
        except (TypeError, ValueError):
            return None

        if year is None:
            year_int = today.year
        else:
            year = year.strip()
            if len(year) == 2:
                year_int = 2000 + int(year)
            else:
                try:
                    year_int = int(year)
                except ValueError:
                    year_int = today.year

        try:
            candidate = date(year_int, month_int, day_int)
        except ValueError:
            return None

        if candidate < today and year is None and (today - candidate).days > 180:
            try:
                candidate = candidate.replace(year=candidate.year + 1)
            except ValueError:
                return candidate

        return candidate

    def _parse_month_text(self, raw: str, today: date) -> Optional[date]:
        cleaned = raw.replace(",", " ")
        cleaned = re.sub(r"(\d)(st|nd|rd|th)", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        formats = ["%B %d %Y", "%b %d %Y", "%B %d", "%b %d"]
        for fmt in formats:
            try:
                parsed = datetime.strptime(cleaned.title(), fmt)
                if "%Y" not in fmt:
                    parsed = parsed.replace(year=today.year)
                    if parsed.date() < today and (today - parsed.date()).days > 180:
                        parsed = parsed.replace(year=parsed.year + 1)
                return parsed.date()
            except ValueError:
                continue
        return None

    def _normalize_leave_type(self, value: Optional[str]) -> str:
        if not value:
            return ""
        val = value.strip().lower()
        mapping = {
            "sick": "sick leave",
            "sick day": "sick leave",
            "annual": "annual leave",
            "vacation": "annual leave",
            "flexi": "flexi leave",
            "unpaid": "unpaid leave",
        }
        return mapping.get(val, value.strip().lower())

    def _replace_placeholders(self, node: Any, values: Dict[str, Any], prefix: str, suffix: str):
        def replace(value: Any) -> Any:
            if isinstance(value, str):
                result = value
                for key, val in values.items():
                    placeholder = f"{prefix}{key}{suffix}"
                    result = result.replace(placeholder, str(val if val is not None else ""))
                return result
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, dict):
                return {k: replace(v) for k, v in value.items()}
            return value

        return replace(node)

    async def aclose(self):
        if self._session:
            await self._session.aclose()
            self._session = None


class LeaveCalculatorTool(Tool):
    """Tool that orchestrates leave analysis and validation with hint overrides."""

    def __init__(self, config: ConfigManager, logger: Logger):
        super().__init__(config, logger)
        self._session: Optional[httpx.AsyncClient] = None
        self._request_template_text = self._load_template_text("request_adaptive_card.json")
        self._response_template_json = self._load_template_json("response_adaptive_card.json")
        self._api_base = os.getenv("LEAVE_CALCULATOR_API_URL", "http://localhost:8000")
        self._groq_api_key = os.getenv("GROQ_API_KEY")

    def _load_template_text(self, filename: str) -> str:
        """Load template file as text for string substitution."""
        path = Path(__file__).resolve().parent / "templates" / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as exc:
            self.logger.error(f"Failed to load template {filename}: {exc}")
            return "{}"

    def _load_template_json(self, filename: str) -> Dict[str, Any]:
        """Load template file as JSON."""
        path = Path(__file__).resolve().parent / "templates" / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            self.logger.error(f"Failed to load JSON template {filename}: {exc}")
            return {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._session is None:
            self._session = httpx.AsyncClient(timeout=30.0)
        return self._session

    async def execute(
        self,
        mode: str,
        query: Optional[str] = None,
        employee_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        leave_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main execution method for leave calculator tool.
       
        Args:
            mode: "analyze" (build form) or "validate" (check eligibility)
            query: Natural language query for analysis mode
            employee_id: Employee ID (required for validation)
            start_date: ISO date string (YYYY-MM-DD)
            end_date: ISO date string (YYYY-MM-DD)
            leave_type: Type of leave (sick leave, annual leave, etc.)
       
        Returns:
            Dict with "type", "payload", "metadata" for adaptive cards
            or "error" key if something went wrong
        """
        parameters = {
            "mode": mode,
            "query": query,
            "employee_id": employee_id,
            "start_date": start_date,
            "end_date": end_date,
            "leave_type": leave_type,
        }

        try:
            if mode == "analyze":
                return await self._analyze_query(
                    query,
                    employee_id,
                    start_hint=start_date,
                    end_hint=end_date,
                    leave_type_hint=leave_type,
                )
            elif mode == "validate":
                if not all([employee_id, start_date, end_date, leave_type]):
                    return {"error": "Missing required fields for validation"}
                return await self._validate_leave(
                    employee_id,
                    start_date,
                    end_date,
                    leave_type,
                )
            else:
                return {"error": f"Unknown mode: {mode}"}
        except Exception as exc:
            self.logger.error(f"LeaveCalculatorTool.execute error: {exc}")
            self.log_tool_call("leave_calculator", parameters, error=str(exc))
            return {"error": str(exc)}

    async def _analyze_query(
        self,
        query: Optional[str],
        employee_id: Optional[str],
        *,
        start_hint: Optional[str] = None,
        end_hint: Optional[str] = None,
        leave_type_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze natural language query to extract leave details and build request form.
       
        Uses LLM to extract dates and leave type, then generates an adaptive card.
        """
        if not query:
            return {"error": "Query is required for analysis mode"}

        # Extract using LLM
        extracted = await self._extract_with_llm(query)

        # Override with hints if provided
        overrides: Dict[str, Optional[str]] = {
            "start_date": self._normalize_date_hint(start_hint),
            "end_date": self._normalize_date_hint(end_hint),
            "leave_type": self._normalize_leave_type(leave_type_hint) if leave_type_hint else None,
        }

        merge_order = ["start_date", "end_date", "leave_type"]
        for key in merge_order:
            if overrides.get(key):
                extracted[key] = overrides[key]

        # Build adaptive card from template
        template = Template(self._request_template_text)
        card_str = template.safe_substitute(
            start_date=extracted.get("start_date", ""),
            end_date=extracted.get("end_date", ""),
            leave_type=extracted.get("leave_type", ""),
        )
        card_payload = json.loads(card_str)

        return {
            "type": "card",
            "template": "leave_request",
            "payload": card_payload,
            "metadata": {
                "employee_id": employee_id,
                "extracted": extracted,
                "hints": {k: v for k, v in overrides.items() if v},
            },
        }

    async def _validate_leave(
        self,
        employee_id: str,
        start_date: str,
        end_date: str,
        leave_type: str,
    ) -> Dict[str, Any]:
        """
        Validate leave request against employee's available balance.
       
        Calls the Leave Calculator API and builds a response card.
        """
        client = await self._get_client()
        url = f"{self._api_base.rstrip('/')}/leave/validate"
        payload = {
            "employee_id": int(employee_id),
            "start_date": start_date,
            "end_date": end_date,
            "leave_type": leave_type,
        }

        resp = await client.post(url, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            self.logger.error(f"Leave validation API error: {exc}")
            return {"error": f"API error: {exc.response.status_code}"}

        data = resp.json()
        return self._build_validation_card(data)

    def _build_validation_card(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build adaptive card response from API validation data.
       
        Enriches the data with display-friendly fields and uses template.
        """
        enriched = copy.deepcopy(data)
        eligible = bool(enriched.get("eligible"))
        sick_hours = enriched.get("sick_leave_hours", 0)
        vacation_hours = enriched.get("vacation_hours", enriched.get("available_hours", 0))

        # Add display fields
        enriched.setdefault("employee_name", f"Employee #{enriched.get('employee_id')}")
        enriched.setdefault("employee_info", f"Requested {enriched.get('leave_type', '').title()} leave")
        enriched.setdefault("status_message", enriched.get("message", "Leave status unavailable"))
        enriched.setdefault("status_style", "good" if eligible else "attention")
        enriched.setdefault("remaining_color", "Good" if eligible else "Attention")
        enriched.setdefault("vacation_hours", vacation_hours)
        enriched.setdefault("sick_hours", sick_hours)
        enriched.setdefault("flexi_hours", vacation_hours)
        enriched.setdefault("unpaid_hours", vacation_hours)
        enriched.setdefault("requested_days", enriched.get("requested_days", 0))
        enriched.setdefault("required_hours", enriched.get("requested_hours", 0))
        enriched.setdefault("remaining_hours", enriched.get("remaining_hours", 0))
        enriched.setdefault("remaining_days", enriched.get("remaining_days", 0))
        enriched.setdefault("shortage_hours", enriched.get("shortage_hours", 0))
        enriched.setdefault("shortage_days", enriched.get("shortage_days", 0))

        # Load response template and fill placeholders
        base_card = self._response_template_json.get("leave_calculator_with_dates_card", {})
        card = copy.deepcopy(base_card)
        filled_card = self._replace_placeholders(card, enriched, prefix="{", suffix="}")

        return {
            "type": "card",
            "template": "leave_response",
            "payload": filled_card,
            "metadata": enriched,
        }

    async def _extract_with_llm(self, query: str) -> Dict[str, str]:
        """
        Use Groq LLM to extract dates and leave type from natural language.
       
        Falls back to simple heuristics if LLM fails.
        """
        if not self._groq_api_key:
            self.logger.warning("GROQ_API_KEY not set, using fallback extraction")
            return self._fallback_extract(query)

        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self._groq_api_key}",
            "Content-Type": "application/json",
        }
        today = datetime.utcnow().date().isoformat()
        system_prompt = (
            "You analyse employee leave requests. Extract the intended start date, end date, and leave type."
            "\n- Use ISO format YYYY-MM-DD.\n"
            "- Assume unspecified start dates refer to today ({today}).\n"
            "- If only one day is mentioned (e.g. 'tomorrow'), set start and end to that date.\n"
            "- Leave type should be one of: sick leave, annual leave, flexi leave, unpaid leave.\n"
            "- If leave type not mentioned, return an empty string.\n"
            "Return a JSON object with keys start_date, end_date, leave_type, and confidence (0-1)."
        ).format(today=today)
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "temperature": 0.1,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }

        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            self.logger.error(f"LLM extraction API error: {exc}")
            return self._fallback_extract(query)

        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            self.logger.error(f"LLM response parsing error: {exc}")
            return self._fallback_extract(query)

        start_date = str(parsed.get("start_date", ""))
        end_date = str(parsed.get("end_date", ""))
        leave_type = self._normalize_leave_type(parsed.get("leave_type", ""))

        if not start_date or not end_date:
            self.logger.warning("LLM did not extract dates, using fallback")
            return self._fallback_extract(query)

        return {
            "start_date": start_date,
            "end_date": end_date,
            "leave_type": leave_type,
            "confidence": parsed.get("confidence", 0.0),
        }

    def _normalize_date_hint(self, value: Optional[str]) -> Optional[str]:
        """Normalize and validate date hint."""
        if not value or not isinstance(value, str):
            return None
        candidate = value.strip()
        if not candidate:
            return None
        normalized = candidate.replace("Z", "+00:00")
        try:
            datetime.fromisoformat(normalized)
        except ValueError:
            self.logger.warning(f"Invalid date hint: {candidate}")
            return None
        return candidate

    def _fallback_extract(self, query: str) -> Dict[str, str]:
        """Simple fallback extraction using text patterns."""
        lowered = query.lower()
        leave_type = ""
        for candidate in [
            "sick leave",
            "annual leave",
            "flexi leave",
            "unpaid leave",
            "sick day",
            "sick",
            "vacation",
            "pto",
        ]:
            if candidate in lowered:
                leave_type = self._normalize_leave_type(candidate)
                break

        start_date, end_date = self._extract_dates_from_text(query)

        return {
            "start_date": start_date,
            "end_date": end_date,
            "leave_type": leave_type,
            "confidence": 0.0,
        }

    def _extract_dates_from_text(self, text: str) -> Tuple[str, str]:
        """Extract dates from text using simple heuristics."""
        today = datetime.utcnow().date()
        matches: List[Tuple[int, date]] = []
        lowered = text.lower()

        # Look for relative dates
        if "today" in lowered:
            matches.append((lowered.index("today"), today))
        if "tomorrow" in lowered:
            matches.append((lowered.index("tomorrow"), today + timedelta(days=1)))
        if "next week" in lowered:
            matches.append((lowered.index("next week"), today + timedelta(weeks=1)))

        # Sort by position in text
        matches.sort(key=lambda x: x[0])

        if not matches:
            # Default to tomorrow
            return (today + timedelta(days=1)).isoformat(), (today + timedelta(days=1)).isoformat()
        elif len(matches) == 1:
            # Single date mentioned - use as both start and end
            dt = matches[0][1].isoformat()
            return dt, dt
        else:
            # Two or more dates - use first two
            return matches[0][1].isoformat(), matches[1][1].isoformat()

    def _normalize_leave_type(self, value: Optional[str]) -> str:
        """Normalize leave type to standard format."""
        if not value:
            return ""
        lowered = str(value).lower().strip()
       
        if "sick" in lowered:
            return "sick leave"
        elif "annual" in lowered or "vacation" in lowered or "pto" in lowered:
            return "annual leave"
        elif "flexi" in lowered:
            return "flexi leave"
        elif "unpaid" in lowered:
            return "unpaid leave"
       
        return lowered

    def _replace_placeholders(self, node: Any, values: Dict[str, Any], prefix: str, suffix: str):
        """Recursively replace placeholders in nested structure."""
        if isinstance(node, dict):
            result = {}
            for k, v in node.items():
                result[k] = self._replace_placeholders(v, values, prefix, suffix)
            return result
        elif isinstance(node, list):
            return [self._replace_placeholders(item, values, prefix, suffix) for item in node]
        elif isinstance(node, str):
            for key, val in values.items():
                placeholder = f"{prefix}{key}{suffix}"
                if placeholder in node:
                    node = node.replace(placeholder, str(val))
            return node
        else:
            return node

    async def aclose(self):
        """Close HTTP client session."""
        if self._session:
            await self._session.aclose()
            self._session = None
