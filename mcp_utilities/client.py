import asyncio
import aiohttp
import json
import logging
import re
import concurrent.futures
import requests
from typing import Dict, Any, Optional
import sys
from mcp_utilities.config import ConfigManager
from mcp_utilities.logger import Logger
from mcp_utilities.tools import (
    WebSearchTool,
    WeatherTool,
    TopHeadlinesTool,
    SearchNewsTool,
    LeaveCalculatorTool,
)
from groq import Groq
import os
 
logger = logging.getLogger(__name__)

# On Windows, prefer SelectorEventLoop to avoid Proactor destructor warnings
try:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass








class MCPClient:
    """Client for calling MCP tools directly with LLM-based tool selection"""
   
    def __init__(self):
        self.config = ConfigManager()
        self.logger = Logger(__name__)
       
        # Initialize tools
        self.web_search_tool = WebSearchTool(self.config, self.logger)
        self.weather_tool = WeatherTool(self.config, self.logger)
        self.top_headlines_tool = TopHeadlinesTool(self.config, self.logger)
        self.search_news_tool = SearchNewsTool(self.config, self.logger)
        self.leave_calculator_tool = LeaveCalculatorTool(self.config, self.logger)
       
        # Initialize Groq client
        self.groq_client = self._initialize_groq()
   
    def _initialize_groq(self):
        """Initialize Groq client with API key"""
        try:
            # Try to get API key from environment variable or config
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                # Try to get from config if available
                try:
                    api_key = self.config.get('groq_api_key')
                except:
                    pass
           
            if not api_key:
                logger.error("GROQ_API_KEY not found in environment variables or config")
                return None
           
            return Groq(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            return None
   
    async def decide_tool_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to decide which tool to use based on the query"""
        if not self.groq_client:
            logger.error("Groq client not initialized")
            return {"tool": "none", "error": "LLM service unavailable"}
       
        try:
            system_prompt = """You are an AI assistant that decides which tool to use based on user queries.
You have access to the following tools:
 
1. web_search - For general web searches, finding information, looking up facts, research queries
2. weather - For weather-related queries (temperature, forecast, climate information)
3. top_headlines - For getting latest news headlines without specific search terms
4. search_news - For searching specific news topics or events
5. leave_calculator - For employee leave planning and eligibility questions
 
Analyze the user query and respond with a JSON object containing:
- "tool": one of ["web_search", "weather", "top_headlines", "search_news", "leave_calculator", "none"]
- "parameters": relevant parameters for the selected tool
- "reasoning": brief explanation of your choice
 
For web_search: extract the search query and set max_results (default 5)
For weather: extract the city name from the query
For top_headlines: set page_size (default 5) and country (default "us")  
For search_news: extract the news search query and set page_size (default 5)
For leave_calculator: set mode="analyze" and include any inferred date windows or leave type
For none: when the query doesn't match any tool or is unclear

Examples:
- "I need 3 days off next week" -> {"tool": "leave_calculator", "parameters": {"query": "3 days off next week"}}
- "Check my leave balance" -> {"tool": "leave_calculator", "parameters": {"query": "leave balance"}}

 
Examples:
- "What's the weather in New York?" -> {"tool": "weather", "parameters": {"city": "New York"}, "reasoning": "User asking for weather information for a specific city"}
- "Search for latest AI developments" -> {"tool": "web_search", "parameters": {"query": "latest AI developments", "max_results": 5}, "reasoning": "General search query for information"}
- "Show me today's headlines" -> {"tool": "top_headlines", "parameters": {"page_size": 5, "country": "us"}, "reasoning": "User wants general news headlines"}
- "News about climate change" -> {"tool": "search_news", "parameters": {"query": "climate change", "page_size": 5}, "reasoning": "Specific news search query"}
 
Respond only with valid JSON."""
 
            user_message = f"User query: {query}"
           
            # Make the API call to Groq
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model="llama-3.1-8b-instant",  # Updated model
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
           
            response_text = chat_completion.choices[0].message.content
            decision = json.loads(response_text)
           
            logger.info(f"LLM decision for query '{query}': {decision}")
            return decision
           
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {"tool": "none", "error": "Invalid LLM response format"}
        except Exception as e:
            logger.error(f"LLM decision error: {e}")
            return {"tool": "none", "error": f"LLM service error: {str(e)}"}
   
    async def process_query_with_llm(self, query: str) -> str:
        """Process user query using LLM for tool selection"""
        # Get LLM decision
        decision = await self.decide_tool_with_llm(query)
       
        if "error" in decision:
            return "Unable to process your request, please try again."
       
        tool_name = decision.get("tool", "none")
        parameters = decision.get("parameters", {})
       
        try:
            if tool_name == "web_search":
                search_query = parameters.get("query", query)
                max_results = parameters.get("max_results", 5)
                result = await self.call_web_search(search_query, max_results)
                return self.format_web_search_response(result)
           
            elif tool_name == "weather":
                city = parameters.get("city")
                if not city:
                    # Fallback to extract city from original query
                    city = self.extract_city_from_query(query)
               
                if not city:
                    return "I couldn't identify which city you want weather information for. Please specify a city name."
               
                result = await self.call_weather(city)
                return self.format_weather_response(result)
           
            elif tool_name == "top_headlines":
                page_size = parameters.get("page_size", 5)
                country = parameters.get("country", "us")
                result = await self.call_top_headlines(page_size, country)
                return self.format_news_response(result, is_search=False)
           
            elif tool_name == "search_news":
                news_query = parameters.get("query", query)
                page_size = parameters.get("page_size", 5)
                result = await self.call_search_news(news_query, page_size)
                return self.format_news_response(result, is_search=True, query=news_query)
           
            elif tool_name == "leave_calculator":
                employee_id = parameters.get("employee_id")
                overrides = {
                    key: parameters.get(key)
                    for key in ("start_date", "end_date", "leave_type")
                    if parameters.get(key)
                }
                result = await self.call_leave_analyze(query, employee_id, overrides=overrides or None)
                return json.dumps(result)
           
            else:  # tool_name == "none" or unknown
                return "Unable to process your request, please try again."
               
        except Exception as e:
            logger.error(f"Error processing query with selected tool {tool_name}: {e}")
            return "Unable to process your request, please try again."
   
    async def call_web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Call web search tool"""
        try:
            return await self.web_search_tool.execute(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=True
            )
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {"error": str(e)}
   
    async def call_weather(self, city: str) -> Dict[str, Any]:
        """Call weather tool"""
        try:
            return await self.weather_tool.execute(city=city)
        except Exception as e:
            logger.error(f"Weather error: {e}")
            return {"error": str(e)}
   
    async def call_top_headlines(self, page_size: int = 5, country: str = "us") -> Dict[str, Any]:
        """Call top headlines tool"""
        try:
            return await self.top_headlines_tool.execute(page_size=page_size, country=country)
        except Exception as e:
            logger.error(f"Top headlines error: {e}")
            return {"error": str(e)}
   
    async def call_search_news(self, query: str, page_size: int = 5) -> Dict[str, Any]:
        """Call search news tool"""
        try:
            return await self.search_news_tool.execute(query=query, page_size=page_size)
        except Exception as e:
            logger.error(f"Search news error: {e}")
            return {"error": str(e)}

    async def call_leave_analyze(
        self,
        query: str,
        employee_id: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze a leave request query and return a leave request card"""
        overrides = overrides or {}
        try:
            result = await self.leave_calculator_tool.execute(
                mode="analyze",
                query=query,
                employee_id=employee_id,
                start_date=overrides.get("start_date"),
                end_date=overrides.get("end_date"),
                leave_type=overrides.get("leave_type"),
            )
            self.logger.info(f"Leave analyze result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Leave analyze failed: {e}")
            return {"error": str(e)}

    async def call_leave_validate(
        self,
        employee_id: str,
        start_date: str,
        end_date: str,
        leave_type: str,
    ) -> Dict[str, Any]:
        """Validate a leave request via the leave calculator API"""
        try:
            result = await self.leave_calculator_tool.execute(
                mode="validate",
                employee_id=employee_id,
                start_date=start_date,
                end_date=end_date,
                leave_type=leave_type,
            )
            self.logger.info(f"Leave validate result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Leave validate failed: {e}")
            return {"error": str(e)}

    @staticmethod
    def _format_markdown_link(label: Optional[str], url: Optional[str]) -> str:
        """Render a Markdown hyperlink when possible, escaping the title safely."""
        clean_label = (label or "View Source").strip() or "View Source"
        if not url:
            return clean_label

        safe_label = (
            clean_label
            .replace("[", r"\[")
            .replace("]", r"\]")
        )
        safe_url = url.strip()
        return f"[{safe_label}]({safe_url})"

 
    # Legacy methods kept for backward compatibility (now unused but preserved)
    def should_use_web_search(self, query: str) -> bool:
        """Conservative heuristic for web search trigger.
        Only trigger when the user explicitly asks to search the web.
        Avoid generic phrases like "what is" which caused over-triggering.
        """
        web_indicators = [
            "search", "look up", "lookup", "google",
            "web search", "internet search", "on the web", "online search",
            "wikipedia", "wiki"
        ]
        text = query.lower().strip()
        return any(tok in text for tok in web_indicators)
   
    def should_use_weather(self, query: str) -> bool:
        """Determine if query should use weather tool (DEPRECATED - kept for compatibility)"""
        weather_indicators = [
            "weather", "temperature", "forecast", "rain", "sunny", "cloudy",
            "hot", "cold", "climate", "humidity", "wind", "storm"
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in weather_indicators)
   
    def should_use_news(self, query: str) -> bool:
        """Trigger news only when user clearly refers to news/headlines."""
        news_indicators = [
            "news", "headlines", "breaking news", "top headlines", "latest news",
            "current events", "news about", "headlines about"
        ]
        text = query.lower().strip()
        return any(tok in text for tok in news_indicators)

    def is_general_news_request(self, query: str) -> bool:
        """Return True when the user asks for general headlines rather than a specific news search."""
        q = (query or "").lower()
        general_phrases = [
            "headlines",
            "top headlines",
            "latest news",
            "what's in the news",
            "show me the news",
            "today's headlines",
        ]
        return any(p in q for p in general_phrases)

    def should_use_leave_calculator(self, query: str) -> bool:
        """Detect if the query is related to leave requests/validation.
        Excludes technical terms to prevent routing conflicts."""
       
        # Technical/system terms that should NOT trigger leave calculator
        technical_exclusions = [
            "submit_leave_form",
            "leave_form",
            "form_submission",
            "action_process",
            "leave_submission",
            "validate_leave",
            "calculate leave impact",
            "leave impact",
            "impact calculation",
            "/submit",
            "intent:",
            "action:",
            "slot:",
            "entity:",
            "response:",
            "utter_",
            "rasa_",
            "bot_"
        ]
       
        q = (query or "").lower().strip()
       
        # First check for technical exclusions
        for exclusion in technical_exclusions:
            if exclusion in q:
                return False
       
        # Then check for leave indicators
        leave_indicators = [
            "leave",
            "vacation",
            "sick",
            "sick leave",
            "annual leave",
            "time off",
            "pto",
            "apply leave",
            "request leave",
            "leave request",
            "apply for leave",
            "flexi leave",
            "casual leave"
        ]
       
        return any(tok in q for tok in leave_indicators)
   
    def extract_city_from_query(self, query: str) -> Optional[str]:
        """Extract city name from weather query"""
        query_lower = query.lower()
       
        # Enhanced patterns for weather queries with more variations
        patterns = [
            "weather in ",
            "weather for ",
            "weather like in ",
            "weather like at ",
            "temperature in ",
            "temperature for ",
            "temperature at ",
            "forecast for ",
            "forecast in ",
            "climate in ",
            "climate at ",
            "weather at ",
        ]
       
        # Try pattern-based extraction first
        for pattern in patterns:
            if pattern in query_lower:
                # Extract the part after the pattern
                parts = query_lower.split(pattern, 1)
                if len(parts) > 1:
                    # Get the city name, handle multi-word cities
                    city_part = parts[1].strip()
                    # Remove common trailing words
                    city_part = re.sub(r'\s+(today|tomorrow|now|currently|please).*$', '', city_part)
                    # Take up to 2 words for city names like "New York"
                    city_words = city_part.split()[:2]
                    if city_words:
                        return ' '.join(word.capitalize() for word in city_words)
       
        # Common Indian cities (handle both English and common misspellings)
        indian_cities = {
            'delhi': 'Delhi', 'dehli': 'Delhi', 'new delhi': 'New Delhi',
            'mumbai': 'Mumbai', 'bombay': 'Mumbai',
            'bangalore': 'Bangalore', 'bengaluru': 'Bangalore', 'banglore': 'Bangalore',
            'chennai': 'Chennai', 'madras': 'Chennai',
            'kolkata': 'Kolkata', 'calcutta': 'Kolkata',
            'hyderabad': 'Hyderabad', 'hyd': 'Hyderabad',
            'pune': 'Pune', 'poona': 'Pune',
            'ahmedabad': 'Ahmedabad', 'amdavad': 'Ahmedabad',
            'jaipur': 'Jaipur', 'surat': 'Surat', 'lucknow': 'Lucknow',
            'kanpur': 'Kanpur', 'nagpur': 'Nagpur', 'indore': 'Indore',
            'thane': 'Thane', 'bhopal': 'Bhopal', 'visakhapatnam': 'Visakhapatnam',
            'pimpri': 'Pimpri-Chinchwad', 'patna': 'Patna', 'vadodara': 'Vadodara',
            'ghaziabad': 'Ghaziabad', 'ludhiana': 'Ludhiana', 'agra': 'Agra',
            'nashik': 'Nashik', 'faridabad': 'Faridabad', 'meerut': 'Meerut',
            'rajkot': 'Rajkot', 'kalyan': 'Kalyan', 'vasai': 'Vasai-Virar',
            'varanasi': 'Varanasi', 'srinagar': 'Srinagar', 'aurangabad': 'Aurangabad',
            'dhanbad': 'Dhanbad', 'amritsar': 'Amritsar', 'navi mumbai': 'Navi Mumbai',
            'allahabad': 'Allahabad', 'ranchi': 'Ranchi', 'howrah': 'Howrah',
            'coimbatore': 'Coimbatore', 'jabalpur': 'Jabalpur', 'gwalior': 'Gwalior',
            'vijayawada': 'Vijayawada', 'jodhpur': 'Jodhpur', 'madurai': 'Madurai',
        }
       
        # Check for known cities (case-insensitive)
        for city_key, city_proper in indian_cities.items():
            if city_key in query_lower:
                return city_proper
       
        # International cities
        international_cities = {
            'london': 'London', 'paris': 'Paris', 'tokyo': 'Tokyo', 'new york': 'New York',
            'sydney': 'Sydney', 'dubai': 'Dubai', 'singapore': 'Singapore',
            'los angeles': 'Los Angeles', 'chicago': 'Chicago', 'toronto': 'Toronto',
            'berlin': 'Berlin', 'madrid': 'Madrid', 'rome': 'Rome', 'amsterdam': 'Amsterdam'
        }
       
        for city_key, city_proper in international_cities.items():
            if city_key in query_lower:
                return city_proper
       
        # Only consider a capitalized-word fallback if the query clearly looks like a weather question
        weather_indicators = {
            'weather', 'temperature', 'forecast', 'climate', 'humidity', 'wind', 'rain', 'sunny', 'cloudy', 'hot', 'cold'
        }
        if any(tok in query_lower for tok in weather_indicators):
            # Fallback: Look for capitalized words that might be cities
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', query)
            if words:
                # Filter out common non-city words and task words
                non_cities = {
                    'Weather','Temperature','Forecast','Climate','Today','Tomorrow','Please','Tell','What','How',
                    'Summarize','Summary','Explain','Refactor','Return','Write','JSON','Poem','Black','Hole','Eli5'
                }
                for word in words:
                    if word not in non_cities and len(word) > 2:
                        return word

        return None
 
    def format_web_search_response(self, result: Dict[str, Any]) -> str:
        """Format web search response for the chatbot"""
        if "error" in result:
            return f"I couldn't search the web right now: {result['error']}"
       
        try:
            response = ""
            if "answer" in result and result["answer"]:
                response += f"🔍 Search Result:\n{result['answer']}\n\n"
           
            if "results" in result and result["results"]:
                response += "📝 Sources:\n"
                for i, item in enumerate(result["results"][:3], 1):
                    title = item.get("title", "No title")
                    url = item.get("url")
                    content = item.get("content", "")[:100] + "..." if item.get("content") else ""
                    link_text = self._format_markdown_link(title, url)
                    response += f"{i}. {link_text}\n"
                    if content:
                        response += f"   {content}\n"
                    response += "\n"
           
            return response if response else "I couldn't find specific information on that topic."
           
        except Exception as e:
            logger.error(f"Error formatting web search response: {e}")
            return "I found some information but couldn't format it properly."
 
    def format_weather_response(self, result: Dict[str, Any]) -> str:
        """Format weather response for the chatbot"""
        if "error" in result:
            return f"I couldn't get weather information: {result['error']}"
       
        try:
            if "main" in result and "weather" in result:
                city = result.get("name", "the location")
                temp = result["main"].get("temp", "N/A")
                feels_like = result["main"].get("feels_like", "N/A")
                description = result["weather"][0].get("description", "").title()
                humidity = result["main"].get("humidity", "N/A")
               
                response = f"🌤️ Weather in {city}:\n"
                response += f"Temperature: {temp}°C (feels like {feels_like}°C)\n"
                response += f"Condition: {description}\n"
                response += f"Humidity: {humidity}%"
               
                return response
            else:
                return "I received weather data but couldn't parse it properly."
               
        except Exception as e:
            logger.error(f"Error formatting weather response: {e}")
            return "I found weather information but couldn't format it properly."
 
    def format_news_response(self, result: Dict[str, Any], is_search: bool = False, query: Optional[str] = None) -> str:
        """Format news response for the chatbot"""
        if "error" in result:
            return f"I couldn't get news information: {result['error']}"
       
        try:
            if "articles" in result and result["articles"]:
                articles = result["articles"]
                # Optional post-filtering by query terms for relevance on search
                if is_search:
                    q = (query or result.get("_search_query") or "").lower()
                    terms = [t for t in q.replace('"', ' ').split() if len(t) > 2]
                    if terms:
                        def relevant(a):
                            title = (a.get("title") or "").lower()
                            desc = (a.get("description") or "").lower()
                            return any(t in title or t in desc for t in terms)
                        filtered = [a for a in articles if relevant(a)]
                        if filtered:
                            articles = filtered
                articles = articles[:5]  # Limit to 5
               
                title = "🗞️ Latest News:" if not is_search else "🔍 News Search Results:"
                response = f"{title}\n\n"
               
                for i, article in enumerate(articles, 1):
                    title = article.get("title", "No title")
                    description = article.get("description", "")
                    url = article.get("url")
                    published_at = article.get("publishedAt", "")
                   
                    link_text = self._format_markdown_link(title, url)
                    response += f"{i}. {link_text}\n"
                    if description:
                        desc_short = description[:100] + "..." if len(description) > 100 else description
                        response += f"   {desc_short}\n"
                    if published_at:
                        # Format date
                        try:
                            from datetime import datetime
                            date_obj = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                            formatted_date = date_obj.strftime('%Y-%m-%d %H:%M')
                            response += f"   Published: {formatted_date}\n"
                        except:
                            response += f"   Published: {published_at}\n"
                    response += "\n"
               
                return response
            else:
                return "I couldn't find any recent news articles."
               
        except Exception as e:
            logger.error(f"Error formatting news response: {e}")
            return "I found news information but couldn't format it properly."
 
# Global instance for use in actions
mcp_client = MCPClient()
 
# Updated thread-safe synchronous wrapper functions for use in RASA actions
def run_llm_query_processing(query: str) -> str:
    """Thread-safe synchronous wrapper for LLM-based query processing"""
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(mcp_client.process_query_with_llm(query))
            return result
        finally:
            # Properly shutdown the loop and cleanup all tasks
            try:
                # Cancel all remaining tasks
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    # Wait for all tasks to complete or be cancelled
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
               
                # Close all open connections
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
               
            except Exception as cleanup_error:
                logger.warning(f"Event loop cleanup warning: {cleanup_error}")
            finally:
                # Force close the loop
                if not loop.is_closed():
                    loop.close()
   
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async)
            return future.result(timeout=20)  # reduced timeout for LLM processing
    except concurrent.futures.TimeoutError:
        logger.error("LLM query processing timeout")
        return "Unable to process your request, please try again."
    except Exception as e:
        logger.error(f"LLM query processing wrapper error: {e}")
        return "Unable to process your request, please try again."
 
# Legacy wrapper functions kept for backward compatibility
def run_web_search(query: str, max_results: int = 5) -> str:
    """Thread-safe synchronous wrapper for web search (DEPRECATED - kept for compatibility)"""
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(mcp_client.call_web_search(query, max_results))
            return mcp_client.format_web_search_response(result)
        finally:
            # Properly shutdown the loop and cleanup all tasks
            try:
                # Cancel all remaining tasks
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    # Wait for all tasks to complete or be cancelled
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
               
                # Close all open connections
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
               
            except Exception as cleanup_error:
                logger.warning(f"Event loop cleanup warning: {cleanup_error}")
            finally:
                # Force close the loop
                if not loop.is_closed():
                    loop.close()
   
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async)
            return future.result(timeout=8)
    except concurrent.futures.TimeoutError:
        logger.error("Web search timeout")
        return "I'm sorry, the web search request timed out. Please try again."
    except Exception as e:
        logger.error(f"Web search wrapper error: {e}")
        return f"I encountered an error while searching: {str(e)}"
 
def run_weather_search(city: str) -> str:
    """Thread-safe synchronous wrapper for weather search (DEPRECATED - kept for compatibility)"""
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = None
        try:
            result = loop.run_until_complete(mcp_client.call_weather(city))
            return mcp_client.format_weather_response(result)
        finally:
            # Properly shutdown the loop and cleanup all tasks
            try:
                # Cancel all remaining tasks
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    # Wait for all tasks to complete or be cancelled
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
               
                # Close all open connections
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
               
            except Exception as cleanup_error:
                logger.warning(f"Event loop cleanup warning: {cleanup_error}")
            finally:
                # Force close the loop
                if not loop.is_closed():
                    loop.close()
   
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async)
            return future.result(timeout=15)  # reduced timeout
    except concurrent.futures.TimeoutError:
        logger.error("Weather search timeout")
        return "I'm sorry, the weather request timed out. Please try again."
    except Exception as e:
        logger.error(f"Weather search wrapper error: {e}")
        return f"I encountered an error while getting weather information: {str(e)}"
 
def run_news_search(query: str = None, is_headlines: bool = True, page_size: int = 5) -> str:
    """Thread-safe synchronous wrapper for news search (DEPRECATED - kept for compatibility)"""
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if is_headlines or not query:
                result = loop.run_until_complete(mcp_client.call_top_headlines(page_size=page_size))
                return mcp_client.format_news_response(result, is_search=False)
            else:
                result = loop.run_until_complete(mcp_client.call_search_news(query, page_size))
                return mcp_client.format_news_response(result, is_search=True, query=query)
        finally:
            # Properly shutdown the loop and cleanup all tasks
            try:
                # Cancel all remaining tasks
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    # Wait for all tasks to complete or be cancelled
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
               
                # Close all open connections
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
               
            except Exception as cleanup_error:
                logger.warning(f"Event loop cleanup warning: {cleanup_error}")
            finally:
                # Force close the loop
                if not loop.is_closed():
                    loop.close()
   
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async)
            return future.result(timeout=15)  # reduced timeout
    except concurrent.futures.TimeoutError:
        logger.error("News search timeout")
        return "I'm sorry, the news request timed out. Please try again."
    except Exception as e:
        logger.error(f"News search wrapper error: {e}")
        return f"I encountered an error while getting news: {str(e)}"


def run_leave_analysis(
    query: str,
    employee_id: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Thread-safe wrapper to build a leave request card with fresh client instance."""

    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Create fresh MCPClient instance for complete isolation
            fresh_client = MCPClient()
            result = loop.run_until_complete(
                fresh_client.call_leave_analyze(query, employee_id, overrides)
            )
            return result
        finally:
            # Properly shutdown the loop and cleanup all tasks
            try:
                # Cancel all remaining tasks
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    # Wait for all tasks to complete or be cancelled
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
               
                # Close all open connections
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
               
            except Exception as cleanup_error:
                logger.warning(f"Event loop cleanup warning: {cleanup_error}")
            finally:
                # Force close the loop
                if not loop.is_closed():
                    loop.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async)
            return future.result(timeout=30)
    except concurrent.futures.TimeoutError:
        logger.error("Leave analysis timed out")
        return {"error": "Unable to prepare leave form at the moment."}
    except Exception as e:
        logger.error(f"Leave analysis wrapper error: {e}")
        return {"error": "Unable to prepare leave form at the moment."}


def run_leave_validation(
    employee_id: str,
    start_date: str,
    end_date: str,
    leave_type: str,
) -> Dict[str, Any]:
    """Thread-safe wrapper to validate a leave request via API with fresh client instance."""

    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Create fresh MCPClient instance for complete isolation
            fresh_client = MCPClient()
            result = loop.run_until_complete(
                fresh_client.call_leave_validate(
                    employee_id, start_date, end_date, leave_type
                )
            )
            return result
        finally:
            # Properly shutdown the loop and cleanup all tasks
            try:
                # Cancel all remaining tasks
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    # Wait for all tasks to complete or be cancelled
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
               
                # Close all open connections
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
               
            except Exception as cleanup_error:
                logger.warning(f"Event loop cleanup warning: {cleanup_error}")
            finally:
                # Force close the loop
                if not loop.is_closed():
                    loop.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async)
            return future.result(timeout=30)
    except concurrent.futures.TimeoutError:
        logger.error("Leave validation timed out")
        return {"error": "Unable to validate leave at the moment."}
    except Exception as e:
        logger.error(f"Leave validation wrapper error: {e}")
        return {"error": "Unable to validate leave at the moment."}
 
