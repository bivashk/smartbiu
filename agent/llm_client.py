"""
Unified LLM Client for SmartBIU Text-to-SQL Agent
Supports Groq (Llama-3.3-70B), Google Gemini REST, OpenAI (GPT-4o-mini), and local Ollama.
Operates with standard 'requests' library (zero extra heavy dependencies).
"""

import os
import re
import json
from typing import Optional, Dict, Any, List

def _load_dotenv_if_exists():
    """Reads .env from project root if present and sets os.environ without extra dependencies."""
    root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(root_env):
        try:
            with open(root_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and not os.getenv(k):
                            os.environ[k] = v
        except Exception:
            pass

class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 12
    ):
        _load_dotenv_if_exists()
        self.timeout = timeout
        
        # Determine provider & credentials
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        
        if provider:
            self.provider = provider.lower()
        elif os.getenv("GROQ_API_KEY"):
            self.provider = "groq"
        elif os.getenv("GEMINI_API_KEY"):
            self.provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            self.provider = "openai"
        elif os.getenv("OLLAMA_HOST"):
            self.provider = "ollama"
        else:
            self.provider = "offline"
            
        # Defaults per provider
        if self.provider == "groq":
            self.base_url = base_url or "https://api.groq.com/openai/v1/chat/completions"
            self.model = model or "llama-3.3-70b-versatile"
        elif self.provider == "gemini":
            self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/models"
            self.model = model or "gemini-1.5-flash"
        elif self.provider == "openai":
            self.base_url = base_url or "https://api.openai.com/v1/chat/completions"
            self.model = model or "gpt-4o-mini"
        elif self.provider == "ollama":
            self.base_url = base_url or "http://localhost:11434/v1/chat/completions"
            self.model = model or "llama3.2"
        else:
            self.base_url = ""
            self.model = "semantic-template-fallback"

    def is_available(self) -> bool:
        """Returns True if a live LLM endpoint is configured and active."""
        if self.provider == "offline" or not self.provider:
            return False
        if self.provider in ["groq", "gemini", "openai"] and not self.api_key:
            return False
        return True

    def get_info(self) -> Dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "active": "LIVE_LLM" if self.is_available() else "SEMANTIC_FALLBACK"
        }

    def generate_sql(self, user_query: str, system_prompt: str) -> Optional[str]:
        """Calls the configured LLM to generate SQL given schema context."""
        if not self.is_available():
            return None
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Translate this natural language business question into valid SQLite SQL:\n\"{user_query}\"\nReturn ONLY the raw SQL query inside a ```sql ... ``` block or as plain text."}
        ]
        
        try:
            raw_response = self._call_chat_completion(messages)
            if raw_response:
                return self._extract_sql(raw_response)
        except Exception as e:
            print(f"[SmartBIU LLM Warning] LLM generation failed ({e}). Falling back to semantic engine.")
            return None
            
        return None

    def heal_sql(self, failed_sql: str, error_message: str, schema_context: str, user_query: str) -> Optional[str]:
        """Self-healing LLM reflection: asks the LLM to inspect its runtime error and fix the query."""
        if not self.is_available():
            return None
            
        system_prompt = f"""You are an expert SQLite optimization engineer for a bank BIU.
A previously generated SQL query failed during execution.
Examine the database schema, diagnose the root cause of the error, and return ONLY the corrected, executable SQLite query.

{schema_context}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""User Goal: "{user_query}"
Failed SQL:
```sql
{failed_sql}
```
Execution Error:
{error_message}

Fix the SQL so it executes successfully against SQLite. Return ONLY the corrected SQL query."""}
        ]
        
        try:
            raw_response = self._call_chat_completion(messages)
            if raw_response:
                return self._extract_sql(raw_response)
        except Exception as e:
            print(f"[SmartBIU LLM Warning] LLM self-healing call failed ({e}).")
            return None
            
        return None

    def _call_chat_completion(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Dispatches chat request to OpenAI-compatible or Gemini endpoints."""
        if self.provider == "gemini":
            return self._call_gemini_rest(messages)
            
        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 600
        }
        
        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:150]}")

    def _call_gemini_rest(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Direct REST call to Google Gemini generateContent API."""
        import requests
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        
        # Combine system prompt and user contents
        prompt_parts = []
        for m in messages:
            prefix = "SYSTEM INSTRUCTION:\n" if m["role"] == "system" else "USER:\n"
            prompt_parts.append(f"{prefix}{m['content']}")
            
        full_text = "\n\n".join(prompt_parts)
        
        payload = {
            "contents": [{"parts": [{"text": full_text}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 600
            }
        }
        
        resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=self.timeout)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                return candidates[0]["content"]["parts"][0]["text"]
        else:
            raise RuntimeError(f"Gemini API HTTP {resp.status_code}: {resp.text[:150]}")
            
        return None

    def _extract_sql(self, text: str) -> str:
        """Strips markdown code blocks, explanatory prose, and trailing punctuation."""
        # Find ```sql ... ``` or ``` ... ```
        match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
        else:
            # Look for lines starting with SELECT or WITH
            lines = [l for l in text.strip().split("\n") if not l.strip().startswith("--")]
            extracted = "\n".join(lines).strip()
            
        return extracted
