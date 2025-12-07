import httpx
import json
import asyncio
from typing import Dict, Any, Optional, List
import logging
from ..config import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """Service for integrating with Ollama LLM."""

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout
        self.max_retries = settings.ollama_max_retries
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.use_chat_api = True  # Try chat API first, fallback to generate

    async def health_check(self) -> bool:
        """Check if Ollama service is available and model is loaded."""
        try:
            url = f"{self.base_url}/api/tags"
            response = await self.client.get(url, timeout=5)

            if response.status_code != 200:
                return False

            tags_data = response.json()
            models = [model.get("name", "") for model in tags_data.get("models", [])]
            model_available = any(self.model in model for model in models)

            if not model_available:
                logger.warning(f"Model '{self.model}' not found. Available: {models}")
                return False

            logger.info(f"✅ Ollama service available with model '{self.model}'")
            return True

        except Exception as e:
            logger.warning(f"Ollama health check failed: {str(e)}")
            return False

    async def generate_response(
        self,
        prompt: str,
        system_prompt: str = None,
        context: Dict[str, Any] = None
    ) -> str:
        """Generate response from Ollama with retry mechanism."""
        for attempt in range(self.max_retries):
            try:
                if self.use_chat_api:
                    messages = self._build_chat_messages(prompt, system_prompt, context)
                    response = await self._call_ollama_chat(messages)
                else:
                    full_prompt = self._build_prompt(prompt, system_prompt, context)
                    response = await self._call_ollama_generate(full_prompt)

                return response

            except Exception as e:
                logger.warning(f"Ollama attempt {attempt + 1} failed: {str(e)}")

                if attempt == self.max_retries - 1:
                    logger.error("All Ollama attempts failed. Returning fallback response.")
                    return self._get_fallback_response(prompt, context)

                # Exponential backoff: 1s, 2s, 4s
                await asyncio.sleep(2 ** attempt)

    def _build_chat_messages(
        self,
        user_prompt: str,
        system_prompt: str = None,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, str]]:
        """Build messages array for Ollama chat API."""
        messages = []

        # Add system message
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add context as system message
        if context:
            context_str = json.dumps(context, indent=2)
            messages.append({
                "role": "system",
                "content": f"Context:\n{context_str}"
            })

        # Add user message
        messages.append({"role": "user", "content": user_prompt})

        return messages

    def _build_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        context: Dict[str, Any] = None
    ) -> str:
        """Build single prompt string for generate API."""
        parts = []

        if system_prompt:
            parts.append(f"System: {system_prompt}")

        if context:
            parts.append(f"Context: {json.dumps(context)}")

        parts.append(f"User: {prompt}")

        return "\n\n".join(parts)

    async def _call_ollama_chat(self, messages: List[Dict[str, str]]) -> str:
        """Call Ollama chat API."""
        url = f"{self.base_url}/api/chat"
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }

        try:
            response = await self.client.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "").strip()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404 and self.use_chat_api:
                # Fallback to generate API
                logger.info("Chat API not available, falling back to generate API")
                self.use_chat_api = False
                prompt = self._build_prompt(messages[-1]["content"])
                return await self._call_ollama_generate(prompt)
            raise

    async def _call_ollama_generate(self, prompt: str) -> str:
        """Call Ollama generate API."""
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        response = await self.client.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()

    def _get_fallback_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Fallback response when Ollama is unavailable."""
        return json.dumps({
            "violations": [],
            "overall_assessment": "AI analysis service temporarily unavailable. Please try again later.",
            "key_issues": ["AI service unavailable"]
        })

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()


# Singleton instance
ollama_service = OllamaService()
