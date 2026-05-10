"""
llm_client.py
Gemini 2.5 Flash LLM client for AI Resume Optimizer.
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"


class LLMClient:
    """LLM client — Gemini 2.5 Flash (Google)."""

    def __init__(self, provider: str = "gemini", api_key: str = None, model: str = None):
        """
        Initialize Gemini 2.5 Flash client.

        Args:
            provider: Ignored — always uses Gemini.
            api_key: Google AI API key. Falls back to GOOGLE_API_KEY env var.
            model: Ignored — always uses gemini-2.5-flash-preview-05-20.
        """
        self.provider = "gemini"
        self.model = GEMINI_MODEL
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

        if not self.api_key:
            logger.warning("No GOOGLE_API_KEY found. Set it or pass api_key=...")

        self._client = self._init_gemini()
        logger.info(f"LLM client ready: {self.model}")

    def _init_gemini(self):
        """Initialize Google Gemini generative model."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            return genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )

    def complete(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> str:
        """
        Send a completion request to Gemini 2.5 Flash.

        Args:
            prompt: User prompt text.
            system: Optional system instruction prepended to the prompt.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Response text string.
        """
        if not self.api_key:
            raise ValueError(
                "No Google API key provided. Enter your key in the API Key field."
            )

        import google.generativeai as genai

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        try:
            response = self._client.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def extract_json(self, prompt: str) -> dict | list | None:
        """
        Complete a prompt and parse the JSON response.

        Returns:
            Parsed JSON object/array, or None on failure.
        """
        system = (
            "You are an expert resume analyst. "
            "Always respond with ONLY valid JSON — no markdown fences, "
            "no preamble, no trailing text."
        )
        response = self.complete(prompt, system=system)

        if not response:
            return None

        # Strip markdown code fences
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()

        # Try JSON object
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start:end + 1])
            except json.JSONDecodeError:
                pass

        # Try JSON array
        start = response.find('[')
        end = response.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start:end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse JSON from Gemini response")
        return None

    def is_configured(self) -> bool:
        """Return True if an API key is set."""
        return bool(self.api_key)


def create_llm_client(provider: str = "gemini", api_key: str = None, model: str = None) -> LLMClient:
    """Factory — always returns a Gemini 2.5 Flash client."""
    return LLMClient(api_key=api_key)
