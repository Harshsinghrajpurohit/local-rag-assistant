"""
LLM service for interacting with Ollama (or other compatible models).
Provides methods to generate answers using retrieved context.
"""

from typing import List, Dict, Any, Optional
import json
import requests
from ..config import Config

class LLMService:
    """Service for interacting with Ollama LLM endpoints."""

    def __init__(self):
        self.ollama_url = Config.OLLAMA_URL.rstrip('/')
        # You can extend this to support other providers by adding more endpoints

    def generate_answer(
        self,
        query: str,
        context: List[Dict[str, Any]],
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Generate an answer using the provided context.

        Args:
            query: The original user query (for context)
            context: List of source passages (each should contain 'text' field)
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with 'answer' and 'tokens_used' fields.
        """
        # Build the prompt by concatenating context passages
        context_text = "\n\n".join([passage.get("text", "") for passage in context])
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
Answer directly and concisely, citing sources when appropriate.

Question: {query}

Context:
{context_text}

Answer:"""

        # Prepare request payload
        payload = {
            "model": "llama3",  # default model; can be configured
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            # Extract answer and token usage from Ollama response
            answer = data.get("response", "")
            # Ollama token usage info (may be in different fields depending on version)
            tokens_used = data.get("usage", {}).get("total_tokens", 0)

            return {
                "answer": answer.strip(),
                "tokens_used": tokens_used,
            }

        except requests.exceptions.RequestException as e:
            # Handle network errors, timeouts, etc.
            raise OllamaUnavailableError(str(e))
        except Exception as e:
            # Unexpected errors
            raise OllamaUnavailableError(f"LLM generation failed: {str(e)}")