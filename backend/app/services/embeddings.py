"""
Embeddings service using Ollama's nomic-embed-text model.
Converts text to 768-dimensional semantic vectors.
"""

import json
from typing import List, Union
import numpy as np
import requests

from backend.app.config import Config


class EmbeddingsService:
    """Wrapper around Ollama embedding API."""

    def __init__(self, base_url: str = None, model: str = None):
        """
        Initialize embeddings service.

        Args:
            base_url: Ollama API base URL (default: from config)
            model: Embedding model name (default: from config)
        """
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.model = model or Config.EMBEDDING_MODEL
        self.embed_url = f"{self.base_url}/api/embed"
        self._cache = {}  # Simple cache: text -> embedding

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            text: Single text string or list of strings

        Returns:
            numpy array:
            - Single text: shape (768,) — one embedding
            - List of texts: shape (N, 768) — N embeddings

        Raises:
            requests.RequestException: If Ollama API call fails
        """
        # Handle single vs batch
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        # Get embeddings
        embeddings = []
        for t in texts:
            # Check cache first
            if t in self._cache:
                embeddings.append(self._cache[t])
                continue

            # Call Ollama API
            try:
                response = requests.post(
                    self.embed_url,
                    json={"model": self.model, "input": t},
                    timeout=60,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                raise requests.RequestException(
                    f"Failed to embed text with {self.model}: {e}"
                ) from e

            data = response.json()
            if "embeddings" not in data or not data["embeddings"]:
                # Ollama returns empty embeddings for empty/whitespace-only text
                # Return zero vector as fallback
                embedding = np.zeros(Config.EMBEDDING_DIMENSION, dtype=np.float32)
            else:
                embedding = np.array(data["embeddings"][0], dtype=np.float32)
            self._cache[t] = embedding
            embeddings.append(embedding)

        # Return as single array or stack of arrays
        result = np.array(embeddings, dtype=np.float32)
        return result[0] if is_single else result

    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings (768 for nomic-embed-text)."""
        return Config.EMBEDDING_DIMENSION

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
