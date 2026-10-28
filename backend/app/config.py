"""
Configuration management for RAG Q&A system.
Reads environment variables with sensible defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import time (application startup). Default override=False
# ensures real process environment variables are never overwritten.
load_dotenv()


class Config:
    """Environment-based configuration."""

    # Ollama settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

    # ChromaDB settings
    CHROMA_DB_PATH = os.getenv(
        "CHROMA_DB_PATH",
        str(Path(__file__).parent.parent / "chroma_db")
    )

    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    # LLM generation settings (Phase 6-A1)
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b-instruct")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))

    @classmethod
    def ensure_chroma_db_dir(cls):
        """Create ChromaDB directory if it doesn't exist."""
        Path(cls.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)

    @classmethod
    def to_dict(cls):
        """Return config as dictionary (useful for logging/debugging)."""
        return {
            "ollama_base_url": cls.OLLAMA_BASE_URL,
            "embedding_model": cls.EMBEDDING_MODEL,
            "embedding_dimension": cls.EMBEDDING_DIMENSION,
            "chroma_db_path": cls.CHROMA_DB_PATH,
            "environment": cls.ENVIRONMENT,
            "api_host": cls.API_HOST,
            "api_port": cls.API_PORT,
        }
