"""Shared helpers for the evaluation scripts (run everything from the repo root)."""
import json
import os
import re
import sys
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

# Windows consoles default to cp1252 and crash on characters such as ligatures.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from backend.app.config import Config  # noqa: E402
from backend.app.services.document_processor import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    process_document,
)

EVAL_DIR = Path(__file__).resolve().parent
DOCS_DIR = EVAL_DIR / "docs"
GOLDEN_PATH = EVAL_DIR / "golden.json"


def gen_model() -> str:
    """Model that writes the questions (override with the GEN_MODEL env var)."""
    return os.getenv("GEN_MODEL") or Config.LLM_MODEL


def judge_model() -> str:
    """Model that grades answers (override with the JUDGE_MODEL env var)."""
    return os.getenv("JUDGE_MODEL") or Config.LLM_MODEL


_QUOTES = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
)


def normalize(text) -> str:
    """Lowercase, fold ligatures/quotes, collapse whitespace, trim edge punctuation."""
    text = unicodedata.normalize("NFKC", str(text or "")).translate(_QUOTES).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:!?\"'()[]")


def list_docs():
    if not DOCS_DIR.exists():
        return []
    return sorted(
        p for p in DOCS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def check_ollama(models):
    """Return a list of problems (empty list means everything is fine)."""
    try:
        r = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as e:
        return [f"Cannot reach Ollama at {Config.OLLAMA_BASE_URL}: {e}"]
    installed = {m.get("name", "") for m in r.json().get("models", [])}
    problems = []
    for m in dict.fromkeys(models):
        if m not in installed and not (":" not in m and f"{m}:latest" in installed):
            problems.append(f"Model '{m}' is not installed (run: ollama pull {m})")
    return problems


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    try:
        return json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
        raise ValueError(f"No JSON object in model output: {raw[:120]!r}")


def ollama_json(prompt: str, model: str, timeout: int = 180, retries: int = 2) -> dict:
    """Call Ollama in JSON mode (temperature 0). Raises RuntimeError after all retries fail."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    last = None
    for _ in range(retries + 1):
        try:
            r = requests.post(f"{Config.OLLAMA_BASE_URL}/api/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            return _parse_json(r.json().get("response", ""))
        except Exception as e:  # network, HTTP, or bad JSON: retry
            last = e
            time.sleep(2)
    raise RuntimeError(f"{model}: {last}")


def ingest_documents(chroma, paths):
    """Ingest files the same way the upload endpoint does. Returns {file name: chunk count}."""
    summary = {}
    for p in paths:
        chunks = process_document(str(p), original_filename=p.name)
        doc_id = str(uuid.uuid4())
        stamp = datetime.now(timezone.utc).isoformat()
        for c in chunks:
            c["doc_id"] = doc_id
            c["file_name"] = p.name
            c["file_type"] = p.suffix.lstrip(".").lower()
            c["upload_time"] = stamp
            c["size_bytes"] = p.stat().st_size
        chroma.add_chunks(chunks)
        summary[p.name] = len(chunks)
    return summary
