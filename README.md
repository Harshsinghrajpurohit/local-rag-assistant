# RAG Document Q&A Assistant

Upload documents, ask questions, and get answers with source citations. Runs fully locally using Ollama, with no API keys needed.

## Features

- Hybrid search (BM25 + vector search) for accurate retrieval
- Supports PDF, DOCX, TXT, MD, CSV and XLSX files
- Answers include sources (page, section or row range)
- FastAPI backend with a Streamlit web UI

## Tech Stack

FastAPI · Streamlit · ChromaDB · Ollama (Qwen + nomic-embed-text) · PyMuPDF

## Setup

**Prerequisites:** Python 3.11+ and [Ollama](https://ollama.com)

```bash
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text:latest

python -m venv env
env\Scripts\activate          # macOS/Linux: source env/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Terminal 1 - backend
python -m backend.app.main

# Terminal 2 - frontend
python -m streamlit run frontend/app.py
```

Open http://localhost:8501 (API docs at http://localhost:8000/docs).

## Tests

```bash
python -m pytest tests/ -q
```

## Project Structure

```
backend/app/    FastAPI app (api, services, config, models)
frontend/       Streamlit UI
tests/          Test suite
```
