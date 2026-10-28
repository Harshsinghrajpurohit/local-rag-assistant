# RAG-Q&A Document Assistant

Hybrid Retrieval-Augmented Generation (RAG) Q&A system. Upload documents
(PDF, DOCX, TXT, MD, CSV, XLSX), ask questions in natural language, and get
answers grounded in your documents with page-level source citations.

All processing is local — no external API keys required.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Supported Document Formats](#supported-document-formats)
- [Testing](#testing)
- [Configuration](#configuration)
- [Project Layout](#project-layout)
- [License](#license)

## Features

| Feature | Description |
|---|---|
| Multi-format ingestion | PDF, DOCX, TXT, MD, CSV, XLSX — validated by extension + magic bytes |
| Hybrid retrieval | BM25 keyword search + ChromaDB dense vectors, fused with Reciprocal Rank Fusion (RRF) |
| Grounded answers | Qwen (via Ollama) generates answers only from retrieved passages |
| Source citations | Every answer cites source file, page/section, and excerpt |
| Evidence sufficiency | System abstains (no hallucination) when <25% of query terms appear in retrieved docs |
| Structured filtering | Numeric comparisons on tabular data (e.g., stipend >= 40000 AND duration <= 3) |
| Document lifecycle | Upload list delete with BM25 index rebuild on delete |
| REST API | FastAPI backend with structured JSON error responses (request_id + UTC timestamps) |
| Streamlit UI | Thin frontend over the API — no RAG logic lives here |

## Architecture

`
+--------------------+--------------------+
| STREAMLIT FRONTEND | FASTAPI BACKEND |
## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11-3.13 | Validated on 3.13.0 |
| Ollama | Installed and running locally (default: http://localhost:11434) |
| Ollama models | ollama pull qwen3:4b-instruct (answers) ollama pull nomic-embed-text:latest (embeddings) |

## Quick Start

Run all commands from the repository root (the backend package is resolved
from the current working directory).

### PowerShell

`powershell
# 1. Create and activate a virtual environment
python -m venv env
env\Scripts\Activate.ps1

# 2. Install Python dependencies
python -m pip install -r requirements.txt

# 3. Pull required Ollama models (separately, not via pip)
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text:latest

# 4. (Optional) Copy the env template — every key is optional;
defaults in backend/app/config.py work out of the box
copy .env.example .env

# 5. Start Ollama (in a separate terminal)
ollama serve

# 6. Start the backend
python -m backend.app.main

# 7. (Optional) Start the Streamlit frontend
python -m streamlit run frontend/app.py
`

### Bash / Git Bash

`ash
python -m venv env
source env/Scripts/activate
python -m pip install -r requirements.txt
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text:latest
cp .env.example .env
ollama serve &          # background terminal
python -m backend.app.main  # backend
python -m streamlit run frontend/app.py  # frontend (optional)
`

