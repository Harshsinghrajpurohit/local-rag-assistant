# Architecture Document

## System Overview

A hybrid RAG system with three main components:
1. **Frontend** (Streamlit) - User interface for document upload and Q&A
2. **Backend** (FastAPI) - RESTful API for document operations and question answering
3. **Data Layer** - ChromaDB (vector storage) + BM25 (keyword index) + metadata

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND (app.py)              │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │ Document Upload  │          │ Chat Interface   │        │
│  │ Document List    │          │ Conversation     │        │
│  │ Delete Actions   │          │ Source Display   │        │
│  └──────────────────┘          └──────────────────┘        │
└─────────────────────────────────────────────────────────────┘
              HTTP Requests via FastAPI Client
┌─────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND (main.py)                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ /documents/upload   - POST: Upload PDFs               │ │
│  │ /documents/list     - GET: List uploaded documents    │ │
│  │ /documents/delete   - DELETE: Remove document        │ │
│  │ /chat/query         - POST: Ask a question            │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
              Uses Services Layer
┌─────────────────────────────────────────────────────────────┐
│                   SERVICES LAYER                            │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ pdf_processor    │  │ embeddings   │  │hybrid_search │  │
│  │ ├─ extract       │  │ ├─ generate  │  │├─ dense      │  │
│  │ ├─ split chunks  │  │ └─ cache     │  │├─ sparse     │  │
│  │ └─ preserve meta │  │              │  │└─ fuse (RRF) │  │
│  └──────────────────┘  └──────────────┘  └──────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ answer_generator                                     │  │
│  │ ├─ format prompt with retrieved passages            │  │
│  │ ├─ call Qwen LLM via Ollama                        │  │
│  │ └─ extract source citations                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
              Reads/Writes to Data Layer
┌─────────────────────────────────────────────────────────────┐
│                   DATA LAYER                                │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │ ChromaDB         │  │ BM25 Index + Metadata            │ │
│  │ ├─ embeddings    │  │ ├─ document chunks               │ │
│  │ ├─ chunk texts   │  │ ├─ document metadata             │ │
│  │ └─ metadata      │  │ └─ page number mapping           │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
│            (Persistent local storage: chroma_db/)           │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
RAG-Q&A-Project/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, route definitions
│   │   ├── config.py            # Configuration, environment variables
│   │   ├── api/
│   │   │   ├── documents.py     # /documents/* endpoints
│   │   │   └── chat.py          # /chat/* endpoints
│   │   └── services/
│   │       ├── pdf_processor.py # PDF extraction, chunking
│   │       ├── embeddings.py    # Sentence Transformer wrapper
│   │       ├── hybrid_retriever.py # Dense + BM25 + RRF fusion
│   │       └── answer_generator.py # LLM prompt + generation
│   └── chroma_db/               # Persistent ChromaDB storage
├── frontend/
│   └── app.py                   # Streamlit UI
├── tests/
│   ├── test_pdf_processor.py    # PDF extraction tests
│   ├── test_embeddings.py       # Embedding generation tests
│   ├── test_retriever.py        # Retrieval quality tests
│   └── test_e2e.py              # End-to-end tests
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment variables
├── .gitignore                   # Git exclusions
├── README.md                    # User-facing documentation
├── PRD.md                       # This project's requirements
├── Architecture.md              # This file: system design
├── Rules.md                     # Implementation boundaries
├── Phases.md                    # Development phases
├── Design.md                    # Visual/UX design (minimal for CLI)
├── Memory.md                    # AI session memory (auto-updated)
└── PROJECT_NOTES.md             # Progress tracking
```

## Data Flow: Question Answering

```
User Question
      ↓
[Streamlit] POST /chat/query (question, conversation_id)
      ↓
[FastAPI] Receive query
      ↓
[hybrid_retriever] 
  ├─→ Dense retrieval: ChromaDB semantic search
  ├─→ Sparse retrieval: BM25 keyword search
  └─→ RRF fusion: Combine ranked results
      ↓
[Retrieved Passages + Metadata]
      ↓
[answer_generator]
  ├─→ Format LangChain prompt with passages
  ├─→ Call Qwen via Ollama
  └─→ Extract source citations
      ↓
[Response with Answer + Sources]
      ↓
[Streamlit] Display answer + expandable sources
```

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Frontend | Streamlit | Simple, no separate frontend code, native components |
| Backend | FastAPI + Uvicorn | Fast, async, minimal boilerplate |
| LLM | Qwen (via Ollama) | Local, no API keys, reproducible |
| Embeddings | Sentence Transformers | Fast, good quality, no external calls |
| Vector DB | ChromaDB | Persistent, lightweight, manages metadata |
| Keyword Search | BM25 | Proven algorithm, lightweight |
| Fusion | Reciprocal Rank Fusion | Simple, effective combination of rankings |
| Document Processing | LangChain | Splits, chunks, metadata handling |
| PDF Extraction | PyMuPDF | Preserves page numbers, fast |

## Key Design Decisions

1. **Backend/Frontend Separation**: Streamlit talks to FastAPI via HTTP. This allows independent scaling and keeps RAG logic centralized.

2. **Hybrid Retrieval**: Combines dense (semantic) + sparse (keyword) search via RRF. Neither alone is optimal; together they handle both paraphrased questions and exact-match lookups.

3. **Persistent Indexes**: ChromaDB is persistent; BM25 index is rebuilt from persisted metadata on startup. No reprocessing of unchanged PDFs.

4. **Metadata Preservation**: Every chunk stores original document ID, page number, and position. This enables traceability and source citations.

5. **LangChain (Not LangGraph)**: LangChain handles document splitting, embedding integration, and prompt formatting. LangGraph is unnecessary for this deterministic RAG pipeline; we don't need dynamic agent routing or conditional logic.

6. **No Agents or Tool Calling**: Initial implementation is a simple prompt + context + answer pipeline. Agents add complexity; we verify basic retrieval works first.

## Error Handling Strategy

- **Retrieval Errors**: Log and return empty results → LLM generates "I don't have enough information" response
- **PDF Processing Errors**: Return clear error to user with file name and reason
- **Embedding Errors**: Retry or skip; log for debugging
- **LLM Errors**: Return error message; don't retry silently
- **Storage Errors**: Fail fast; never silently drop data

## Security and Privacy

- All processing is local; no external API calls except Ollama (which runs locally)
- Uploaded documents stay on the user's machine
- No authentication in MVP (single-user, local use)
- Configuration via environment variables (no hardcoded secrets)
