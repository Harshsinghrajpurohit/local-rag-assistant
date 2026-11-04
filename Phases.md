# Development Phases

This document breaks the RAG project into manageable, sequential phases. Each phase has clear goals, expected completion criteria, and dependencies on previous phases.

**Key Principle**: Work in small approved steps. Explain concepts before implementation. Wait for approval before execution. Run tests after approval and report actual results.

---

## Phase 0: Project Setup and Environment Verification
**Goal**: Verify Python environment, installed dependencies, Ollama setup, and create minimal project structure.

**Tasks**:
- [ ] Verify Python version (3.11+ required)
- [ ] Check if venv exists and activate it
- [ ] Review existing requirements.txt
- [ ] Verify pip dependencies are installable
- [ ] Verify Ollama is installed and running
- [ ] Verify qwen3:4b-instruct model is available (or determine correct model name)
- [ ] Create minimal project directory structure (backend/, frontend/, tests/)
- [ ] Create .env.example template
- [ ] Create initial .gitignore

**Deliverables**:
- Working Python environment
- Verified Ollama + model
- Project folders created
- No code or models modified

**Status**: ⏳ Awaiting approval

---

## Phase 1: PDF Extraction and Chunking
**Goal**: Extract text from PDFs while preserving page numbers and metadata. Split documents into overlapping chunks.

**Concepts to Learn**:
- How PDF extraction preserves document structure
- Why chunking is necessary for retrieval
- How overlapping chunks improve context retention
- Document metadata: original file name, unique ID, page mapping

**Tasks**:
- [ ] Create `backend/app/services/pdf_processor.py`
- [ ] Implement PDF text extraction with PyMuPDF (preserving page numbers)
- [ ] Implement document chunking with LangChain (configurable size and overlap)
- [ ] Implement metadata attachment (document ID, page numbers, chunk index)
- [ ] Write unit tests for PDF extraction and chunking
- [ ] Test with sample PDFs (different sizes, formats, number of pages)
- [ ] Verify page number accuracy in metadata

**Deliverables**:
- `pdf_processor.py` with extraction and chunking functions
- Passing unit tests
- Test report showing chunk distribution, page preservation
- No database writes yet

**Dependencies**: Phase 0 (environment setup)

**Status**: ⏳ Awaiting Phase 0 completion

---

## Phase 2: Embeddings and ChromaDB Integration
**Goal**: Generate dense embeddings using Sentence Transformers and store them in ChromaDB with metadata.

**Concepts to Learn**:
- How embeddings represent semantic meaning (text → vector)
- Why embeddings enable similarity search
- How ChromaDB stores vectors and metadata
- Persistence: how ChromaDB survives application restarts
- Metadata indexing: querying by document ID, page number

**Tasks**:
- [ ] Create `backend/app/services/embeddings.py`
- [ ] Initialize Sentence Transformer model locally (all-MiniLM-L6-v2 or similar)
- [ ] Implement embedding generation for text chunks
- [ ] Create `backend/app/config.py` with ChromaDB path configuration
- [ ] Initialize ChromaDB persistent client pointing to `backend/chroma_db/`
- [ ] Implement function to insert chunks with embeddings and metadata into ChromaDB
- [ ] Implement function to query ChromaDB by semantic similarity
- [ ] Write unit tests for embedding generation and ChromaDB operations
- [ ] Test persistence: write data, restart app, verify data is still there
- [ ] Test metadata filtering: retrieve by document ID, page number

**Deliverables**:
- `embeddings.py` with embedding generation
- ChromaDB integration code
- Persistent data storage in `backend/chroma_db/`
- Passing unit tests including persistence verification
- Test report showing embedding quality and retrieval accuracy

**Dependencies**: Phase 1 (PDF processor)

**Status**: ⏳ Awaiting Phase 1 completion

---

## Phase 3: Hybrid Retrieval (Dense + BM25 + RRF)
**Goal**: Implement keyword-based (BM25) retrieval, dense (ChromaDB) retrieval, and combine them via Reciprocal Rank Fusion.

**Concepts to Learn**:
- How BM25 (Okapi algorithm) ranks documents by keyword relevance
- Sparse vs. dense retrieval: strengths and weaknesses of each
- Why hybrid retrieval outperforms either method alone
- Reciprocal Rank Fusion: combining ranked lists without a common score
- Index synchronization: keeping ChromaDB and BM25 consistent
- Rebuilding BM25 from persisted metadata

**Tasks**:
- [ ] Create `backend/app/services/hybrid_retriever.py`
- [ ] Implement BM25 index initialization from ChromaDB metadata
- [ ] Implement BM25 search (keyword matching and ranking)
- [ ] Implement dense retrieval from ChromaDB
- [ ] Implement RRF fusion algorithm (combine two ranked lists)
- [ ] Implement index rebuild function (called on app startup)
- [ ] Write unit tests for dense retrieval, BM25, and RRF fusion
- [ ] Test query coverage: questions that need semantic search vs. keyword search
- [ ] Test consistency: verify results are deduplicated and properly ranked
- [ ] Document expected retrieval quality

**Deliverables**:
- `hybrid_retriever.py` with dense, sparse, and RRF functions
- BM25 index rebuilt from metadata on startup
- Passing unit tests
- Test report: retrieval accuracy for different query types
- Example queries showing why hybrid retrieval is better than either alone

**Dependencies**: Phase 2 (ChromaDB + embeddings)

**Status**: ⏳ Awaiting Phase 2 completion

---

## Phase 4: Answer Generation with LLM
**Goal**: Use LangChain to format prompts with retrieved passages and generate answers via Qwen (Ollama).

**Concepts to Learn**:
- How context windows work in LLMs
- How prompts guide LLM behavior (role, instructions, examples)
- How retrieved passages ground answers
- Why temperature and other parameters affect determinism
- Token limits: what happens if context is too long
- Handling insufficient evidence: explicit "I don't know" vs. hallucination

**Tasks**:
- [ ] Create `backend/app/services/answer_generator.py`
- [ ] Implement LangChain prompt template for RAG
- [ ] Implement Ollama LLM integration (qwen3:4b-instruct or verified model name)
- [ ] Implement passage formatting into prompt context
- [ ] Implement answer generation with retrieved passages as context
- [ ] Implement source citation extraction (document name, page, supporting passage)
- [ ] Implement graceful handling of missing information ("I don't have enough context")
- [ ] Write unit tests for prompt formatting and answer generation
- [ ] Test with known-good retrieval results
- [ ] Test with empty retrieval results
- [ ] Document prompt structure and expected LLM behavior

**Deliverables**:
- `answer_generator.py` with LLM integration
- LangChain prompt template
- Passing unit tests
- Example Q&A interactions showing grounded answers with source citations
- Log of Ollama calls (for debugging)

**Dependencies**: Phase 3 (hybrid retrieval)

**Status**: ⏳ Awaiting Phase 3 completion

---

## Phase 5: FastAPI Backend Endpoints
**Goal**: Expose RAG functionality through REST API endpoints for document operations and question answering.

**Concepts to Learn**:
- FastAPI request/response models (Pydantic)
- Async handlers and dependency injection
- Error handling and HTTP status codes
- File upload handling
- Session/conversation state management

**Tasks**:
- [ ] Create `backend/app/main.py` with FastAPI app
- [ ] Create `backend/app/api/documents.py` with endpoints:
  - POST `/documents/upload` - Accept PDF files, extract, chunk, embed, store
  - GET `/documents/list` - List uploaded documents with metadata
  - DELETE `/documents/{doc_id}` - Remove document from all indexes
- [ ] Create `backend/app/api/chat.py` with endpoints:
  - POST `/chat/query` - Submit question, retrieve passages, generate answer
  - GET `/chat/conversations` - List active conversations
  - DELETE `/chat/conversations/{conv_id}` - Clear conversation
- [ ] Implement request validation with Pydantic
- [ ] Implement error handling (clear error messages, appropriate status codes)
- [ ] Write integration tests for each endpoint
- [ ] Test file upload size limits and validation
- [ ] Test error paths (missing document, invalid query, Ollama down, etc.)
- [ ] Document all endpoints (path, method, request/response schemas, errors)

**Deliverables**:
- `main.py` with FastAPI app and route definitions
- `api/documents.py` with document endpoints
- `api/chat.py` with chat endpoints
- Pydantic models for requests/responses
- Passing integration tests
- API documentation (can use FastAPI's auto-generated docs)

**Dependencies**: Phase 4 (answer generation)

**Status**: ⏳ Awaiting Phase 4 completion

---

## Phase 6: Streamlit Frontend
**Goal**: Build a user-friendly interface for document upload and multi-turn Q&A with source display.

**Concepts to Learn**:
- Streamlit session state management (persistence within a session)
- Streamlit widgets: file_uploader, chat_message, chat_input, selectbox, expander
- Making HTTP requests from Streamlit to FastAPI
- Async loading indicators and error displays
- Source reference expansion

**Tasks**:
- [ ] Create `frontend/app.py` with Streamlit interface
- [ ] Implement sidebar for document management:
  - File uploader for PDFs
  - List of uploaded documents with document names
  - Delete buttons with confirmation
  - Loading indicators during upload
- [ ] Implement main chat interface:
  - Chat message display (user questions and AI answers)
  - Chat input for user questions
  - Session state to maintain conversation history
  - Expandable source references (document name, page number, passage)
- [ ] Implement error handling and display to user
- [ ] Make HTTP calls to FastAPI backend
- [ ] Test with sample documents and questions
- [ ] Verify source citations are displayed correctly
- [ ] Test conversation isolation (different sessions don't mix)

**Deliverables**:
- `frontend/app.py` with complete Streamlit UI
- Document upload and management
- Chat interface with Q&A
- Source reference display
- Error messages and loading indicators
- No manual testing failures

**Dependencies**: Phase 5 (FastAPI backend)

**Status**: ⏳ Awaiting Phase 5 completion

---

## Phase 7: Document Management and Session Handling
**Goal**: Improve reliability of document deletion, conversation isolation, and error recovery.

**Tasks**:
- [ ] Implement hard deletion: remove documents from ChromaDB and BM25 completely
- [ ] Verify deleted documents don't appear in subsequent queries
- [ ] Implement document metadata storage (file name, upload time, unique ID)
- [ ] Implement conversation state isolation (multi-user sessions don't interfere)
- [ ] Test edge cases: delete while querying, upload same document twice, etc.
- [ ] Implement graceful restart: rebuild BM25 from ChromaDB on app startup
- [ ] Add logging for debugging document operations
- [ ] Write tests for all deletion and isolation scenarios

**Deliverables**:
- Robust document lifecycle management
- Conversation isolation
- Passing tests for edge cases
- Logging for troubleshooting

**Dependencies**: Phase 6 (Streamlit frontend)

**Status**: ⏳ Awaiting Phase 6 completion

---

## Phase 8: Testing and Evaluation
**Goal**: Write comprehensive tests and evaluate retrieval and answer quality.

**Tasks**:
- [ ] Create test suite structure (`tests/test_*.py`)
- [ ] Write retrieval quality tests (accuracy, coverage, ranking correctness)
- [ ] Write answer generation tests (grounding, source citations, missing info handling)
- [ ] Write end-to-end tests (upload → query → get answer)
- [ ] Create test dataset with known documents and questions
- [ ] Measure retrieval metrics: precision, recall, MRR (Mean Reciprocal Rank)
- [ ] Measure answer quality: grounding accuracy, source correctness
- [ ] Document test results
- [ ] Identify failure cases and improvement opportunities

**Deliverables**:
- Comprehensive test suite
- Test report with metrics
- Identified improvement areas
- Pass/fail summary

**Dependencies**: Phase 7 (document management)

**Status**: ⏳ Awaiting Phase 7 completion

---

## Phase 9: Documentation, Polish, and Optional Enhancements
**Goal**: Complete documentation, handle remaining edge cases, and optionally add reranking.

**Tasks**:
- [ ] Complete README.md (installation, usage, troubleshooting)
- [ ] Complete PROJECT_NOTES.md (decisions, learnings, test results)
- [ ] Add inline code comments for non-obvious logic
- [ ] Polish error messages and loading indicators
- [ ] Optional: Implement reranking (FlashRank or similar)
- [ ] Optional: Add conversation export/import
- [ ] Final end-to-end testing
- [ ] Performance profiling and optimization if needed

**Deliverables**:
- Complete documentation
- Polished UI/UX
- Final test results
- Ready-to-run application

**Dependencies**: Phase 8 (testing)

**Status**: ⏳ Awaiting Phase 8 completion

---

## Phase Flowchart

```
Phase 0: Environment
     ↓
Phase 1: PDF Extraction
     ↓
Phase 2: Embeddings + ChromaDB
     ↓
Phase 3: Hybrid Retrieval (BM25 + RRF)
     ↓
Phase 4: Answer Generation (LLM)
     ↓
Phase 5: FastAPI Backend
     ↓
Phase 6: Streamlit Frontend
     ↓
Phase 7: Document & Session Management
     ↓
Phase 8: Testing & Evaluation
     ↓
Phase 9: Documentation & Polish
```

Each phase is sequential and depends on the previous one. Do not skip or reorder phases without explicit approval.
