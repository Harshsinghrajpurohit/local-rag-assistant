# Project Notes: Progress, Decisions, and Learnings

**Project**: Hybrid RAG Application for AI-Powered Document Q&A
**Started**: 2026-10-12
**Status**: Pre-implementation (documentation phase complete)

---

## Completed Phases

### Documentation Setup (2026-10-12)
- ✅ Created PRD.md (project requirements and vision)
- ✅ Created Architecture.md (system design and data flow)
- ✅ Created Rules.md (implementation boundaries)
- ✅ Created Phases.md (development phases and sequences)
- ✅ Created Design.md (UI/UX guidelines)
- ✅ Created Memory.md (session memory index)
- ✅ Created PROJECT_NOTES.md (this file)

---

## Architectural Decisions

### 1. Hybrid Retrieval (Dense + Sparse + RRF)
**Decision**: Combine ChromaDB semantic search with BM25 keyword search via Reciprocal Rank Fusion
**Rationale**: 
- Dense retrieval excels at paraphrased questions ("What's the return policy?" vs "How do I send something back?")
- Sparse retrieval excels at exact-match lookups ("30 days" appears in policy)
- RRF combines both without requiring a unified scoring function
- Empirically outperforms either method alone for real-world queries

### 2. LangChain Without LangGraph
**Decision**: Use LangChain for document processing, splitting, embeddings, prompts; skip LangGraph
**Rationale**:
- MVP is a deterministic RAG pipeline, not an agent system
- No dynamic routing, tool-calling, or conditional logic needed
- LangGraph adds complexity without solving current problems
- Can add LangGraph later if agentic features become necessary

### 3. Backend/Frontend Separation
**Decision**: Streamlit frontend talks to FastAPI backend via HTTP; no RAG logic in frontend
**Rationale**:
- Keeps concerns separate (UI vs. data/retrieval)
- Allows independent scaling
- Simplifies testing (API can be tested without frontend)
- Easier to swap frontend later (e.g., for a web app)

### 4. Persistent Storage Strategy
**Decision**: ChromaDB persistent; BM25 rebuilt from metadata on startup
**Rationale**:
- ChromaDB handles its own persistence natively
- BM25 index can be reconstructed from chunk metadata (no data loss)
- Avoids serializing BM25 to disk (add complexity)
- Ensures consistency: BM25 always built from current ChromaDB state

### 5. No Reranking in MVP
**Decision**: MVP uses RRF fusion only; reranking scheduled for Phase 9 (optional)
**Rationale**:
- RRF already combines retrieval effectively
- Reranking adds another LLM call (latency, tokens)
- Defer until retrieval quality is proven insufficient
- Optional: can add FlashRank later without changing architecture

### 6. Local LLM (Ollama + Qwen)
**Decision**: Use Ollama with qwen3:4b-instruct locally, no external APIs
**Rationale**:
- No API keys or rate limits
- Reproducible results (same model, local environment)
- Low latency
- Educational: understand the full stack without cloud dependencies
- Cost: runs on user's hardware

### 7. Minimal Code Philosophy
**Decision**: Smallest clear implementation; no premature abstractions
**Rationale**:
- Easier to understand and modify
- Fewer bugs (less code)
- Can refactor once patterns emerge
- Avoid "frameworks solving problems we don't have"

---

## Key Concepts (Learned as We Build)

### Reciprocal Rank Fusion (RRF)
When we implement Phase 3, we'll learn:
- Why simple averaging of BM25 and semantic scores doesn't work well
- How RRF normalizes ranking lists from different sources
- Formula: RRF = Σ (1 / (k + rank)) for each result in a list
- Advantages: no common scoring required, robust to outliers

### Embeddings and Semantic Search
When we implement Phase 2, we'll learn:
- How text becomes vectors (Sentence Transformers)
- Why similar meanings produce similar vectors (cosine distance)
- Why dense search finds paraphrased questions but not exact phrases
- Token limits and embedding dimensions

### BM25 and Keyword Ranking
When we implement Phase 3, we'll learn:
- How BM25 (Okapi algorithm) scores term relevance
- Why term frequency and document length matter
- Why inverse document frequency (IDF) prevents common words from dominating
- Weaknesses: can't handle synonyms or paraphrases

### Chunking and Context
When we implement Phase 1, we'll learn:
- Why fixed-size chunks aren't optimal
- How overlap prevents losing information at boundaries
- Trade-offs: larger chunks have more context but lower granularity
- Metadata: every chunk must know its source

---

## Test Results

### Phase 0: Environment Verification
[Awaiting execution]

### Phase 1: PDF Extraction
[Awaiting implementation]

---

## Commands Used

[None yet; will be recorded as we build]

---

## Outstanding Issues

### None currently

### Known Risks
1. Ollama might not have the exact model name (`qwen3:4b-instruct`) installed
   - **Mitigation**: Phase 0 will verify actual model name before proceeding
   - **Impact**: May need to use different model or download it first

2. BM25 index consistency if ChromaDB is modified outside the app
   - **Mitigation**: Always rebuild BM25 from ChromaDB on startup
   - **Impact**: Adds startup latency (negligible for typical doc sets)

3. LLM hallucination if retrieval returns no relevant passages
   - **Mitigation**: Explicitly prompt LLM to say "I don't know" when evidence is insufficient
   - **Impact**: May require prompt tuning after testing

---

## Performance Targets (Aspirational)

- **Upload 10 PDFs**: < 30 seconds
- **Query response time**: < 5 seconds (retrieve + generate)
- **Retrieval recall**: > 0.8 (finding 80%+ of relevant passages)
- **Answer accuracy**: Grounded in retrieved passages, no hallucinations

---

## Future Enhancements (Post-MVP)

1. **Reranking** (Phase 9): Cross-encoder reranking to improve precision
2. **Multi-model support**: Allow switching between different LLMs
3. **Query expansion**: Rephrase user questions to improve retrieval
4. **Conversation summarization**: Condense long conversations for context
5. **Analytics**: Track query success rates, user feedback
6. **Fine-tuning**: Adapt embeddings or LLM to domain-specific questions
7. **Multi-turn context**: Better handling of pronouns and references in follow-up questions

---

## Learning Goals

By the end of this project, we'll understand:
- How RAG pipelines work end-to-end
- Trade-offs between dense and sparse retrieval
- How embeddings represent meaning
- How LLMs use context to generate grounded answers
- How to build maintainable, testable data pipelines
- How to evaluate retrieval and generation quality
- Why simple designs are often superior to complex ones

---

## Questions for Future Phases

1. **Phase 1**: How does PyMuPDF handle complex PDFs (scanned images, form fields)?
2. **Phase 2**: Which Sentence Transformer model balances quality and speed best?
3. **Phase 3**: What's the optimal k value for top-k retrieval?
4. **Phase 4**: How sensitive is the LLM to prompt phrasing?
5. **Phase 6**: How do we handle very long source passages in the UI?
6. **Phase 8**: What metrics best evaluate retrieval and answer quality?

---

**Next milestone**: Phase 0 approval and execution (verify Python, Ollama, project structure)
