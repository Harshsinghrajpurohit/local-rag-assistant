# Project Requirements Document (PRD)
## Hybrid RAG Application for AI-Powered Document Q&A

### Project Vision
Build a complete Hybrid RAG (Retrieval-Augmented Generation) application that allows users to upload PDF documents and ask questions about them. The system combines semantic search (dense embeddings) with keyword search (BM25) to retrieve the most relevant passages, then uses an LLM to generate grounded answers with source citations.

### Target Users
- Knowledge workers who need to query large document collections
- Users who want answers grounded in specific source documents
- Teams managing internal documentation and FAQs
- Anyone who needs reliable, traceable answers rather than hallucinated content

### Core Features (MVP)

**Document Management**
1. Upload multiple PDF documents
2. View list of uploaded documents
3. Delete documents from the system
4. Persistent storage across application restarts

**Document Processing**
1. Extract text from PDFs while preserving page numbers and document metadata
2. Split documents into overlapping chunks for better context
3. Generate dense embeddings using Sentence Transformers
4. Store embeddings and metadata in ChromaDB (persistent)
5. Build and maintain BM25 keyword indexes

**Retrieval**
1. Perform dense semantic retrieval via ChromaDB
2. Perform sparse keyword retrieval via BM25
3. Combine results using Reciprocal Rank Fusion (RRF)
4. Return ranked, deduplicated passages with metadata

**Question Answering**
1. Accept multi-turn chat questions
2. Retrieve relevant passages using hybrid retrieval
3. Generate answers using Qwen LLM (via Ollama)
4. Ground answers in retrieved passages
5. Display source document names, page numbers, and supporting passages
6. Handle insufficient evidence gracefully (no hallucinations)

**Conversation Management**
1. Support isolated chat sessions
2. Maintain conversation history within a session
3. Clear conversation state when needed

**User Experience**
1. Clean Streamlit interface for document upload and Q&A
2. Expandable source reference display
3. Clear loading indicators during retrieval and generation
4. Useful error messages for debugging and UX

### Success Criteria
- Answers are grounded in retrieved passages
- Users can verify answers by viewing source documents and page numbers
- System handles missing information without inventing answers
- Application runs locally with minimal setup
- All retrieval steps are traceable (what documents were retrieved, what passages matched)

### Out of Scope (Initial Release)
- Advanced reranking (planned for Phase 9)
- Multi-document summarization
- Agentic workflows or tool-calling systems
- Web-scale deployments
- User authentication or multi-tenant isolation
