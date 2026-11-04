# Rules: Implementation Boundaries and Constraints

These rules govern how the AI implements the RAG system. They ensure code is simple, maintainable, and aligned with the project vision.

## Technology Stack (Fixed Unless Approved)

**Required:**
- Frontend: Streamlit (native components only)
- Backend: FastAPI + Uvicorn
- RAG Orchestration: LangChain for document processing, splitting, embeddings, prompts, LLM integration
- Vector Database: ChromaDB (persistent local storage)
- Embeddings: Sentence Transformers
- Hybrid Retrieval: Dense (ChromaDB) + Sparse (BM25) + RRF fusion
- LLM: Ollama + Qwen (local, no API keys)
- PDF Processing: PyMuPDF
- Configuration: python-dotenv, Pydantic

**Explicitly NOT allowed without approval:**
- LangGraph (not needed for deterministic RAG pipeline)
- Agent frameworks or tool-calling systems
- Separate frontend framework (React, Angular, Vue, etc.)
- HTML/CSS/JavaScript frontend code
- Multiple embedding models
- Multiple LLM providers
- Agentic workflows

**Rationale**: Simple, self-contained stack. No unnecessary abstractions or frameworks.

## Code Quality Rules

### Minimal Code Rule
Write the smallest, clearest implementation that correctly solves the current task.

- Prefer simple Python functions over unnecessary classes
- Use clear variable names and explicit data flow
- Keep modules small with single responsibilities
- Straightforward error handling (log clearly, fail fast)
- Small, focused tests
- No premature optimization
- No unnecessary dependencies

### No Unnecessary Abstractions
- Don't create classes just to encapsulate a function
- Don't pre-optimize for future features
- Don't add "flexibility" that isn't currently needed
- Don't duplicate logic across files

### Code Style and Conventions
- Follow existing patterns in the codebase
- Use type hints where helpful (not everywhere)
- Use docstrings for functions explaining inputs, outputs, and side effects
- Keep functions under 50 lines when possible
- Clear variable names over cryptic abbreviations

### Testing Rule
- Write tests for each component before integration
- Test PDF extraction independently
- Test embedding generation independently
- Test retrieval independently
- Test answer generation independently
- Run actual tests; never claim tests pass without execution proof
- Never delete databases or documents without explicit approval

## Data and Storage Rules

### ChromaDB and BM25 Consistency
- Keep ChromaDB and BM25 indexes synchronized
- BM25 index must be rebuildable from persisted metadata after restarts
- Document deletions must remove data from both indexes
- Never silently drop chunks or reuse stale data after PDF changes

### Metadata Preservation
- Every chunk must store: original document ID, document name, page number, text
- Every document must store: file name, upload timestamp, unique ID
- Metadata enables source citations and traceability

### Persistent Storage
- ChromaDB data lives in `backend/chroma_db/` (persistent)
- BM25 index is rebuilt on startup from ChromaDB metadata
- No data loss on application restart
- Document deletion is hard (removes from all indexes)

## API and Interface Rules

### Backend/Frontend Separation
- Streamlit communicates with FastAPI via HTTP requests only
- FastAPI owns all RAG logic (pdf processing, embeddings, retrieval, generation)
- Streamlit handles UI state and user input only
- No duplicate RAG logic in the frontend

### Error Messages
- Return clear, actionable errors to the user
- Log detailed errors server-side for debugging
- Never expose stack traces to the user UI
- Distinguish between user errors (bad input) and system errors (service down)

### Response Format
- All API responses are JSON
- Include metadata (source document, page number, confidence/rank if applicable)
- Include error details when queries fail
- Timestamp responses where relevant

## RAG Pipeline Rules

### Retrieval Strategy
- Always use hybrid retrieval (dense + sparse) combined via RRF
- Never retrieve without metadata (documents, pages, chunks must be traceable)
- Retrieve top-k passages; exact number is configurable
- Log retrieval results for debugging and evaluation

### Answer Generation
- Ground all answers in retrieved passages
- If no relevant passages are retrieved, say so explicitly (don't hallucinate)
- Include source citations (document name, page number) in responses
- Use temperature and other LLM parameters conservatively (deterministic behavior preferred)

### Handling Missing Information
- Retrieve empty results → LLM generates "I don't have enough information" response
- Low-confidence retrieval → Still include passages; let LLM decide relevance
- LLM refuses to answer → Return refusal to user, don't retry with different prompt

## Development Process Rules

### Human-in-the-Loop Approval
Before any file modification, package installation, model download, database operation, or code execution:
1. Explain what you intend to do
2. Explain why it's needed
3. Teach the underlying concept
4. Show the proposed minimal code or command
5. Explain what each important part does
6. Identify files that will change and risks
7. Ask for explicit approval
8. Wait for the user's response

Approval for one action does NOT authorize the next action. Ask again before every step.

### No Automatic Execution
- Do not create files or install packages during initial assessment
- Do not download models without approval
- Do not modify databases without confirmation
- Do not commit or push code without approval
- Do not claim tests pass unless actually executed

### Verification
- Run the project's build/test step after code changes
- Never present code as complete without verification
- Report actual test results, not assumed success
- State clearly what was verified and what could not be verified

## Documentation Rules

### Code Comments
- Explain *why*, not *what* (code shows what)
- Focus on non-obvious decisions and gotchas
- Update comments when code changes

### Project Notes
Maintain `PROJECT_NOTES.md` tracking:
- Completed phases and date
- Architectural decisions and reasoning
- Concepts learned and how they apply
- Commands used (for reproducibility)
- Test results (actual output, not just "pass/fail")
- Outstanding issues and blockers

### README
- User-facing guide (how to install and run)
- Example questions and expected behavior
- Troubleshooting section
- Not a tutorial; focus on getting started quickly

## Secrets and Security
- No credentials in code or version control
- Use .env file for sensitive values; .env.example shows structure
- Never log full API responses or sensitive data
- Environment variables via python-dotenv

## Dependency Management
- Use exact or pinned versions in requirements.txt
- Prefer well-known, actively maintained packages
- No typosquatting variants; verify package names
- Install only when approved
- Document why each dependency is needed

## Git Practices
- Push to new branches, never directly to main (unless explicitly asked)
- Commit messages: clear, concise, include context
- Create PRs before merging (code review step)
- One logical change per commit
- Include attribution line per project settings

## When Rules Conflict
- Rules in this document take precedence over general AI preferences
- Architecture.md describes the system; follow it as written
- Phases.md defines work sequence; follow it unless blocked and diagnosed
- PRD.md defines success criteria; build exactly what's described
- If a rule blocks progress, stop and explain the conflict to the user; don't work around it

**The goal is a simple, maintainable, grounded RAG system built incrementally with full user understanding and control.**
