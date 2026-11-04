from fastapi import APIRouter, Request
from ..models import QueryRequest, QueryResponse

router = APIRouter()

# Lazy-initialized service instances
_chroma_service = None
_hybrid_retriever = None
_answer_generator = None

def _get_chroma_service():
    global _chroma_service
    if _chroma_service is None:
        from ..services.chroma_service import ChromaService
        _chroma_service = ChromaService()
    return _chroma_service

def _get_hybrid_retriever():
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from ..services.chroma_service import ChromaService
        from ..services.hybrid_retriever import HybridRetriever
        _chroma_svc = _get_chroma_service()
        _hybrid_retriever = HybridRetriever(chroma_service=_chroma_svc)
    return _hybrid_retriever


def get_retriever():
    """Shared getter - returns the same HybridRetriever instance across requests."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from ..services.chroma_service import ChromaService
        from ..services.hybrid_retriever import HybridRetriever
        _chroma_svc = _get_chroma_service()
        _hybrid_retriever = HybridRetriever(chroma_service=_chroma_svc)
    return _hybrid_retriever


def reset_retriever():
    """Reset the retriever instance (use in test setup/teardown)."""
    global _hybrid_retriever
    _hybrid_retriever = None

def _get_answer_generator():
    global _answer_generator
    if _answer_generator is None:
        from ..services.answer_generator import AnswerGenerator
        _answer_generator = AnswerGenerator()
    return _answer_generator

@router.post("/chat/query", response_model=QueryResponse)
async def chat_query(request: QueryRequest):
    # Initialize services (lazy - use getters)
    # chroma_service, retriever, and answer_gen are obtained via _get_* getters
    from ..services.structured_query import parse_conditions, filter_structured_rows
    chroma_service = _get_chroma_service()
    retriever = get_retriever()
    answer_gen = _get_answer_generator()

    # Detect structured query conditions (e.g., "stipend >= 40000 AND duration <= 3")
    conditions = parse_conditions(request.query)

    if conditions:
        # Structured query path: evaluate ALL tabular rows from ChromaDB,
        # not just top-k semantic retrieval results.
        all_results = chroma_service.collection.get(
            include=["documents", "metadatas"]
        )
        documents = all_results.get("documents", [])
        metadatas = all_results.get("metadatas", [])
        ids = all_results.get("ids", [])

        matching_docs, matching_metas, matching_ids = filter_structured_rows(
            documents, metadatas, ids, conditions
        )

        if not matching_docs:
            # No rows satisfy all conditions — return insufficient evidence
            return answer_gen._insufficient_evidence_response(
                request.query, 0.0, []
            )

        retriever_response = {
            "ids": matching_ids,
            "documents": matching_docs,
            "metadatas": matching_metas,
            "distances": [1.0] * len(matching_docs),
            "bm25_scores": [0.0] * len(matching_docs),
            "rrf_scores": [0.03] * len(matching_docs),
        }
    else:
        # Normal retrieval path (BM25 + dense + RRF fusion)
        retriever_response = retriever.retrieve(
            request.query, n_results=request.max_sources
        )

    # Generate answer
    response = answer_gen.generate(request.query, retriever_response, request.max_sources)
    return response

@router.get("/chat/conversations")
async def get_conversations():
    return []

@router.delete("/chat/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    return {"status": "deleted"}