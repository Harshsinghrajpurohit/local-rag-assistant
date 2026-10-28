from fastapi import APIRouter, Request
from ..models import QueryRequest, QueryResponse

router = APIRouter()

@router.post("/chat/query", response_model=QueryResponse)
async def chat_query(request: QueryRequest):
    # Initialize services
    from ..services.chroma_service import ChromaService
    from ..services.hybrid_retriever import HybridRetriever
    from ..services.answer_generator import AnswerGenerator
    from ..services.structured_query import parse_conditions, filter_structured_rows

    chroma_service = ChromaService()
    retriever = HybridRetriever(chroma_service=chroma_service)
    answer_gen = AnswerGenerator()

    # Detect structured query conditions (e.g., "stipend >= 40000 AND duration <= 3")
    conditions = parse_conditions(request.query)

    if conditions:
        # Structured query path: evaluate ALL tabular rows from ChromaDB,
        # not just top-k semantic retrieval results.
        all_results = chroma_service.collection.get(
            include=["documents", "metadatas", "embeddings"]
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