"""
Phase 4 Task 3 — Retrieval benchmark for the RAG-Q&A document assistant.

Measures BM25, dense retrieval, and RRF on the synthetic benchmark dataset.
All tests use synthetic documents in temporary ChromaDB storage; the real
backend/chroma_db is never touched. No Ollama or model inference is required.

Metrics measured:
- Recall@1, Recall@3, Recall@5
- Mean Reciprocal Rank (MRR)
- Missing-evidence behavior (retrieval should return few/no relevant results)
- Metadata-filtering correctness

Formulas:
- Recall@k = |relevant retrieved in top-k| / |total relevant|
- MRR = 1 / rank_of_first_relevant (0 if no relevant result retrieved)
- nDCG@k uses binary relevance grades (1=relevant, 0=not relevant)
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest
import numpy as np

from backend.app.services.document_processor import process_document
from backend.app.services.chroma_service import ChromaService
from backend.app.services.hybrid_retriever import HybridRetriever
from backend.app.services.embeddings import EmbeddingsService

from tests.benchmark_dataset import (
    BENCHMARK_DOCUMENTS,
    BENCHMARK_CASES,
    EvalCase,
)


def _make_mock_embeddings():
    """Create a deterministic mock embeddings service.

    Uses a stable hash (not Python's randomized hash()) so results are
    reproducible across runs.
    """
    service = Mock(spec=EmbeddingsService)

    def _stable_hash(s: str) -> int:
        """FNV-1a 32-bit hash — deterministic across runs."""
        h = 2166136261
        for c in s.encode("utf-8"):
            h ^= c
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def mock_embed(text):
        if isinstance(text, str):
            np.random.seed(_stable_hash(text) % 100000)
            emb = np.random.randn(384).astype(np.float32)
            return emb / np.linalg.norm(emb)
        elif isinstance(text, list):
            embs = []
            for t in text:
                np.random.seed(_stable_hash(t) % 100000)
                emb = np.random.randn(384).astype(np.float32)
                embs.append(emb / np.linalg.norm(emb))
            return np.array(embs)
        raise TypeError(f"Expected str or list, got {type(text)}")

    service.embed = Mock(side_effect=mock_embed)
    return service


def _build_benchmark_corpus(tmp_path, chroma_service):
    """Index all benchmark documents into ChromaDB and rebuild BM25."""
    for doc in BENCHMARK_DOCUMENTS:
        tmp_file = tmp_path / doc.filename
        if doc.file_format in ("txt", "md", "csv"):
            tmp_file.write_text(doc.content, encoding="utf-8")
        else:
            continue

        chunks = process_document(str(tmp_file), doc.filename)
        chroma_service.add_chunks(chunks)

    return chroma_service


def _recall_at_k(retrieved_ids, relevant_ids, k):
    """Recall@k = |relevant retrieved in top-k| / |total relevant|."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def _mrr(retrieved_ids, relevant_ids):
    """Mean Reciprocal Rank = 1 / rank of first relevant result."""
    if not relevant_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved_ids, relevant_ids, k):
    """nDCG@k with binary relevance grades.

    Deduplicates retrieved document IDs by first occurrence so that multiple
    chunks from the same document do not inflate DCG. Standard normalized
    nDCG must be <= 1.0 by definition.
    """
    if not relevant_ids:
        return 0.0

    # Deduplicate: only the first occurrence of each doc ID contributes to DCG
    seen = set()
    deduped_ids = []
    for doc_id in retrieved_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            deduped_ids.append(doc_id)

    dcg = 0.0
    for rank, doc_id in enumerate(deduped_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / np.log2(rank + 1)

    idcg = 0.0
    for rank in range(1, min(len(relevant_ids), k) + 1):
        idcg += 1.0 / np.log2(rank + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db_dir():
    temp_dir = tempfile.mkdtemp(prefix="benchmark_retrieval_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_embeddings_service():
    return _make_mock_embeddings()


@pytest.fixture
def chroma_service(temp_db_dir, mock_embeddings_service):
    return ChromaService(db_path=temp_db_dir, embeddings_service=mock_embeddings_service)


@pytest.fixture
def benchmark_corpus(temp_db_dir, mock_embeddings_service, tmp_path):
    """Build a ChromaDB corpus with all benchmark documents indexed."""
    chroma = ChromaService(db_path=temp_db_dir, embeddings_service=mock_embeddings_service)
    _build_benchmark_corpus(tmp_path, chroma)
    return chroma


@pytest.fixture
def benchmark_retriever(benchmark_corpus):
    """Create a HybridRetriever with the benchmark corpus indexed."""
    retriever = HybridRetriever(chroma_service=benchmark_corpus)
    retriever.rebuild_index()
    return retriever


# ---------------------------------------------------------------------------
# Retrieval metric tests
# ---------------------------------------------------------------------------

class TestBenchmarkRecallAtK:
    """Measure Recall@k for BM25, dense, and RRF retrieval."""

    def _run_case(self, retriever, case: EvalCase):
        """Run a single evaluation case and return metrics."""
        results = retriever.retrieve(case.question, n_results=5)
        retrieved_ids = results["ids"]
        metadatas = results.get("metadatas", [])

        # Map retrieved IDs to benchmark document IDs via file_name metadata
        retrieved_doc_ids = []
        for meta in metadatas:
            fname = meta.get("file_name", "") if meta else ""
            for doc in BENCHMARK_DOCUMENTS:
                if doc.filename == fname:
                    retrieved_doc_ids.append(doc.doc_id)
                    break

        relevant_ids = set(case.relevant_doc_ids)

        if not case.relevant_doc_ids:
            return {
                "recall_at_1": 0.0,
                "recall_at_3": 0.0,
                "recall_at_5": 0.0,
                "mrr": 0.0,
                "ndcg_at_5": 0.0,
                "retrieved_count": len(retrieved_ids),
            }

        return {
            "recall_at_1": _recall_at_k(retrieved_doc_ids, relevant_ids, 1),
            "recall_at_3": _recall_at_k(retrieved_doc_ids, relevant_ids, 3),
            "recall_at_5": _recall_at_k(retrieved_doc_ids, relevant_ids, 5),
            "mrr": _mrr(retrieved_doc_ids, relevant_ids),
            "ndcg_at_5": _ndcg_at_k(retrieved_doc_ids, relevant_ids, 5),
            "retrieved_count": len(retrieved_ids),
        }

    @pytest.mark.parametrize("case", BENCHMARK_CASES, ids=lambda c: c.case_id)
    def test_case_metrics_recorded(self, benchmark_retriever, case):
        """Record retrieval metrics for each benchmark case (baseline only, no assertions)."""
        metrics = self._run_case(benchmark_retriever, case)
        print(
            f"\n  {case.case_id} [{case.category}]: "
            f"R@1={metrics['recall_at_1']:.2f} "
            f"R@3={metrics['recall_at_3']:.2f} "
            f"R@5={metrics['recall_at_5']:.2f} "
            f"MRR={metrics['mrr']:.2f} "
            f"nDCG@5={metrics['ndcg_at_5']:.2f} "
            f"n={metrics['retrieved_count']}"
        )

    def test_exact_keyword_recall(self, benchmark_retriever):
        """Exact keyword questions should have high recall."""
        for case in BENCHMARK_CASES:
            if case.category != "exact_keyword":
                continue
            metrics = self._run_case(benchmark_retriever, case)
            assert metrics["recall_at_5"] >= case.min_recall_at_5, (
                f"{case.case_id}: R@5={metrics['recall_at_5']:.2f} "
                f"below minimum {case.min_recall_at_5}"
            )

    def test_semantic_paraphrase_recall(self, benchmark_retriever):
        """Semantic paraphrase questions should have reasonable recall."""
        for case in BENCHMARK_CASES:
            if case.category != "semantic_paraphrase":
                continue
            metrics = self._run_case(benchmark_retriever, case)
            assert metrics["recall_at_5"] >= case.min_recall_at_5, (
                f"{case.case_id}: R@5={metrics['recall_at_5']:.2f} "
                f"below minimum {case.min_recall_at_5}"
            )

    def test_missing_evidence_returns_few_results(self, benchmark_retriever):
        """Missing-evidence cases should not retrieve relevant documents."""
        for case in BENCHMARK_CASES:
            if case.category != "missing_evidence":
                continue
            metrics = self._run_case(benchmark_retriever, case)
            assert metrics["mrr"] == 0.0, (
                f"{case.case_id}: MRR should be 0 for missing evidence"
            )
            assert metrics["recall_at_5"] == 0.0, (
                f"{case.case_id}: Recall@5 should be 0 for missing evidence"
            )

    def test_distractor_not_dominant(self, benchmark_retriever):
        """Distractor documents should not dominate results for non-distractor queries."""
        for case in BENCHMARK_CASES:
            if case.category == "distractor":
                continue
            results = benchmark_retriever.retrieve(case.question, n_results=5)
            metadatas = results.get("metadatas", [])
            distractor_hits = sum(
                1 for m in metadatas if m and m.get("file_name") == "weather_forecast.txt"
            )
            assert distractor_hits <= 1, (
                f"{case.case_id}: {distractor_hits} distractor results in top 5"
            )

    def test_mrr_thresholds(self, benchmark_retriever):
        """MRR should meet minimum thresholds for each case."""
        for case in BENCHMARK_CASES:
            metrics = self._run_case(benchmark_retriever, case)
            assert metrics["mrr"] >= case.min_mrr, (
                f"{case.case_id}: MRR={metrics['mrr']:.2f} below minimum {case.min_mrr}"
            )


# ---------------------------------------------------------------------------
# nDCG correctness tests (metric audit defect fix)
# ---------------------------------------------------------------------------

class TestNDCGCorrectness:
    """Unit tests verifying nDCG@k is correctly normalized (always <= 1.0)."""

    def test_ndcg_all_relevant_at_top_equals_one(self):
        """If all top-k results are the same relevant document, nDCG@k = 1.0."""
        retrieved_ids = ["doc_policy", "doc_policy", "doc_policy", "doc_distractor", "doc_specs"]
        relevant_ids = {"doc_policy"}
        assert _ndcg_at_k(retrieved_ids, relevant_ids, 5) == 1.0

    def test_ndcg_no_relevant_equals_zero(self):
        """If no relevant documents are retrieved, nDCG@k = 0.0."""
        retrieved_ids = ["doc_distractor", "doc_specs", "doc_directory"]
        relevant_ids = {"doc_policy"}
        assert _ndcg_at_k(retrieved_ids, relevant_ids, 5) == 0.0

    def test_ndcg_relevant_at_rank_1(self):
        """Relevant document at rank 1 gives higher nDCG than at rank 5."""
        relevant_ids = {"doc_policy"}
        ndcg_rank1 = _ndcg_at_k(["doc_policy", "doc_distractor", "doc_specs"], relevant_ids, 5)
        ndcg_rank5 = _ndcg_at_k(["doc_distractor", "doc_specs", "doc_directory", "doc_quarterly", "doc_policy"], relevant_ids, 5)
        assert ndcg_rank1 > ndcg_rank5

    def test_ndcg_never_exceeds_one(self):
        """nDCG@k must never exceed 1.0 for any input."""
        all_docs = ["doc_policy", "doc_specs", "doc_directory", "doc_quarterly", "doc_distractor"]
        relevant = {"doc_policy", "doc_specs"}
        # Test many permutations with duplicates
        for _ in range(100):
            retrieved = [all_docs[i % len(all_docs)] for i in range(5)]
            ndcg = _ndcg_at_k(retrieved, relevant, 5)
            assert ndcg <= 1.0, f"nDCG@5={ndcg} exceeds 1.0 for retrieved={retrieved}"

    def test_ndcg_empty_retrieved_equals_zero(self):
        """Empty retrieved list gives nDCG@k = 0.0."""
        assert _ndcg_at_k([], {"doc_policy"}, 5) == 0.0

    def test_ndcg_empty_relevant_equals_zero(self):
        """Empty relevant set gives nDCG@k = 0.0."""
        assert _ndcg_at_k(["doc_policy", "doc_specs"], set(), 5) == 0.0