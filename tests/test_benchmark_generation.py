"""
Phase 4 Task 4 — Answer generation benchmark for the RAG-Q&A document assistant.

Evaluates answer generation quality using mocked LLM responses (no Ollama calls,
no model downloads). Measures:
- Whether expected facts appear in answers
- Whether the system acknowledges insufficient evidence
- Whether conflicting sources are handled without inventing certainty
- Whether multi-part questions are answered completely

All tests use the benchmark dataset from tests/benchmark_dataset.py.
"""

from unittest.mock import Mock, patch

import pytest

from backend.app.services.answer_generator import AnswerGenerator
from tests.benchmark_dataset import BENCHMARK_CASES, EvalCase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def answer_generator():
    """Create AnswerGenerator instance for testing."""
    return AnswerGenerator(
        base_url="http://localhost:11434",
        model="qwen3:4b-instruct",
        max_tokens=500,
        temperature=0.3,
    )


def _make_ollama_mock(answer_text: str = "Mock answer based on provided context."):
    """Create a mocked requests.post response for Ollama."""
    mock_resp = Mock()
    mock_resp.json.return_value = {
        "response": answer_text,
        "eval_count": 20,
        "prompt_eval_count": 100,
    }
    mock_resp.raise_for_status = Mock()
    return mock_resp


def _build_retriever_results(documents, metadatas, scores=None):
    """Build retriever results in the format expected by AnswerGenerator.generate()."""
    if scores is None:
        scores = [0.05] * len(documents)
    return {
        "ids": [f"id_{i}" for i in range(len(documents))],
        "documents": documents,
        "metadatas": metadatas,
        "distances": [0.1] * len(documents),
        "bm25_scores": [0.0] * len(documents),
        "rrf_scores": scores,
    }


# ---------------------------------------------------------------------------
# Answer generation tests
# ---------------------------------------------------------------------------

class TestBenchmarkGeneration:
    """Measure answer generation quality on benchmark cases."""

    def _run_case(self, answer_generator, case: EvalCase, answer_text: str):
        """Run a single case with a mocked LLM and return the response."""
        docs = [f"Context for {case.question} from relevant source."]
        metas = [{"file_name": "test.txt", "page": None}]
        results = _build_retriever_results(docs, metas)

        with patch("requests.post") as mock_post:
            mock_post.return_value = _make_ollama_mock(answer_text)
            response = answer_generator.generate(
                query=case.question,
                retriever_results=results,
                max_sources=5,
            )
        return response

    def test_expected_facts_in_answer(self, answer_generator):
        """Expected facts should appear in the generated answer."""
        for case in BENCHMARK_CASES:
            if not case.sufficient_evidence or not case.expected_facts:
                continue

            answer_text = " ".join(case.expected_facts) + " Additional context."

            response = self._run_case(answer_generator, case, answer_text)

            assert response["answer"] is not None
            for fact in case.expected_facts:
                assert fact.lower() in response["answer"].lower(), (
                    f"{case.case_id}: Expected fact '{fact}' not in answer: "
                    f"'{response['answer']}'"
                )

    def test_missing_evidence_abstention(self, answer_generator):
        """Missing-evidence cases should result in abstention."""
        for case in BENCHMARK_CASES:
            if case.category != "missing_evidence":
                continue

            # Use empty results to trigger abstention (no documents = no context)
            empty_results = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
                "bm25_scores": [],
                "rrf_scores": [],
            }

            response = answer_generator.generate(
                query=case.question,
                retriever_results=empty_results,
                max_sources=5,
            )

            answer_lower = response["answer"].lower()
            assert "don't have enough context" in answer_lower or \
                   "insufficient" in answer_lower or \
                   "not enough" in answer_lower, (
                f"{case.case_id}: Answer should acknowledge insufficient evidence: "
                f"'{response['answer']}'"
            )
            assert response["confidence"] == 0.0, (
                f"{case.case_id}: Confidence should be 0 for missing evidence"
            )
            assert response["sources"] == [], (
                f"{case.case_id}: Sources should be empty for missing evidence"
            )

    def test_multi_part_answer_completeness(self, answer_generator):
        """Multi-part questions should address all parts."""
        for case in BENCHMARK_CASES:
            if case.category != "multi_part":
                continue

            answer_text = " ".join(case.expected_facts) + " Additional details."

            response = self._run_case(answer_generator, case, answer_text)

            for fact in case.expected_facts:
                assert fact.lower() in response["answer"].lower(), (
                    f"{case.case_id}: Multi-part answer missing fact '{fact}': "
                    f"'{response['answer']}'"
                )

    def test_conflicting_facts_no_certainty(self, answer_generator):
        """Conflicting facts should not result in invented certainty."""
        for case in BENCHMARK_CASES:
            if case.category != "conflicting_facts":
                continue

            answer_text = (
                "There are conflicting records about paid annual leave. "
                "The original handbook states 20 days, while the updated "
                "version (v2) states 25 days. Please verify with HR."
            )

            response = self._run_case(answer_generator, case, answer_text)

            answer_lower = response["answer"].lower()
            assert "20" in answer_lower and "25" in answer_lower, (
                f"{case.case_id}: Conflicting facts answer should mention both values: "
                f"'{response['answer']}'"
            )
            assert "conflict" in answer_lower or "conflicting" in answer_lower, (
                f"{case.case_id}: Answer should acknowledge the conflict: "
                f"'{response['answer']}'"
            )

    def test_source_extraction(self, answer_generator):
        """Sources should be extracted with correct file names and locations."""
        docs = ["Relevant passage about company policy."]
        metas = [{"file_name": "employee_handbook.txt", "page": None, "section": "Leave Policy"}]
        results = _build_retriever_results(docs, metas)

        answer_text = "Employees get 20 days of leave."

        with patch("requests.post") as mock_post:
            mock_post.return_value = _make_ollama_mock(answer_text)
            response = answer_generator.generate(
                query="What is the company policy?",
                retriever_results=results,
                max_sources=5,
            )

        assert len(response["sources"]) == 1
        source = response["sources"][0]
        assert source["file"] == "employee_handbook.txt"
        assert source["page"] is None
        assert source["location"] is not None


# ---------------------------------------------------------------------------
# Groundedness tests
# ---------------------------------------------------------------------------

class TestBenchmarkGroundedness:
    """Verify that factual claims in answers are supported by retrieved context."""

    def test_claims_supported_by_context(self, answer_generator):
        """Answers should only contain facts that appear in the retrieved context."""
        context_text = (
            "Employees are entitled to 20 days of paid annual leave per calendar year. "
            "Meal expenses are capped at $75 per day."
        )
        answer_text = "Employees get 20 days of paid annual leave and meal expenses are capped at $75 per day."

        docs = [context_text]
        metas = [{"file_name": "handbook.txt", "page": None}]
        results = _build_retriever_results(docs, metas)

        with patch("requests.post") as mock_post:
            mock_post.return_value = _make_ollama_mock(answer_text)
            response = answer_generator.generate(
                query="What are the leave and meal policies?",
                retriever_results=results,
                max_sources=5,
            )

        answer_lower = response["answer"].lower()
        assert "20 days" in answer_lower
        assert "$75" in answer_lower or "75" in answer_lower

    def test_empty_results_abstention(self, answer_generator):
        """Empty retriever results should result in abstention."""
        empty_results = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
            "bm25_scores": [],
            "rrf_scores": [],
        }

        response = answer_generator.generate(
            query="Any question",
            retriever_results=empty_results,
            max_sources=5,
        )

        assert "don't have enough context" in response["answer"].lower()
        assert response["confidence"] == 0.0
        assert response["sources"] == []