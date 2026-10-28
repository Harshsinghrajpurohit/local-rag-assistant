"""
Test suite for AnswerGenerator service.
Tests answer generation, source citations, token handling, and edge cases.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
import json

from backend.app.exceptions import OllamaUnavailableError
from backend.app.services.answer_generator import AnswerGenerator


@pytest.fixture
def answer_generator():
    """Create AnswerGenerator instance for testing."""
    return AnswerGenerator(
        base_url="http://localhost:11434",
        model="qwen3:4b-instruct",
        max_tokens=500,
        temperature=0.3,
    )


@pytest.fixture
def mock_retriever_results():
    """Mock retriever results from Phase 3."""
    return {
        "ids": ["doc1_chunk1", "doc1_chunk2", "doc2_chunk1"],
        "documents": [
            "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
            "Deep learning uses neural networks with multiple layers to learn complex patterns.",
            "Neural networks are inspired by biological neurons and form the basis of deep learning.",
        ],
        "metadatas": [
            {"page": 1, "file_name": "ai_guide.pdf", "chunk_index": 0},
            {"page": 3, "file_name": "ai_guide.pdf", "chunk_index": 5},
            {"page": 2, "file_name": "dl_basics.pdf", "chunk_index": 2},
        ],
        "scores": {
            "bm25": [8.5, 6.2, 5.1],
            "semantic": [0.85, 0.78, 0.72],
            "rrf": [0.0324, 0.0265, 0.0214],
        },
    }


@pytest.fixture
def empty_retriever_results():
    """Mock retriever results with no documents."""
    return {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "scores": {"bm25": [], "semantic": [], "rrf": []},
    }


# ============================================================================
# Test Answer Generation
# ============================================================================


def test_answer_generation_basic(answer_generator, mock_retriever_results):
    """Test basic answer generation from retrieved passages."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Machine learning is a subset of AI that learns from data.",
            "eval_count": 20,
            "prompt_eval_count": 100,
        }
        mock_post.return_value = mock_response

        result = answer_generator.generate(
            query="What is machine learning?",
            retriever_results=mock_retriever_results,
        )

        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert "tokens_used" in result
        assert "model" in result
        assert "timestamp" in result

        # Verify answer is from mock
        assert (
            "Machine learning" in result["answer"]
        )

        # Verify API was called
        mock_post.assert_called_once()


def test_answer_generation_with_sources(answer_generator, mock_retriever_results):
    """Test that sources are included in response."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Based on the documents, ML is a subset of AI.",
            "eval_count": 15,
            "prompt_eval_count": 95,
        }
        mock_post.return_value = mock_response

        result = answer_generator.generate(
            query="What is machine learning?",
            retriever_results=mock_retriever_results,
        )

        sources = result["sources"]
        assert len(sources) > 0
        assert all("page" in s for s in sources)
        assert all("file" in s for s in sources)
        assert all("excerpt" in s for s in sources)
        assert all("score" in s for s in sources)

        # Verify file names are preserved
        file_names = {s["file"] for s in sources}
        assert "ai_guide.pdf" in file_names or "dl_basics.pdf" in file_names


def test_answer_generation_max_sources(answer_generator, mock_retriever_results):
    """Test that max_sources parameter limits context."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer text",
            "eval_count": 10,
            "prompt_eval_count": 90,
        }
        mock_post.return_value = mock_response

        result = answer_generator.generate(
            query="Question?",
            retriever_results=mock_retriever_results,
            max_sources=2,
        )

        # Should have at most 2 sources
        assert len(result["sources"]) <= 2


def test_answer_generation_token_counting(answer_generator, mock_retriever_results):
    """Test that tokens are counted correctly."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer",
            "eval_count": 25,
            "prompt_eval_count": 150,
        }
        mock_post.return_value = mock_response

        result = answer_generator.generate(
            query="machine learning", retriever_results=mock_retriever_results
        )

        # Should be sum of eval_count and prompt_eval_count
        assert result["tokens_used"] == 175


# ============================================================================
# Test Source Citation Extraction
# ============================================================================


def test_source_extraction_page_numbers(answer_generator, mock_retriever_results):
    """Test that page numbers are correctly extracted."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer",
            "eval_count": 10,
            "prompt_eval_count": 90,
        }
        mock_post.return_value = mock_response

        result = answer_generator.generate(
            query="Q?", retriever_results=mock_retriever_results
        )

        pages = {s["page"] for s in result["sources"]}
        assert pages == {1, 2, 3}  # From metadata


def test_source_extraction_file_names(answer_generator, mock_retriever_results):
    """Test that file names are correctly extracted."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer",
            "eval_count": 10,
            "prompt_eval_count": 90,
        }
        mock_post.return_value = mock_response

        result = answer_generator.generate(
            query="Q?", retriever_results=mock_retriever_results
        )

        files = {s["file"] for s in result["sources"]}
        assert "ai_guide.pdf" in files
        assert "dl_basics.pdf" in files


def test_source_extraction_excerpts(answer_generator, mock_retriever_results):
    """Test that excerpts are trimmed to reasonable length."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer",
            "eval_count": 10,
            "prompt_eval_count": 90,
        }
        mock_post.return_value = mock_response

        result = answer_generator.generate(
            query="Q?", retriever_results=mock_retriever_results
        )

        for source in result["sources"]:
            excerpt = source["excerpt"]
            # Excerpt should be <= 153 chars (150 + "...")
            assert len(excerpt) <= 153


def test_source_extraction_scores(answer_generator, mock_retriever_results):
    """Test that RRF scores are included in sources."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer",
            "eval_count": 10,
            "prompt_eval_count": 90,
        }
        mock_post.return_value = mock_response

        result = answer_generator.generate(
            query="Q?", retriever_results=mock_retriever_results
        )

        scores = [s["score"] for s in result["sources"]]
        # Should match RRF scores from input
        assert scores[0] > scores[1] > scores[2]  # Descending order


# ============================================================================
# Test Confidence Calculation
# ============================================================================


def test_confidence_calculation(answer_generator, mock_retriever_results):
    """Test that confidence is calculated from RRF scores."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer",
            "eval_count": 10,
            "prompt_eval_count": 90,
        }
        mock_post.return_value = mock_response

        result = answer_generator.generate(
            query="machine learning", retriever_results=mock_retriever_results
        )

        # Confidence should be between 0 and 1
        assert 0.0 <= result["confidence"] <= 1.0

        # With high RRF scores and matching terms, confidence should be high
        assert result["confidence"] > 0.5


def test_confidence_zero_for_empty_results(
    answer_generator, empty_retriever_results
):
    """Test that confidence is 0 when no results."""
    result = answer_generator.generate(
        query="Q?", retriever_results=empty_retriever_results
    )

    assert result["confidence"] == 0.0


def test_confidence_single_high_score(answer_generator):
    """Test confidence calculation with single high-scoring result."""
    retriever_results = {
        "ids": ["doc1"],
        "documents": ["High quality passage"],
        "metadatas": [{"page": 1, "file_name": "doc.pdf", "chunk_index": 0}],
        "scores": {"bm25": [9.0], "semantic": [0.95], "rrf": [0.05]},
    }

    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer",
            "eval_count": 10,
            "prompt_eval_count": 90,
        }
        mock_post.return_value = mock_response

        result = answer_generator.generate(
            query="high quality", retriever_results=retriever_results
        )

        # High RRF score should yield high confidence
        assert result["confidence"] > 0.7


# ============================================================================
# Test Empty / No Results Handling
# ============================================================================


def test_empty_retriever_results(answer_generator, empty_retriever_results):
    """Test handling when retriever returns no results."""
    result = answer_generator.generate(
        query="Question about unknown topic?",
        retriever_results=empty_retriever_results,
    )

    # Should return graceful "insufficient context" response
    assert "don't have enough context" in result["answer"].lower()
    assert result["sources"] == []
    assert result["confidence"] == 0.0


def test_none_retriever_results(answer_generator):
    """Test handling when retriever_results is None."""
    result = answer_generator.generate(query="Q?", retriever_results=None)

    assert "don't have enough context" in result["answer"].lower()
    assert result["sources"] == []


def test_retriever_results_no_documents(answer_generator):
    """Test handling when retriever_results has no documents key."""
    retriever_results = {
        "ids": [],
        "metadatas": [],
        "scores": {},
    }
    result = answer_generator.generate(query="Q?", retriever_results=retriever_results)

    assert "don't have enough context" in result["answer"].lower()


# ============================================================================
# Test Prompt Formatting
# ============================================================================


def test_prompt_includes_context(answer_generator):
    """Test that prompt includes retrieved passages."""
    retriever_results = {
        "ids": ["doc1"],
        "documents": ["Important information about topic X"],
        "metadatas": [{"page": 1, "file_name": "test.pdf", "chunk_index": 0}],
        "scores": {"bm25": [8.0], "semantic": [0.85], "rrf": [0.03]},
    }

    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer",
            "eval_count": 10,
            "prompt_eval_count": 90,
        }
        mock_post.return_value = mock_response

        answer_generator.generate(query="topic X", retriever_results=retriever_results)

        # Check that prompt was sent to Ollama
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        prompt = payload["prompt"]

        # Prompt should include document content and source attribution
        assert "Important information" in prompt
        assert "test.pdf" in prompt


def test_prompt_includes_query(answer_generator):
    """Test that prompt includes user's question."""
    retriever_results = {
        "ids": ["doc1"],
        "documents": ["Some passage"],
        "metadatas": [{"page": 1, "file_name": "test.pdf", "chunk_index": 0}],
        "scores": {"bm25": [8.0], "semantic": [0.85], "rrf": [0.03]},
    }

    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer",
            "eval_count": 10,
            "prompt_eval_count": 90,
        }
        mock_post.return_value = mock_response

        question = "What does some passage say?"
        answer_generator.generate(
            query=question, retriever_results=retriever_results
        )

        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        prompt = payload["prompt"]

        # Prompt should include the question
        assert question in prompt


# ============================================================================
# Test API Parameters
# ============================================================================


def test_ollama_api_parameters(answer_generator, mock_retriever_results):
    """Test that correct parameters are sent to Ollama."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer",
            "eval_count": 10,
            "prompt_eval_count": 90,
        }
        mock_post.return_value = mock_response

        answer_generator.generate(query="machine learning", retriever_results=mock_retriever_results)

        # Verify API was called with correct parameters
        call_args = mock_post.call_args
        url = call_args[0][0]
        payload = call_args[1]["json"]

        assert "generate" in url
        assert payload["model"] == "qwen3:4b-instruct"
        assert payload["stream"] is False
        # Generation parameters must live inside Ollama's "options" object
        assert payload["options"]["temperature"] == 0.3
        assert payload["options"]["num_predict"] == 500
        assert "temperature" not in payload
        assert "prompt" in payload


def test_ollama_timeout(answer_generator, mock_retriever_results):
    """Test handling of Ollama API timeout."""
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout("Connection timeout")

        with pytest.raises(OllamaUnavailableError) as exc_info:
            answer_generator.generate(
                query="machine learning", retriever_results=mock_retriever_results
            )

        assert exc_info.value.status_code == 503


def test_ollama_connection_error(answer_generator, mock_retriever_results):
    """Test handling of Ollama connection error."""
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError(
            "Cannot connect to Ollama"
        )

        with pytest.raises(OllamaUnavailableError) as exc_info:
            answer_generator.generate(
                query="machine learning", retriever_results=mock_retriever_results
            )

        assert exc_info.value.status_code == 503


# ============================================================================
# Test Response Structure
# ============================================================================


def test_response_structure(answer_generator, mock_retriever_results):
    """Test that response has all required fields."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer",
            "eval_count": 10,
            "prompt_eval_count": 90,
        }
        mock_post.return_value = mock_response

        result = answer_generator.generate(
            query="Q?", retriever_results=mock_retriever_results
        )

        # Check all required fields
        required_fields = [
            "answer",
            "sources",
            "confidence",
            "tokens_used",
            "model",
            "timestamp",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"


def test_response_types(answer_generator, mock_retriever_results):
    """Test that response fields have correct types."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer",
            "eval_count": 10,
            "prompt_eval_count": 90,
        }
        mock_post.return_value = mock_response

        result = answer_generator.generate(
            query="Q?", retriever_results=mock_retriever_results
        )

        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["tokens_used"], int)
        assert isinstance(result["model"], str)
        assert isinstance(result["timestamp"], str)


# ============================================================================
# Test Configuration
# ============================================================================


def test_custom_base_url():
    """Test that custom base URL is used."""
    gen = AnswerGenerator(
        base_url="http://custom-ollama:11434",
        model="custom-model",
    )

    assert gen.base_url == "http://custom-ollama:11434"
    assert gen.model == "custom-model"
    assert "custom-ollama" in gen.generate_url


def test_default_config():
    """Test that defaults are applied when not specified."""
    gen = AnswerGenerator()

    assert gen.max_tokens == 500
    assert gen.temperature == 0.3
    assert "qwen3" in gen.model.lower() or "qwen" in gen.model.lower()


# ============================================================================
# Integration Tests
# ============================================================================


def test_end_to_end_answer_generation(answer_generator, mock_retriever_results):
    """End-to-end test of answer generation pipeline."""
    with patch("requests.post") as mock_post:
        # Simulate realistic Ollama response
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": (
                "Machine learning is a subset of artificial intelligence that enables "
                "systems to learn and improve from experience without being explicitly programmed. "
                "It focuses on the development of algorithms and statistical models."
            ),
            "eval_count": 45,
            "prompt_eval_count": 185,
        }
        mock_post.return_value = mock_response

        result = answer_generator.generate(
            query="What is machine learning and how does it work?",
            retriever_results=mock_retriever_results,
            max_sources=3,
        )

        # Verify complete response
        assert len(result["answer"]) > 0
        assert len(result["sources"]) == 3
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["tokens_used"] == 230  # 45 + 185
        assert "qwen" in result["model"].lower()


def test_multiple_sequential_calls(answer_generator, mock_retriever_results):
    """Test that multiple calls work independently."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Answer",
            "eval_count": 10,
            "prompt_eval_count": 90,
        }
        mock_post.return_value = mock_response

        # Call twice
        result1 = answer_generator.generate(
            query="What is machine learning?", retriever_results=mock_retriever_results
        )
        result2 = answer_generator.generate(
            query="What is deep learning?", retriever_results=mock_retriever_results
        )

        # Both should succeed
        assert result1["answer"]
        assert result2["answer"]
        assert mock_post.call_count == 2


# ============================================================================
# Abstention and Confidence Tests (Phase 10)
# ============================================================================


def test_irrelevant_results_abstention(answer_generator):
    """Retrieved documents exist but none contain query terms → abstain."""
    irrelevant_docs = [
        "Employees are entitled to 20 days of paid annual leave per calendar year.",
        "Sick leave is accrued at 1 day per month, up to 12 days per year.",
        "Meal expenses are capped at $75 per day.",
        "Remote work is allowed up to 3 days per week.",
        "Equipment stipend of $500 is available for home office setup.",
    ]
    metas = [
        {"file_name": "employee_handbook.txt", "page": None, "section": "Leave Policy"}
    ] * len(irrelevant_docs)
    results = {
        "ids": [f"doc_{i}" for i in range(len(irrelevant_docs))],
        "documents": irrelevant_docs,
        "metadatas": metas,
        "distances": [0.1] * len(irrelevant_docs),
        "bm25_scores": [0.0] * len(irrelevant_docs),
        "rrf_scores": [0.05] * len(irrelevant_docs),
    }

    response = answer_generator.generate(
        query="What is the company's dress code policy?",
        retriever_results=results,
        max_sources=5,
    )

    assert response["confidence"] < 0.3, (
        f"Expected low confidence for irrelevant results, got {response['confidence']}"
    )
    # Sources are extracted before the abstention check so that source
    # truncation can be verified independently of LLM generation. Even when
    # evidence is insufficient, retrieved sources are still returned for
    # transparency (up to max_sources).
    assert len(response["sources"]) == 5, (
        f"Expected 5 sources (max_sources=5), got {len(response['sources'])}"
    )
    answer_lower = response["answer"].lower()
    assert "don't have enough information" in answer_lower or (
        "not contain relevant evidence" in answer_lower
    ), f"Expected abstention message, got: {response['answer']}"


def test_missing_evidence_abstention(answer_generator):
    """Case with relevant_doc_ids=[] (missing evidence) → no false facts generated."""
    irrelevant_docs = [
        "QuantumEdge QX-500 features a 32-core Xeon Platinum processor.",
        "The system includes 512 GB of DDR5 ECC memory.",
        "Storage is 15.36 TB NVMe SSD.",
    ]
    metas = [
        {"file_name": "product_specs.md", "page": None, "section": "Overview"}
    ] * len(irrelevant_docs)
    results = {
        "ids": [f"doc_{i}" for i in range(len(irrelevant_docs))],
        "documents": irrelevant_docs,
        "metadatas": metas,
        "distances": [0.1] * len(irrelevant_docs),
        "bm25_scores": [0.0] * len(irrelevant_docs),
        "rrf_scores": [0.05] * len(irrelevant_docs),
    }

    response = answer_generator.generate(
        query="What is the CEO's salary?",
        retriever_results=results,
        max_sources=5,
    )

    assert response["confidence"] < 0.3, (
        f"Expected low confidence for missing evidence, got {response['confidence']}"
    )
    answer_lower = response["answer"].lower()
    assert "$" not in response["answer"], (
        f"Should not fabricate salary info, got: {response['answer']}"
    )
    assert "don't have enough information" in answer_lower or (
        "not contain relevant evidence" in answer_lower
    ), f"Should abstain rather than discuss CEO salary, got: {response['answer']}"


def test_confidence_zero_for_irrelevant(answer_generator):
    """Confidence should approach 0.0 when evidence_sufficiency is 0."""
    irrelevant_docs = [
        "Employees are entitled to 20 days of paid annual leave per calendar year.",
        "Sick leave is accrued at 1 day per month.",
    ]
    metas = [
        {"file_name": "employee_handbook.txt", "page": None}
    ] * len(irrelevant_docs)
    results = {
        "ids": ["doc_0", "doc_1"],
        "documents": irrelevant_docs,
        "metadatas": metas,
        "distances": [0.1, 0.2],
        "bm25_scores": [0.0, 0.0],
        "rrf_scores": [0.05, 0.04],
    }

    response = answer_generator.generate(
        query="What is the weather forecast for Mars?",
        retriever_results=results,
        max_sources=5,
    )

    assert response["confidence"] == 0.0, (
        f"Expected confidence 0.0 for zero evidence overlap, got {response['confidence']}"
    )


def test_supported_question_not_abstained(answer_generator):
    """When relevant evidence is present, system should generate a normal answer."""
    relevant_docs = [
        "Employees are entitled to 20 days of paid annual leave per calendar year.",
        "Meal expenses are capped at $75 per day.",
    ]
    metas = [
        {"file_name": "employee_handbook.txt", "page": None, "section": "Leave Policy"}
    ] * len(relevant_docs)
    results = {
        "ids": ["doc_0", "doc_1"],
        "documents": relevant_docs,
        "metadatas": metas,
        "distances": [0.1, 0.2],
        "bm25_scores": [0.0, 0.0],
        "rrf_scores": [0.05, 0.04],
    }

    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Employees get 20 days of leave and meal expenses are capped at $75 per day.",
            "eval_count": 20,
            "prompt_eval_count": 100,
        }
        mock_post.return_value = mock_response

        response = answer_generator.generate(
            query="What are the leave and meal policies?",
            retriever_results=results,
            max_sources=5,
        )

    mock_post.assert_called_once()
    assert response["confidence"] > 0.3, (
        f"Expected reasonable confidence for supported query, got {response['confidence']}"
    )
    assert "20 days" in response["answer"]
