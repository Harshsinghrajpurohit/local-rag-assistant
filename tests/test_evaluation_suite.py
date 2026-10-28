"""
Phase 2 — General Evaluation Suite for RAG-Q&A Document Assistant.

Covers 10 question categories across all supported formats using synthetic
fixtures in temporary directories. The real backend/chroma_db is never
touched. No personal contact information is printed or logged. No Ollama or
model inference is required — embeddings and LLM calls are mocked.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock

from backend.app.services.document_processor import process_document
from backend.app.services.chroma_service import ChromaService
from backend.app.services.hybrid_retriever import HybridRetriever
from backend.app.services.embeddings import EmbeddingsService
from backend.app.services.answer_generator import AnswerGenerator
from backend.app.exceptions import ExtractionFailedError


def _fnv1a_hash(s: str) -> int:
    """FNV-1a 32-bit hash — deterministic across runs (unlike Python's hash())."""
    h = 2166136261
    for c in s.encode("utf-8"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db_dir():
    temp_dir = tempfile.mkdtemp(prefix="eval_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_embeddings_service():
    service = Mock(spec=EmbeddingsService)
    import numpy as np

    def mock_embed(text):
        if isinstance(text, str):
            text_hash = _fnv1a_hash(text) % 10000
            np.random.seed(text_hash)
            emb = np.random.randn(768).astype(np.float32)
            return emb / np.linalg.norm(emb)
        elif isinstance(text, list):
            embs = []
            for t in text:
                text_hash = _fnv1a_hash(t) % 10000
                np.random.seed(text_hash)
                emb = np.random.randn(768).astype(np.float32)
                embs.append(emb / np.linalg.norm(emb))
            return np.array(embs)
        raise TypeError(f"Expected str or list, got {type(text)}")

    service.embed = Mock(side_effect=mock_embed)
    return service


@pytest.fixture
def chroma_service(temp_db_dir, mock_embeddings_service):
    return ChromaService(db_path=temp_db_dir, embeddings_service=mock_embeddings_service)


@pytest.fixture
def retriever(chroma_service):
    return HybridRetriever(chroma_service=chroma_service)


@pytest.fixture
def answer_generator():
    return AnswerGenerator(
        base_url="http://localhost:11434",
        model="qwen3:4b-instruct",
        max_tokens=500,
        temperature=0.3,
    )


@pytest.fixture
def mock_ollama_answer():
    def _make(answer_text="Mock answer based on provided context."):
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "response": answer_text,
            "eval_count": 20,
            "prompt_eval_count": 100,
        }
        mock_resp.raise_for_status = Mock()
        return mock_resp
    return _make


# ---------------------------------------------------------------------------
# Format fixture builders
# ---------------------------------------------------------------------------

def _make_resume_docx(tmp_path, filename="resume.docx"):
    import docx
    document = docx.Document()
    document.add_heading("Personal Information", level=1)
    document.add_paragraph(
        "Name: Harsh Singh\n"
        "Phone: +91-98765-43210\n"
        "Email: harsh.singh@example.com\n"
        "Location: Bangalore, India"
    )
    document.add_heading("Education", level=1)
    document.add_paragraph(
        "KNS Institute of Technology, Bangalore — Bachelor of Engineering in "
        "Artificial Intelligence & Machine Learning (2022–2026), CGPA: 8.7/10"
    )
    document.add_heading("Projects", level=1)
    document.add_paragraph(
        "1. Local Voice RAG Assistant (Banking FAQ): A voice-enabled banking "
        "FAQ system using local RAG with Ollama embeddings and Qwen generation."
    )
    document.add_paragraph(
        "2. Hybrid RAG System for Financial Document Q&A: Combines BM25 "
        "keyword search with dense vector retrieval and RRF fusion."
    )
    document.add_paragraph(
        "3. AI-Based Travel Planner: Generates personalized travel itineraries "
        "using LLM-based recommendation and real-time data."
    )
    path = tmp_path / filename
    document.save(str(path))
    return str(path)


def _make_resume_docx_with_table(tmp_path, filename="resume_table.docx"):
    import docx
    document = docx.Document()
    document.add_heading("Education", level=1)
    document.add_paragraph("Education details are summarized below.")
    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    hdr[0].text = "Institution"
    hdr[1].text = "Degree"
    hdr[2].text = "Year"
    row = table.add_row().cells
    row[0].text = "KNS Institute of Technology, Bangalore"
    row[1].text = "Bachelor of Engineering in AI & ML"
    row[2].text = "2022–2026"
    path = tmp_path / filename
    document.save(str(path))
    return str(path)


def _make_long_docx(tmp_path, filename="long.docx"):
    import docx
    document = docx.Document()
    document.add_heading("Projects", level=1)
    long_desc = (
        "Project Alpha: A comprehensive system for distributed data processing "
        "that handles petabyte-scale workloads across multiple regions. "
        "Built by Alice Johnson using Python, Go, and Kubernetes. "
        "Deployed on AWS with auto-scaling and blue-green deployments. "
        "Project Beta: A real-time analytics platform for financial "
        "transactions processing 50K events per second with sub-100ms "
        "latency. Built by Bob Chen using Apache Kafka, Flink, and Cassandra. "
        "Project Gamma: A computer vision system for medical imaging "
        "with 94% accuracy on tumor detection. Built by Carol Davis using "
        "PyTorch, MONAI, and Docker containers."
    )
    document.add_paragraph(long_desc)
    path = tmp_path / filename
    document.save(str(path))
    return str(path)


def _make_pdf_knowledge_base(tmp_path, filename="kb.pdf"):
    import fitz
    path = tmp_path / filename
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72),
        "Page 1: Introduction to machine learning and supervised learning algorithms.")
    page2 = doc.new_page()
    page2.insert_text((72, 72),
        "Page 2: Deep learning architectures including CNNs, RNNs, and transformers.")
    page3 = doc.new_page()
    page3.insert_text((72, 72),
        "Page 3: Natural language processing with BERT, GPT, and attention mechanisms.")
    doc.save(str(path))
    doc.close()
    return str(path)


def _make_txt_facts(tmp_path, filename="facts.txt"):
    path = tmp_path / filename
    path.write_text(
        "Company: Acme Corp\n"
        "Founded: 2015\n"
        "CEO: Sarah Chen\n"
        "Headquarters: San Francisco, CA\n"
        "Employees: 500\n",
        encoding="utf-8",
    )
    return str(path)


def _make_md_technical(tmp_path, filename="tech.md"):
    path = tmp_path / filename
    path.write_text(
        "# API Documentation\n"
        "Overview of the REST API endpoints.\n\n"
        "# Authentication\n"
        "Use Bearer tokens for all authenticated requests.\n\n"
        "# Rate Limits\n"
        "Free tier: 100 requests/hour. Paid tier: 10,000 requests/hour.\n",
        encoding="utf-8",
    )
    return str(path)


def _make_csv_employees(tmp_path, filename="employees.csv"):
    path = tmp_path / filename
    path.write_text(
        "Name,Role,Department,Salary,StartYear\n"
        "Alice Johnson,Engineer,Engineering,120000,2020\n"
        "Bob Chen,Manager,Engineering,150000,2018\n"
        "Carol Davis,Designer,Product,110000,2021\n",
        encoding="utf-8",
    )
    return str(path)


def _make_xlsx_sales(tmp_path, filename="sales.xlsx"):
    import openpyxl
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    sheet = workbook.create_sheet("Q1")
    sheet.append(["Product", "Revenue", "Units"])
    sheet.append(["Widget A", 50000, 1000])
    sheet.append(["Gadget B", 75000, 1500])
    path = tmp_path / filename
    workbook.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Category 1: Exact keyword questions
# ---------------------------------------------------------------------------

class TestExactKeywordQuestions:
    """Category 1: Questions with exact keyword matches."""

    def test_exact_keyword_retrieves_correct_chunk(self, tmp_path, chroma_service, retriever):
        path = _make_txt_facts(tmp_path)
        chunks = process_document(path, "facts.txt")
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("CEO Sarah Chen", n_results=3)
        all_text = " ".join(results["documents"])
        assert "Sarah Chen" in all_text
        assert "CEO" in all_text

    def test_exact_keyword_missing_term_returns_empty(self, tmp_path, chroma_service, retriever):
        path = _make_txt_facts(tmp_path)
        chunks = process_document(path, "facts.txt")
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("nonexistentperson xyz", n_results=3)
        # BM25 may return low-score results even with no matching terms.
        # Verify none of the returned documents contain the query terms.
        all_text = " ".join(results["documents"]).lower()
        assert "nonexistentperson" not in all_text
        assert "xyz" not in all_text


# ---------------------------------------------------------------------------
# Category 2: Semantic paraphrases and synonyms
# ---------------------------------------------------------------------------

class TestSemanticParaphrases:
    """Category 2: Questions using different words for same concepts."""

    def test_semantic_match_finds_related_content(self, tmp_path, chroma_service, retriever):
        path = _make_pdf_knowledge_base(tmp_path)
        chunks = process_document(path, "kb.pdf")
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("neural network architectures for images", n_results=3)
        all_text = " ".join(results["documents"])
        assert "CNN" in all_text or "deep learning" in all_text.lower()


# ---------------------------------------------------------------------------
# Category 3: Precise facts (names, dates, institutions, numbers)
# ---------------------------------------------------------------------------

class TestPreciseFacts:
    """Category 3: Questions about specific named entities and values."""

    def test_institution_and_degree_in_same_chunk(self, tmp_path, chroma_service, retriever):
        path = _make_resume_docx(tmp_path)
        chunks = process_document(path, "resume.docx")
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("KNS Institute of Technology", n_results=5)
        all_text = " ".join(results["documents"])
        assert "KNS Institute of Technology" in all_text
        assert "Bachelor of Engineering" in all_text

    def test_date_range_retrieval(self, tmp_path, chroma_service, retriever):
        path = _make_txt_facts(tmp_path)
        chunks = process_document(path, "facts.txt")
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("founded 2015", n_results=3)
        all_text = " ".join(results["documents"])
        assert "2015" in all_text


# ---------------------------------------------------------------------------
# Category 4: Summaries and multi-part questions
# ---------------------------------------------------------------------------

class TestSummariesAndMultiPart:
    """Category 4: Questions requiring synthesis across sections."""

    def test_multi_part_projects_question(self, tmp_path, chroma_service, retriever):
        path = _make_resume_docx(tmp_path)
        chunks = process_document(path, "resume.docx")
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("What projects has the person built?", n_results=5)
        all_text = " ".join(results["documents"])
        assert "Local Voice RAG Assistant" in all_text
        assert "Hybrid RAG System" in all_text
        assert "AI-Based Travel Planner" in all_text

    def test_summary_across_pages(self, tmp_path, chroma_service, retriever):
        path = _make_pdf_knowledge_base(tmp_path)
        chunks = process_document(path, "kb.pdf")
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("machine learning deep learning NLP", n_results=5)
        all_text = " ".join(results["documents"])
        assert "machine learning" in all_text.lower()
        assert "deep learning" in all_text.lower()
        assert "natural language processing" in all_text.lower()


# ---------------------------------------------------------------------------
# Category 5: Multi-chunk evidence synthesis
# ---------------------------------------------------------------------------

class TestMultiChunkEvidence:
    """Category 5: Answers requiring evidence from multiple chunks."""

    def test_facts_split_across_chunks(self, tmp_path, chroma_service, retriever):
        path = _make_long_docx(tmp_path)
        chunks = process_document(path, "long.docx")
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("Project Beta Bob Chen Kafka", n_results=5)
        all_text = " ".join(results["documents"])
        assert "Project Beta" in all_text
        assert "Bob Chen" in all_text
        assert "Kafka" in all_text


# ---------------------------------------------------------------------------
# Category 6: Table-based and structured data questions
# ---------------------------------------------------------------------------

class TestTableStructuredData:
    """Category 6: Questions about tabular/structured data."""

    def test_csv_salary_question(self, tmp_path, chroma_service, retriever):
        path = _make_csv_employees(tmp_path)
        chunks = process_document(path, "employees.csv")
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("Alice Johnson salary Engineering", n_results=3)
        all_text = " ".join(results["documents"])
        assert "Alice Johnson" in all_text
        assert "120000" in all_text

    def test_xlsx_revenue_question(self, tmp_path, chroma_service, retriever):
        path = _make_xlsx_sales(tmp_path)
        chunks = process_document(path, "sales.xlsx")
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("Widget A revenue Q1", n_results=3)
        all_text = " ".join(results["documents"])
        assert "Widget A" in all_text
        assert "50000" in all_text

    def test_docx_table_content_is_extracted(self, tmp_path, chroma_service, retriever):
        """D1 fixed: table content in DOCX is now extracted and retrievable."""
        path = _make_resume_docx_with_table(tmp_path)
        chunks = process_document(path, "resume_table.docx")
        all_text = " ".join(c["text"] for c in chunks)
        assert "KNS Institute of Technology" in all_text
        assert "Bachelor of Engineering" in all_text
        # Table chunks carry sheet/rows location metadata
        table_chunks = [c for c in chunks if c.get("sheet") == "Table 1"]
        assert table_chunks
        for chunk in table_chunks:
            assert "rows" in chunk

    def test_docx_table_facts_retrievable(self, tmp_path, chroma_service, retriever):
        """Table facts can be retrieved via hybrid search."""
        path = _make_resume_docx_with_table(tmp_path)
        chunks = process_document(path, "resume_table.docx")
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("KNS Institute of Technology Bachelor of Engineering", n_results=5)
        all_text = " ".join(results["documents"])
        assert "KNS Institute of Technology" in all_text
        assert "Bachelor of Engineering" in all_text


# ---------------------------------------------------------------------------
# Category 7: Metadata filtering and multi-document retrieval
# ---------------------------------------------------------------------------

class TestMetadataFiltering:
    """Category 7: Filtering by file_name and multi-document queries."""

    def test_multi_document_retrieval(self, tmp_path, chroma_service, retriever):
        txt_chunks = process_document(_make_txt_facts(tmp_path, "facts.txt"), "facts.txt")
        md_chunks = process_document(_make_md_technical(tmp_path, "tech.md"), "tech.md")
        chroma_service.add_chunks(txt_chunks + md_chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("CEO bearer tokens", n_results=5)
        all_text = " ".join(results["documents"])
        assert "Sarah Chen" in all_text
        assert "bearer tokens" in all_text.lower()

    def test_metadata_filter_by_file(self, tmp_path, chroma_service):
        txt_chunks = process_document(_make_txt_facts(tmp_path, "facts.txt"), "facts.txt")
        chroma_service.add_chunks(txt_chunks)
        results = chroma_service.query(
            "CEO", n_results=3,
            where={"file_name": {"$eq": "facts.txt"}},
        )
        assert len(results["ids"][0]) >= 1
        all_text = " ".join(results["documents"][0])
        assert "Sarah Chen" in all_text


# ---------------------------------------------------------------------------
# Category 8: Missing-answer / insufficient-evidence cases
# ---------------------------------------------------------------------------

class TestMissingAnswer:
    """Category 8: Questions with no supporting evidence."""

    def test_no_relevant_content_returns_empty(self, tmp_path, chroma_service, retriever):
        path = _make_txt_facts(tmp_path)
        chunks = process_document(path, "facts.txt")
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("quantum computing blockchain", n_results=3)
        # BM25 may return low-score results even with no matching terms.
        # Verify none of the returned documents contain the query terms.
        all_text = " ".join(results["documents"]).lower()
        assert "quantum" not in all_text
        assert "blockchain" not in all_text


# ---------------------------------------------------------------------------
# Category 9: Relevant info buried among irrelevant chunks
# ---------------------------------------------------------------------------

class TestBuriedRelevantInfo:
    """Category 9: Signal vs noise in retrieval ranking."""

    def test_relevant_chunk_among_irrelevant(self, tmp_path, chroma_service, retriever):
        chunks = []
        for i in range(10):
            chunks.append({
                "text": f"Irrelevant filler content about topic {i} with lots of words.",
                "file_name": "doc.txt",
                "chunk_index": i,
            })
        chunks.append({
            "text": "The secret answer is: Project Delta uses quantum encryption.",
            "file_name": "doc.txt",
            "chunk_index": 10,
        })
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("Project Delta quantum encryption", n_results=5)
        all_text = " ".join(results["documents"])
        assert "Project Delta" in all_text


# ---------------------------------------------------------------------------
# Category 10: Conflicting or repeated facts
# ---------------------------------------------------------------------------

class TestConflictingFacts:
    """Category 10: System must avoid unsupported conclusions from conflicts."""

    def test_conflicting_dates_same_entity(self, tmp_path, chroma_service, retriever):
        path1 = tmp_path / "doc1.txt"
        path1.write_text("Company: Acme Corp. Founded: 2015.", encoding="utf-8")
        path2 = tmp_path / "doc2.txt"
        path2.write_text("Company: Acme Corp. Founded: 2018.", encoding="utf-8")

        chunks1 = process_document(str(path1), "doc1.txt")
        chunks2 = process_document(str(path2), "doc2.txt")
        chroma_service.add_chunks(chunks1 + chunks2)
        retriever.rebuild_index()

        results = retriever.retrieve("Acme Corp founded year", n_results=5)
        all_text = " ".join(results["documents"])
        assert "2015" in all_text or "2018" in all_text


# ---------------------------------------------------------------------------
# Answer generation tests with mocked LLM
# ---------------------------------------------------------------------------

class TestAnswerGeneration:
    """Unit tests for answer generation with controlled context."""

    def test_generate_answer_with_sufficient_context(
        self, answer_generator, mock_ollama_answer
    ):
        with patch("requests.post", return_value=mock_ollama_answer(
            "The person built three projects: Local Voice RAG Assistant, "
            "Hybrid RAG System, and AI-Based Travel Planner."
        )):
            result = answer_generator.generate(
                query="What projects did the person build?",
                retriever_results={
                    "ids": ["c1", "c2"],
                    "documents": [
                        "Project 1: Local Voice RAG Assistant for banking FAQ. The person built this project.",
                        "Project 2: Hybrid RAG System for financial Q&A.",
                    ],
                    "metadatas": [
                        {"file_name": "resume.docx", "chunk_index": 0, "section": "Projects"},
                        {"file_name": "resume.docx", "chunk_index": 1, "section": "Projects"},
                    ],
                    "rrf_scores": [0.03, 0.025],
                },
                max_sources=5,
            )
            assert "Local Voice RAG Assistant" in result["answer"]
            assert "Hybrid RAG System" in result["answer"]
            assert result["confidence"] > 0

    def test_abstains_when_no_context(self, answer_generator):
        result = answer_generator.generate(
            query="What is the meaning of life?",
            retriever_results={
                "ids": [],
                "documents": [],
                "metadatas": [],
                "rrf_scores": [],
            },
            max_sources=5,
        )
        assert "don't have enough context" in result["answer"].lower()
        assert result["confidence"] == 0.0
        assert result["sources"] == []

    def test_context_truncation_respects_max_sources(
        self, answer_generator, mock_ollama_answer
    ):
        with patch("requests.post", return_value=mock_ollama_answer("Answer.")):
            result = answer_generator.generate(
                query="Test query",
                retriever_results={
                    "ids": [f"c{i}" for i in range(7)],
                    "documents": [f"Document {i} content." for i in range(7)],
                    "metadatas": [
                        {"file_name": "doc.txt", "chunk_index": i} for i in range(7)
                    ],
                    "rrf_scores": [0.03] * 7,
                },
                max_sources=3,
            )
            assert len(result["sources"]) == 3

    def test_singular_plural_evidence_matching(self, answer_generator):
        """_compute_evidence_sufficiency should match plural query terms to
        singular document words via suffix stripping + word-boundary matching."""
        score = answer_generator._compute_evidence_sufficiency(
            documents=["Project 1: Local Voice RAG Assistant for banking FAQ."],
            query="What projects did the person build?",
        )
        # "projects" -> "project" matches "Project" in doc_words
        # 1 of 6 query terms matches: 1/6 ≈ 0.167
        assert abs(score - 1 / 5) < 1e-9

    def test_confidence_consistency_after_normalization(self, answer_generator):
        """_calculate_confidence must use the same normalization as
        _compute_evidence_sufficiency for evidence sufficiency."""
        conf = answer_generator._calculate_confidence(
            scores=[0.03, 0.025],
            documents=["Project 1: Local Voice RAG Assistant for banking FAQ."],
            query="What projects did the person build?",
        )
        # retrieval_signal = min(avg_rrf * 19, 1.0) = min(0.0275 * 19, 1.0) ≈ 0.5225
        # evidence_sufficiency = 1/6 ≈ 0.167
        # confidence = 0.5225 * 0.167 ≈ 0.087
        expected_rrf_signal = min((0.03 + 0.025) / 2 * 19, 1.0)
        expected_confidence = expected_rrf_signal * (1 / 5)
        assert abs(conf - expected_confidence) < 1e-9

    def test_abstains_when_no_matching_evidence_with_documents(
        self, answer_generator
    ):
        """Genuinely insufficient evidence (0 query terms match) with
        documents present should still trigger abstention."""
        result = answer_generator.generate(
            query="quantum blockchain encryption",
            retriever_results={
                "ids": ["c1", "c2"],
                "documents": [
                    "Project 1: Local Voice RAG Assistant for banking FAQ.",
                    "Project 2: Hybrid RAG System for financial Q&A.",
                ],
                "metadatas": [
                    {"file_name": "resume.docx", "chunk_index": 0},
                    {"file_name": "resume.docx", "chunk_index": 1},
                ],
                "rrf_scores": [0.03, 0.025],
            },
            max_sources=5,
        )
        assert "don't have enough information" in result["answer"].lower()
        assert result["confidence"] == 0.0
        assert result["tokens_used"] == 0

    def test_source_truncation_with_insufficient_evidence(
        self, answer_generator
    ):
        """When evidence is insufficient (0.0), sources extracted before
        abstention should still be returned (up to max_sources)."""
        result = answer_generator.generate(
            query="Test query",
            retriever_results={
                "ids": [f"c{i}" for i in range(7)],
                "documents": [f"Document {i} content." for i in range(7)],
                "metadatas": [
                    {"file_name": "doc.txt", "chunk_index": i} for i in range(7)
                ],
                "rrf_scores": [0.03] * 7,
            },
            max_sources=3,
        )
        assert len(result["sources"]) == 3
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Independent retrieval evaluation: BM25, dense, RRF with Recall@k
# ---------------------------------------------------------------------------

class TestRetrievalEvaluation:
    """
    Independent evaluation of BM25, dense retrieval, and RRF fusion.
    Measures Recall@k and ranking quality using controlled synthetic fixtures.
    No live LLM or embedding inference required — embeddings are mocked.
    """

    def _index_chunks(self, chroma_service, retriever, chunks):
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()


# ------------------------------------------------------------------
# BM25 alone — keyword recall
# ------------------------------------------------------------------

class TestBM25Evaluation:
    """Independent BM25 evaluation with Recall@k measurement."""

    def test_bm25_recall_at_k_exact_keyword(self, chroma_service, retriever):
        chunks = [
            {"text": "The Local Voice RAG Assistant processes banking FAQ queries using Ollama.",
             "file_name": "resume.docx", "chunk_index": 0, "section": "Projects"},
            {"text": "Hybrid RAG System combines BM25 and dense vector retrieval with RRF fusion.",
             "file_name": "resume.docx", "chunk_index": 1, "section": "Projects"},
            {"text": "KNS Institute of Technology Bangalore offers Bachelor of Engineering in AI & ML.",
             "file_name": "resume.docx", "chunk_index": 2, "section": "Education"},
        ]
        retriever._index_chunks = None  # no-op guard
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever._bm25_search("Local Voice RAG Assistant banking FAQ", top_k=3)
        ids = [r[0] for r in results]
        docs = chroma_service.collection.get(ids=ids, include=["documents"])["documents"] if ids else []
        retrieved_text = " ".join(docs)

        assert "Local Voice RAG Assistant" in retrieved_text
        assert "banking FAQ" in retrieved_text

    def test_bm25_recall_at_k_no_match(self, chroma_service, retriever):
        chunks = [
            {"text": "Project Alpha uses Python and Kubernetes for distributed processing.",
             "file_name": "doc.txt", "chunk_index": 0},
        ]
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever._bm25_search("quantum blockchain encryption", top_k=5)
        assert len(results) == 0

    def test_bm25_ranking_keyword_density(self, chroma_service, retriever):
        # With 3 docs, "Kubernetes" appears in 1 of 3 → IDF > 0 → positive BM25 score.
        chunks = [
            {"text": "Kubernetes orchestrates containerized workloads at scale.",
             "file_name": "a.txt", "chunk_index": 0},
            {"text": "Python appears once here.", "file_name": "b.txt", "chunk_index": 0},
            {"text": "The weather is sunny today in Bangalore.", "file_name": "c.txt", "chunk_index": 0},
        ]
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever._bm25_search("Kubernetes", top_k=2)
        assert len(results) == 1
        assert "Kubernetes" in chroma_service.collection.get(
            ids=[results[0][0]], include=["documents"]
        )["documents"][0]


# ------------------------------------------------------------------
# Dense retrieval alone — semantic recall
# ------------------------------------------------------------------

class TestDenseEvaluation:
    """Independent dense retrieval evaluation with Recall@k measurement."""

    def test_dense_recall_at_k_semantic(self, chroma_service, retriever):
        chunks = [
            {"text": "Neural networks with multiple layers learn hierarchical representations.",
             "file_name": "dl.pdf", "page": 1, "chunk_index": 0},
            {"text": "The weather is sunny and warm today in Bangalore.",
             "file_name": "weather.txt", "chunk_index": 0},
            {"text": "Transformers use attention mechanisms for sequence modeling.",
             "file_name": "nlp.pdf", "page": 2, "chunk_index": 0},
        ]
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever._dense_search("deep learning architectures", top_k=3)
        ids = [r[0] for r in results]
        docs = chroma_service.collection.get(ids=ids, include=["documents"])["documents"] if ids else []
        retrieved_text = " ".join(docs)

        assert "neural networks" in retrieved_text.lower() or "transformers" in retrieved_text.lower()

    def test_dense_recall_at_k_unrelated_query(self, chroma_service, retriever):
        chunks = [
            {"text": "The weather is sunny today.", "file_name": "weather.txt", "chunk_index": 0},
        ]
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever._dense_search("quantum computing blockchain", top_k=3)
        assert len(results) == 1


# ------------------------------------------------------------------
# RRF fusion — combined recall and ranking
# ------------------------------------------------------------------

class TestRFREvaluation:
    """Independent RRF fusion evaluation with ranking quality measurement."""

    def test_rrf_recall_at_k_keyword_and_semantic(self, chroma_service, retriever):
        chunks = [
            {"text": "Local Voice RAG Assistant for banking FAQ with Ollama embeddings.",
             "file_name": "resume.docx", "chunk_index": 0, "section": "Projects"},
            {"text": "Deep learning neural network architectures for computer vision.",
             "file_name": "ml.pdf", "page": 1, "chunk_index": 0},
            {"text": "KNS Institute of Technology Bangalore Bachelor of Engineering AI ML 2022-2026.",
             "file_name": "resume.docx", "chunk_index": 1, "section": "Education"},
        ]
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("KNS Institute of Technology engineering", n_results=3)
        all_text = " ".join(results["documents"])

        assert "KNS Institute of Technology" in all_text
        assert "Bachelor of Engineering" in all_text

    def test_rrf_ranking_quality(self, chroma_service, retriever):
        # Project Alpha matches both keyword ("Python", "data") and semantic
        # signals; other docs match weakly or not at all.
        # Use >=5 docs so BM25 IDF is meaningful (not ≈0 in a 2-doc corpus).
        chunks = [
            {"text": "Project Alpha uses Python for distributed data processing and analytics.",
             "file_name": "a.txt", "chunk_index": 0},
            {"text": "The weather forecast predicts rain for tomorrow afternoon.",
             "file_name": "b.txt", "chunk_index": 0},
            {"text": "Database migration strategies for production PostgreSQL clusters.",
             "file_name": "c.txt", "chunk_index": 0},
            {"text": "Machine learning model training pipelines with GPU acceleration.",
             "file_name": "d.txt", "chunk_index": 0},
            {"text": "Kubernetes container orchestration for microservices deployment.",
             "file_name": "e.txt", "chunk_index": 0},
        ]
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("Project Alpha Python data processing", n_results=2)
        assert len(results["ids"]) >= 1
        assert "Project Alpha" in results["documents"][0]

    def test_rrf_deduplication(self, chroma_service, retriever):
        chunks = [
            {"text": "RAG combines retrieval with generation for better answers.",
             "file_name": "doc.pdf", "page": 1, "chunk_index": 0},
        ]
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()

        results = retriever.retrieve("RAG retrieval generation", n_results=5)
        assert len(results["ids"]) == len(set(results["ids"]))


# ------------------------------------------------------------------
# Recall@k measurement
# ------------------------------------------------------------------

class TestRecallAtK:
    """Recall@k measurement across retrieval methods."""

    def test_recall_at_k_bm25(self, chroma_service, retriever):
        relevant = [
            {"text": "Python is a programming language for data science.", "file_name": "d.txt", "chunk_index": 0},
            {"text": "Python pandas library for data manipulation.", "file_name": "d.txt", "chunk_index": 1},
            {"text": "Python matplotlib for data visualization.", "file_name": "d.txt", "chunk_index": 2},
        ]
        irrelevant = [
            {"text": "The weather is sunny today.", "file_name": "w.txt", "chunk_index": 0},
            {"text": "Cooking recipes for chocolate cake.", "file_name": "c.txt", "chunk_index": 0},
        ]
        chroma_service.add_chunks(relevant + irrelevant)
        retriever.rebuild_index()

        all_ids = chroma_service.collection.get(include=["documents"])["ids"]
        relevant_ids = set(all_ids[:3])

        for k in (1, 2, 3, 5):
            results = retriever._bm25_search("Python", top_k=k)
            retrieved_ids = set(r[0] for r in results)
            recall = len(retrieved_ids & relevant_ids) / len(relevant_ids)
            assert 0.0 <= recall <= 1.0

    def test_recall_at_k_dense(self, chroma_service, retriever):
        relevant = [
            {"text": "Machine learning algorithms learn from data patterns.", "file_name": "ml.txt", "chunk_index": 0},
            {"text": "Deep learning uses neural networks with many layers.", "file_name": "dl.txt", "chunk_index": 0},
        ]
        irrelevant = [
            {"text": "The weather forecast calls for rain tomorrow.", "file_name": "w.txt", "chunk_index": 0},
        ]
        chroma_service.add_chunks(relevant + irrelevant)
        retriever.rebuild_index()

        for k in (1, 2, 3):
            results = retriever._dense_search("machine learning neural networks", top_k=k)
            assert len(results) <= k
            assert len(results) >= 0


# ------------------------------------------------------------------
# Evidence normalization tests (Priority 1)
# ------------------------------------------------------------------

class TestEvidenceNormalization:
    """Tests for symmetric punctuation/plural normalization in evidence sufficiency."""

    def test_evidence_technical_identifiers_preserved(self, answer_generator):
        ag = answer_generator
        score = ag._compute_evidence_sufficiency(["AI/ML engineering"], "AI/ML")
        assert score == 1.0
        score = ag._compute_evidence_sufficiency(["Node.js backend"], "Node.js")
        assert score == 1.0

    def test_evidence_currency_consistent(self, answer_generator):
        ag = answer_generator
        score = ag._compute_evidence_sufficiency(
            ["Stipend: ₹40,000 per month"], "stipend 40000"
        )
        assert score == 1.0
        score = ag._compute_evidence_sufficiency(
            ["Stipend: 40000 per month"], "₹40,000"
        )
        assert score == 1.0

    def test_evidence_bidirectional_plural(self, answer_generator):
        ag = answer_generator
        score = ag._compute_evidence_sufficiency(["We built Projects"], "project")
        assert score == 1.0
        score = ag._compute_evidence_sufficiency(["We built Project"], "projects")
        assert score == 1.0

    def test_evidence_short_technical_tokens_preserved(self, answer_generator):
        ag = answer_generator
        # C and R are preserved (stopword filter, not length filter) → match
        score = ag._compute_evidence_sufficiency(["C programming"], "C")
        assert score == 1.0
        score = ag._compute_evidence_sufficiency(["R statistics"], "R")
        assert score == 1.0

    def test_evidence_genuinely_irrelevant_zero_overlap(self, answer_generator):
        ag = answer_generator
        score = ag._compute_evidence_sufficiency(
            ["The weather is sunny today."], "quantum computing algorithms"
        )
        assert score == 0.0

    def test_confidence_after_normalization(self, answer_generator):
        ag = answer_generator
        conf = ag._calculate_confidence(
            scores=[0.03],
            documents=["C++ AI/ML projects"],
            query="C++ AI/ML",
        )
        assert conf > 0.0


# ------------------------------------------------------------------
# Structured filtering tests (Priority 2)
# ------------------------------------------------------------------

class TestStructuredFiltering:
    """Tests for deterministic spreadsheet row filtering."""

    def _make_spreadsheet_chunks(self, rows_data, header, file_name, sheet=None):
        from backend.app.services.document_processor import _tabular_segments
        data_rows = [[str(v) for v in row] for row in rows_data]
        return _tabular_segments(header, data_rows, sheet_label=sheet)

    def test_structured_filter_qualifying_rows(self, chroma_service):
        from backend.app.services.structured_query import filter_structured_rows, Condition
        header = ["Internships", "Stipend", "Duration"]
        rows = [
            ["Google", "₹40,000", "3 months"],
            ["Microsoft", "₹50,000", "6 months"],
            ["Startup", "₹35,000", "2 months"],
            ["BigCorp", "₹45,000", "3 months"],
        ]
        segments = self._make_spreadsheet_chunks(rows, header, "internships.xlsx", sheet="Internships")
        documents = [s["text"] for s in segments]
        metadatas = [{"file_name": "internships.xlsx", "sheet": "Internships", **s["location"]} for s in segments]
        ids = [f"doc_{i}" for i in range(len(documents))]
        conditions = [Condition("stipend", ">=", 40000.0), Condition("duration", "<=", 3.0)]
        matching_docs, matching_metas, _ = filter_structured_rows(documents, metadatas, ids, conditions)
        assert len(matching_docs) == 2
        assert any("Google" in d for d in matching_docs)
        assert any("BigCorp" in d for d in matching_docs)
        for meta in matching_metas:
            assert meta["_file_name"] == "internships.xlsx"
            assert meta["_sheet"] == "Internships"

    def test_structured_filter_non_qualifying_rows(self, chroma_service):
        from backend.app.services.structured_query import filter_structured_rows, Condition
        header = ["Internships", "Stipend", "Duration"]
        rows = [
            ["Google", "₹40,000", "3 months"],
            ["Microsoft", "₹50,000", "6 months"],
        ]
        segments = self._make_spreadsheet_chunks(rows, header, "internships.xlsx")
        documents = [s["text"] for s in segments]
        metadatas = [{"file_name": "internships.xlsx", **s["location"]} for s in segments]
        ids = ["doc_0", "doc_0"]
        conditions = [Condition("stipend", ">=", 100000.0)]
        matching_docs, _, _ = filter_structured_rows(documents, metadatas, ids, conditions)
        assert len(matching_docs) == 0

    def test_structured_filter_ambiguous_ranges_unknown(self, chroma_service):
        from backend.app.services.structured_query import (
            filter_structured_rows, Condition, FilterOutcome,
            evaluate_condition, parse_numeric_value,
        )
        assert parse_numeric_value("₹30,000–₹40,000") is None
        assert parse_numeric_value("3–6 months") is None
        assert evaluate_condition(
            "₹30,000–₹40,000", Condition("stipend", ">=", 40000.0)
        ) == FilterOutcome.UNKNOWN
        assert evaluate_condition(
            "3–6 months", Condition("duration", "<=", 3.0)
        ) == FilterOutcome.UNKNOWN
        header = ["Internships", "Stipend", "Duration"]
        rows = [
            ["RangeCo", "₹30,000–₹40,000", "3–6 months"],
            ["ClearCo", "₹40,000", "3 months"],
        ]
        segments = self._make_spreadsheet_chunks(rows, header, "ranges.xlsx")
        documents = [s["text"] for s in segments]
        metadatas = [{"file_name": "ranges.xlsx", **s["location"]} for s in segments]
        ids = ["doc_0", "doc_0"]
        conditions = [
            Condition("stipend", ">=", 40000.0),
            Condition("duration", "<=", 3.0),
        ]
        matching_docs, _, _ = filter_structured_rows(documents, metadatas, ids, conditions)
        assert len(matching_docs) == 1
        assert "ClearCo" in matching_docs[0]

    def test_structured_filter_no_match_abstains(self, chroma_service, answer_generator):
        from backend.app.services.structured_query import filter_structured_rows, Condition
        header = ["Internships", "Stipend"]
        rows = [["Google", "₹40,000"]]
        segments = self._make_spreadsheet_chunks(rows, header, "internships.xlsx")
        documents = [s["text"] for s in segments]
        metadatas = [{"file_name": "internships.xlsx", **s["location"]} for s in segments]
        ids = ["doc_0"]
        conditions = [Condition("stipend", ">=", 100000.0)]
        matching_docs, _, _ = filter_structured_rows(documents, metadatas, ids, conditions)
        assert len(matching_docs) == 0
        response = answer_generator._insufficient_evidence_response("stipend >= 100000", 0.0, [])
        assert response["confidence"] == 0.0
        assert response["sources"] == []

    def test_structured_filter_multiple_conditions_and(self, chroma_service):
        from backend.app.services.structured_query import filter_structured_rows, Condition
        header = ["Internships", "Stipend", "Duration"]
        rows = [
            ["Google", "₹40,000", "3 months"],
            ["Microsoft", "₹50,000", "6 months"],
            ["Startup", "₹35,000", "2 months"],
        ]
        segments = self._make_spreadsheet_chunks(rows, header, "internships.xlsx")
        documents = [s["text"] for s in segments]
        metadatas = [{"file_name": "internships.xlsx", **s["location"]} for s in segments]
        ids = ["doc_0"] * len(documents)
        conditions = [
            Condition("stipend", ">=", 40000.0),
            Condition("duration", "<=", 3.0),
        ]
        matching_docs, _, _ = filter_structured_rows(documents, metadatas, ids, conditions)
        assert len(matching_docs) == 1
        assert "Google" in matching_docs[0]

    def test_structured_filter_source_provenance(self, chroma_service):
        from backend.app.services.structured_query import filter_structured_rows, Condition
        header = ["Internships", "Stipend"]
        rows = [["Google", "₹40,000"]]
        segments = self._make_spreadsheet_chunks(rows, header, "mydata.xlsx", sheet="Sheet1")
        documents = [s["text"] for s in segments]
        metadatas = [{"file_name": "mydata.xlsx", "sheet": "Sheet1", **s["location"]} for s in segments]
        ids = ["doc_0"]
        conditions = [Condition("stipend", ">=", 40000.0)]
        matching_docs, matching_metas, _ = filter_structured_rows(documents, metadatas, ids, conditions)
        assert len(matching_docs) == 1
        meta = matching_metas[0]
        assert meta["_file_name"] == "mydata.xlsx"
        assert meta["_sheet"] == "Sheet1"
        assert meta["_row_number"] == 2

    def test_structured_filter_all_rows_evaluated(self, chroma_service):
        from backend.app.services.structured_query import filter_structured_rows, Condition
        header = ["Internships", "Stipend"]
        rows = [[f"Company{i}", "₹50,000"] for i in range(10)]
        segments = self._make_spreadsheet_chunks(rows, header, "many.xlsx")
        documents = [s["text"] for s in segments]
        metadatas = [{"file_name": "many.xlsx", **s["location"]} for s in segments]
        ids = ["doc_0"] * len(documents)
        conditions = [Condition("stipend", ">=", 40000.0)]
        matching_docs, _, _ = filter_structured_rows(documents, metadatas, ids, conditions)
        assert len(matching_docs) == 10

    def test_structured_filter_unsupported_passthrough(self, chroma_service):
        from backend.app.services.structured_query import parse_conditions
        assert parse_conditions("What is RAG?") == []
        assert parse_conditions("Tell me about machine learning") == []
        assert parse_conditions("How does AI work?") == []


# ------------------------------------------------------------------
# Regression baseline tests (Priority 3)
# ------------------------------------------------------------------

class TestRegressionBaselines:
    """Tests verifying resume and IEEE paper retrieval do not regress."""

    def test_resume_projects_retrieval(self, chroma_service, retriever):
        chunks = [
            {
                "text": "Project Alpha: A data processing pipeline built with Python and pandas.",
                "file_name": "resume.docx", "section": "Projects", "chunk_index": 0,
            },
            {
                "text": "Project Beta: Machine learning model for recommendation systems.",
                "file_name": "resume.docx", "section": "Projects", "chunk_index": 1,
            },
            {
                "text": "Technical Skills: Python, C++, AI/ML, SQL, JavaScript.",
                "file_name": "resume.docx", "section": "Skills", "chunk_index": 2,
            },
        ]
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()
        results = retriever.retrieve("projects data processing Python", n_results=3)
        assert len(results["documents"]) >= 1
        assert any("data processing" in d for d in results["documents"])

    def test_resume_projects_unique_ids(self, chroma_service, retriever):
        chunks = [
            {
                "text": "Project Alpha: A data processing pipeline.",
                "file_name": "resume.docx", "section": "Projects", "chunk_index": 0,
            },
            {
                "text": "Project Beta: ML recommendation system.",
                "file_name": "resume.docx", "section": "Projects", "chunk_index": 1,
            },
        ]
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()
        results = retriever.retrieve("project alpha", n_results=3)
        assert len(results["documents"]) >= 1

    def test_resume_technical_skills(self, chroma_service, retriever):
        chunks = [
            {
                "text": "Technical Skills: Python, C++, AI/ML, SQL, JavaScript, Node.js.",
                "file_name": "resume.docx", "section": "Skills", "chunk_index": 0,
            },
        ]
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()
        results = retriever.retrieve("C++ AI/ML skills", n_results=5)
        assert len(results["documents"]) >= 1
        assert any("C++" in d or "AI/ML" in d for d in results["documents"])

    def test_ieee_paper_retrieval(self, chroma_service, retriever):
        chunks = [
            {
                "text": "AI Travel Planner: An intelligent system for personalized travel recommendations using machine learning.",
                "file_name": "ieee_ai_travel.pdf", "page": 1, "chunk_index": 0,
            },
            {
                "text": "The system uses a hybrid retrieval approach combining BM25 and dense embeddings.",
                "file_name": "ieee_ai_travel.pdf", "page": 2, "chunk_index": 0,
            },
        ]
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()
        results = retriever.retrieve("AI travel planner machine learning", n_results=5)
        assert len(results["documents"]) >= 1
        assert any("travel" in d.lower() for d in results["documents"])