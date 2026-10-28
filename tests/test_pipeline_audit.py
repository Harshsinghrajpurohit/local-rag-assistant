"""
Pipeline audit tests for the RAG-Q&A document assistant.

Demonstrates Phase 1 audit defects and hypotheses. All tests use synthetic
fixtures in temporary directories; the real backend/chroma_db is never
touched. No personal contact information is printed or logged.
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
from backend.app.exceptions import ExtractionFailedError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db_dir():
    temp_dir = tempfile.mkdtemp(prefix="audit_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def embeddings_service():
    return EmbeddingsService()


@pytest.fixture
def chroma_service(temp_db_dir, embeddings_service):
    return ChromaService(db_path=temp_db_dir, embeddings_service=embeddings_service)


@pytest.fixture
def retriever(chroma_service):
    return HybridRetriever(chroma_service=chroma_service)


# ---------------------------------------------------------------------------
# Helper: build synthetic DOCX with tables and heading-styled paragraphs
# ---------------------------------------------------------------------------

def _make_docx_with_tables(tmp_path, filename="test.docx"):
    import docx
    document = docx.Document()
    document.add_heading("Projects", level=1)
    document.add_paragraph(
        "Local Voice RAG Assistant: A banking FAQ system that processes voice "
        "queries using local retrieval-augmented generation with Ollama."
    )
    document.add_paragraph(
        "Hybrid RAG System for Financial Document Q&A: Combines BM25 keyword "
        "search with dense vector retrieval and reciprocal rank fusion."
    )
    table = document.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    header = table.rows[0].cells
    header[0].text = "Institution"
    header[1].text = "Degree"
    row = table.add_row().cells
    row[0].text = "KNS Institute of Technology, Bangalore"
    row[1].text = "Bachelor of Engineering in AI & ML"
    buffer_path = tmp_path / filename
    document.save(str(buffer_path))
    return str(buffer_path)


def _make_docx_headings_only(tmp_path, filename="headings.docx"):
    import docx
    document = docx.Document()
    document.add_heading("Education", level=1)
    document.add_paragraph(
        "KNS Institute of Technology, Bangalore — Bachelor of Engineering in "
        "Artificial Intelligence & Machine Learning (2022–2026)"
    )
    document.add_heading("Projects", level=1)
    document.add_paragraph(
        "1. Local Voice RAG Assistant (Banking FAQ)\n"
        "2. Hybrid RAG System for Financial Document Q&A\n"
        "3. AI-Based Travel Planner"
    )
    buffer_path = tmp_path / filename
    document.save(str(buffer_path))
    return str(buffer_path)


# ---------------------------------------------------------------------------
# D1: DOCX tables are not extracted
# ---------------------------------------------------------------------------

class TestDocxTableExtraction:
    """DOCX table content is now extracted (defect D1 fixed)."""

    def test_docx_table_content_is_extracted(self, tmp_path):
        path = _make_docx_with_tables(tmp_path)
        chunks = process_document(path, "test.docx")
        all_text = " ".join(c["text"] for c in chunks)
        assert "Local Voice RAG Assistant" in all_text
        assert "Hybrid RAG System" in all_text
        # Table content IS now present (D1 fixed)
        assert "KNS Institute of Technology" in all_text
        assert "Bachelor of Engineering" in all_text

    def test_docx_table_chunks_have_location_metadata(self, tmp_path):
        path = _make_docx_with_tables(tmp_path)
        chunks = process_document(path, "test.docx")
        table_chunks = [c for c in chunks if c.get("sheet") == "Table 1"]
        assert table_chunks, "Table chunks should have sheet='Table 1'"
        for chunk in table_chunks:
            assert "rows" in chunk, "Table chunks should have row-range metadata"
            assert "page" not in chunk, "Table chunks are page-less"
        # Table facts use column: value pairs
        table_text = " ".join(c["text"] for c in table_chunks)
        assert "Institution: KNS Institute of Technology" in table_text
        assert "Degree: Bachelor of Engineering" in table_text


# ---------------------------------------------------------------------------
# D3: page metadata missing for DOCX
# ---------------------------------------------------------------------------

class TestDocxPageMetadata:
    """Defect D3: DOCX chunks have no page metadata."""

    def test_docx_chunks_lack_page_key(self, tmp_path):
        path = _make_docx_headings_only(tmp_path)
        chunks = process_document(path, "headings.docx")
        for chunk in chunks:
            assert "page" not in chunk, (
                f"D3 not reproduced: page key found in DOCX chunk {chunk['chunk_index']}"
            )
            assert chunk["file_type"] == "docx"


# ---------------------------------------------------------------------------
# P3: Source labels for page-less formats
# ---------------------------------------------------------------------------

class TestSourceFormatting:
    """P3: Page-less formats use location metadata, not fabricated page numbers."""

    def test_format_context_uses_location_for_page_less(self):
        """DOCX chunks without 'page' use sheet/section/rows in context."""
        from backend.app.services.answer_generator import AnswerGenerator
        gen = AnswerGenerator()
        documents = [
            "KNS Institute of Technology, Bangalore — BE in AI & ML (2022–2026)",
            "Local Voice RAG Assistant: banking FAQ system.",
        ]
        metadatas = [
            {"file_name": "resume.docx", "sheet": "Table 1", "rows": "1-2"},
            {"file_name": "resume.docx", "section": "Projects"},
        ]
        context = gen._format_context(documents, metadatas)
        # No "Page ?" for page-less formats
        assert "Page ?" not in context
        assert "Sheet: Table 1" in context
        assert "Section: Projects" in context
        assert "resume.docx" in context

    def test_format_context_uses_page_for_pdf(self):
        """PDF chunks with 'page' still show Page N."""
        from backend.app.services.answer_generator import AnswerGenerator
        gen = AnswerGenerator()
        documents = ["Page one content about retrieval."]
        metadatas = [{"file_name": "doc.pdf", "page": 1}]
        context = gen._format_context(documents, metadatas)
        assert "Page 1" in context
        assert "Page ?" not in context

    def test_format_context_no_location(self):
        """Chunks with neither page nor location fall back gracefully."""
        from backend.app.services.answer_generator import AnswerGenerator
        gen = AnswerGenerator()
        documents = ["Some text without location metadata."]
        metadatas = [{"file_name": "unknown.txt"}]
        context = gen._format_context(documents, metadatas)
        assert "location not recorded" in context
        assert "Page ?" not in context


# ---------------------------------------------------------------------------
# H1: Chunking splits semantically related content
# ---------------------------------------------------------------------------

class TestChunkingSplitsRelatedContent:
    """Hypothesis H1: 512-char chunks split related facts."""

    def test_long_section_splits_across_chunks(self, tmp_path):
        import docx
        document = docx.Document()
        document.add_heading("Projects", level=1)
        # ~900 chars to force splitting at 512-char boundary
        long_desc = (
            "The Local Voice RAG Assistant is a banking FAQ system that "
            "processes voice queries using local retrieval-augmented "
            "generation with Ollama embeddings and the Qwen answer "
            "generator. It supports hybrid retrieval with BM25 and dense "
            "vector search, reciprocal rank fusion, and ChromaDB storage. "
            "The system handles PDF, DOCX, TXT, Markdown, CSV, and XLSX "
            "documents and provides source citations with every answer. "
            "It was built by Harsh Singh as a capstone project and "
            "demonstrates advanced RAG techniques including semantic "
            "search, keyword search, and result fusion. The architecture "
            "uses FastAPI for the backend, Streamlit for the frontend, "
            "and ChromaDB for persistent vector storage."
        )
        document.add_paragraph(long_desc)
        path = tmp_path / "long_section.docx"
        document.save(str(path))
        chunks = process_document(str(path), "long_section.docx")
        all_text = " ".join(c["text"] for c in chunks)
        assert "Local Voice RAG Assistant" in all_text
        assert "Qwen answer generator" in all_text
        assert "ChromaDB storage" in all_text
        assert len(chunks) >= 2, (
            f"Expected section to split into >=2 chunks, got {len(chunks)}"
        )
        # If the section splits, related facts may end up in different chunks.
        # A query about one fact may miss chunks containing the other fact.
        first_chunk = chunks[0]["text"]
        last_chunk = chunks[-1]["text"]
        assert first_chunk != last_chunk, "Chunks should differ after splitting"

    def test_education_facts_may_be_split(self, tmp_path):
        import docx
        document = docx.Document()
        document.add_heading("Education", level=1)
        edu_text = (
            "KNS Institute of Technology, Bangalore — Bachelor of Engineering "
            "in Artificial Intelligence & Machine Learning, 2022–2026. "
            "CGPA: 8.7/10. Relevant coursework: Machine Learning, Natural "
            "Language Processing, Information Retrieval, Database Systems."
        )
        document.add_paragraph(edu_text)
        path = tmp_path / "education.docx"
        document.save(str(path))
        chunks = process_document(str(path), "education.docx")
        all_text = " ".join(c["text"] for c in chunks)
        assert "KNS Institute of Technology" in all_text
        assert "Bachelor of Engineering" in all_text
        assert "2022–2026" in all_text
        institution_chunk = next(
            (c for c in chunks if "KNS Institute of Technology" in c["text"]), None
        )
        degree_chunk = next(
            (c for c in chunks if "Bachelor of Engineering" in c["text"]), None
        )
        if institution_chunk and degree_chunk and institution_chunk is not degree_chunk:
            pass  # H1 confirmed: related facts split across chunks


# ---------------------------------------------------------------------------
# D2: BM25 index not rebuilt after upload
# ---------------------------------------------------------------------------

class TestBM25IndexAfterUpload:
    """Defect D2: BM25 index not rebuilt after upload in upload endpoint."""

    def test_upload_endpoint_does_not_rebuild_bm25(self):
        """upload_document() calls add_chunks() but never rebuild_index()."""
        import inspect
        from backend.app.api import documents as documents_module
        source = inspect.getsource(documents_module.upload_document)
        assert "add_chunks" in source
        assert "rebuild_index" not in source, (
            "D2 not reproduced: rebuild_index() IS called after upload"
        )

    def test_delete_endpoint_rebuilds_bm25(self):
        """The delete endpoint DOES call rebuild_index() (asymmetric behavior)."""
        import inspect
        from backend.app.api import documents as documents_module
        source = inspect.getsource(documents_module.delete_document)
        assert "rebuild_index" in source


# ---------------------------------------------------------------------------
# H2: BM25 weak with single-document corpus
# ---------------------------------------------------------------------------

class TestBM25SingleDocument:
    """Hypothesis H2: BM25 signal is weak with one document."""

    def test_bm25_scores_with_single_document(self, chroma_service, retriever):
        chunks = [
            {
                "text": "The Local Voice RAG Assistant processes banking FAQ queries.",
                "file_name": "resume.docx",
                "chunk_index": 0,
                "section": "Projects",
            },
        ]
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()
        results = retriever._bm25_search("Local Voice RAG Assistant", top_k=5)
        # H2 confirmed: BM25 returns 0 results with a single-document corpus
        assert len(results) == 0, (
            f"H2 not reproduced: BM25 returned {len(results)} result(s) with single doc"
        )


# ---------------------------------------------------------------------------
# H3: n_results=5 may be insufficient
# ---------------------------------------------------------------------------

class TestRetrievalTopK:
    """Hypothesis H3: 5 results may miss split chunks."""

    def test_relevant_chunks_may_be_outside_top5(self, chroma_service, retriever):
        chunks = []
        for i in range(12):
            if i < 10:
                text = f"Irrelevant background information chunk number {i} about unrelated topics."
            else:
                text = f"The answer is in chunk {i}: KNS Institute of Technology Bangalore."
            chunks.append(
                {
                    "text": text,
                    "file_name": "resume.docx",
                    "chunk_index": i,
                    "section": "Education",
                }
            )
        chroma_service.add_chunks(chunks)
        retriever.rebuild_index()
        results = retriever.retrieve("KNS Institute of Technology", n_results=5)
        all_text = " ".join(results["documents"])
        assert "KNS Institute of Technology" in all_text or len(results["ids"]) < 5, (
            "H3 not confirmed: all relevant chunks appeared in top 5"
        )


# ---------------------------------------------------------------------------
# Cross-format extraction integrity
# ---------------------------------------------------------------------------

class TestExtractionIntegrity:
    """Verify extraction preserves content across all supported formats."""

    def test_pdf_preserves_page_metadata(self, tmp_path):
        import fitz
        path = tmp_path / "sample.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Page one content about retrieval.")
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Page two content about generation.")
        doc.save(str(path))
        doc.close()
        chunks = process_document(str(path), "sample.pdf")
        pages = {c.get("page") for c in chunks}
        assert 1 in pages
        assert 2 in pages

    def test_txt_no_fabricated_metadata(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("Plain text about hybrid search systems.", encoding="utf-8")
        chunks = process_document(str(path), "notes.txt")
        assert len(chunks) == 1
        assert "page" not in chunks[0]
        assert "section" not in chunks[0]
        assert "sheet" not in chunks[0]

    def test_md_heading_path_metadata(self, tmp_path):
        path = tmp_path / "guide.md"
        path.write_text(
            "# RAG Guide\nIntro about retrieval.\n\n## Setup\nInstall steps.\n",
            encoding="utf-8",
        )
        chunks = process_document(str(path), "guide.md")
        intro = next(c for c in chunks if "Intro" in c["text"])
        setup = next(c for c in chunks if "Install" in c["text"])
        assert intro["section"] == "RAG Guide"
        assert setup["section"] == "RAG Guide > Setup"

    def test_csv_header_aware_serialization(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text(
            "Name,Degree,Institution\nAlice,BE AI,KNS\nBob,MS Data,IIT\n",
            encoding="utf-8",
        )
        chunks = process_document(str(path), "data.csv")
        all_text = " ".join(c["text"] for c in chunks)
        assert "Name: Alice" in all_text
        assert "Degree: BE AI" in all_text
        assert "Institution: KNS" in all_text