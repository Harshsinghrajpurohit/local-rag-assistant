import pytest
import tempfile
import shutil
from pathlib import Path
from backend.app.services.chroma_service import ChromaService
from backend.app.services.embeddings import EmbeddingsService
from backend.app.services.hybrid_retriever import HybridRetriever

class TestDocumentOperations:
    @pytest.fixture
    def temp_db_path(self):
        temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def embeddings_service(self):
        return EmbeddingsService()

    @pytest.fixture
    def chroma_service(self, temp_db_path, embeddings_service):
        return ChromaService(db_path=temp_db_path, embeddings_service=embeddings_service)

    @pytest.fixture
    def retriever(self, chroma_service):
        return HybridRetriever(chroma_service)

    def test_document_add_and_retrieve(self, chroma_service, retriever):
        """Test adding chunks to ChromaDB and retrieving them via HybridRetriever."""
        # Create test chunks with proper metadata
        chunks = []
        for i in range(2):
            chunk = {
                "text": f"Content about machine learning chunk {i+1}",
                "page": i + 1,
                "file_name": "test_document.pdf",
                "chunk_index": i
            }
            chunks.append(chunk)

        # Add chunks to ChromaDB
        chroma_service.add_chunks([
            {
                "text": chunk["text"],
                "page": chunk["page"],
                "file_name": chunk["file_name"],
                "chunk_index": chunk["chunk_index"]
            }
            for chunk in chunks
        ])

        # Rebuild BM25 index
        retriever.rebuild_index()

        # Test document retrieval
        results = retriever.retrieve("test", n_results=1)
        assert len(results["ids"]) == 1
        assert len(results["documents"]) == 1
        assert len(results["metadatas"]) == 1

        # Verify that retrieved documents have proper metadata
        meta = results["metadatas"][0]
        assert meta["file_name"] == "test_document.pdf"
        assert meta["page"] == 1  # First page

        print("Document add and retrieve test completed successfully")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])