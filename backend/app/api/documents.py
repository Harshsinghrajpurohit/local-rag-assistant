"""
FastAPI router for document operations: upload, list, and delete.
"""

import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile

from ..exceptions import (
    DeleteFailedError,
    DeletePartialError,
    DocumentNotFoundError,
    EmptyFileError,
    ExtractionFailedError,
    FileTooLargeError,
    InternalServerError,
    PdfCorruptedError,
    UnsupportedFormatError,
)
import requests
from ..api.chat import get_retriever  # shared retriever getter
from ..models import (
    DeleteResponse,
    Document,
    DocumentListResponse,
    DocumentMetadata,
    DocumentResponse,
    generate_request_id,
    get_current_timestamp,
)
from ..services.document_processor import SUPPORTED_EXTENSIONS, process_document

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Upload and process a PDF document.

    Args:
        file: Uploaded document (PDF, DOCX, TXT, MD, CSV, or XLSX)
        request: FastAPI request for generating request_id

    Returns:
        DocumentResponse with document metadata
    """
    # Generate request ID for logging correlation
    request_id = generate_request_id()

    # Validate file format by extension (browser content-type headers are
    # unreliable for Office formats)
    file_name = file.filename or ""
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(ext or "unknown")

    # Check file size (50 MB limit)
    file_size = await file.read()
    file_size_bytes = len(file_size)
    if file_size_bytes == 0:
        raise EmptyFileError()
    if file_size_bytes > 50 * 1024 * 1024:  # 50 MB
        raise FileTooLargeError(file_size_bytes)

    # Light content sanity checks (magic bytes) where practical
    if ext == ".pdf" and not file_size.startswith(b"%PDF"):
        raise PdfCorruptedError()
    if ext in (".docx", ".xlsx") and not file_size.startswith(b"PK"):
        raise ExtractionFailedError(
            "File does not look like a valid DOCX/XLSX (missing ZIP signature)"
        )

    # Reset file stream
    file.file.seek(0)

    try:
        # Save upload to a temporary file (processors require a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_size)
            tmp_path = tmp_file.name

        try:
            # Extract text into chunks with metadata (format-aware)
            chunks = process_document(tmp_path, original_filename=file_name)
        except ExtractionFailedError:
            raise
        except ValueError:
            raise PdfCorruptedError()
        finally:
            os.unlink(tmp_path)

        # Create document metadata
        doc_id = str(uuid.uuid4())
        upload_time = get_current_timestamp()
        size_bytes = file_size_bytes
        num_chunks = len(chunks)

        # Create DocumentMetadata
        metadata = DocumentMetadata(
            doc_id=doc_id,
            file_name=file_name,
            file_type=ext.lstrip("."),
            upload_time=upload_time,
            size_bytes=size_bytes,
            num_chunks=num_chunks,
            chromadb_chunk_ids=[f"{file_name}_{doc_id}_page_{i}_chunk_{j}" for i in range(num_chunks) for j in range(1)]
        )

        # Store in ChromaDB via ChromaService
        from ..services.chroma_service import ChromaService
        try:
            chroma_service = ChromaService()

            # Attach document-level metadata to every chunk before persisting.
            # file_name is overwritten with the original uploaded filename because
            # process_pdf derives file_name from the temporary file path.
            for chunk in chunks:
                chunk["doc_id"] = doc_id
                chunk["file_name"] = file_name
                chunk["file_type"] = ext.lstrip(".")
                chunk["upload_time"] = upload_time
                chunk["size_bytes"] = size_bytes

            added_ids = chroma_service.add_chunks(chunks)
            get_retriever().rebuild_index()

            return DocumentResponse(
                doc_id=doc_id,
                file_name=file_name,
                file_type=ext.lstrip("."),
                num_chunks=num_chunks,
                upload_time=upload_time,
                size_bytes=size_bytes
            )
        except requests.exceptions.RequestException as e:
            logger.error("Ollama request failed during document upload: %s", e)
            raise OllamaUnavailableError("LLM service unavailable during upload") from e
        except Exception as e:
            logger.exception("Upload failed")
            raise InternalServerError() from e

    except PdfCorruptedError as e:
        raise PdfCorruptedError()
    except Exception as e:
        raise InternalServerError()


@router.get("/documents/list", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents with metadata.

    Returns:
        DocumentListResponse with document list, total count, and total chunks
    """
    # In real implementation, this would query ChromaDB
    from ..services.chroma_service import ChromaService
    chroma_service = ChromaService()

    stats = chroma_service.get_collection_stats()
    collection_count = stats["document_count"]

    # If no documents, return empty list
    if collection_count == 0:
        return DocumentListResponse(
            documents=[],
            total_documents=0,
            total_chunks=0
        )

    # Fetch all chunks from ChromaDB
    all_docs = chroma_service.collection.get(
        include=["metadatas"]
    )

    # Group chunks by persisted doc_id. Legacy chunks without doc_id are
    # excluded from the listing but never modified or deleted.
    grouped = {}
    doc_order = []
    for chunk_metadata in all_docs["metadatas"] or []:
        chunk_doc_id = chunk_metadata.get("doc_id")
        if not chunk_doc_id:
            continue
        if chunk_doc_id not in grouped:
            grouped[chunk_doc_id] = []
            doc_order.append(chunk_doc_id)
        grouped[chunk_doc_id].append(chunk_metadata)

    # Build one Document entry per uploaded document
    documents = []
    for chunk_doc_id in doc_order:
        metadatas = grouped[chunk_doc_id]
        file_names = {m.get("file_name") for m in metadatas}
        file_types = {m.get("file_type") for m in metadatas}
        upload_times = {m.get("upload_time") for m in metadatas}
        size_bytes_values = {m.get("size_bytes") for m in metadatas}

        # Inconsistent metadata within one document: fail explicitly
        # instead of returning misleading values.
        if (
            len(file_names) != 1
            or len(file_types) != 1
            or len(upload_times) != 1
            or len(size_bytes_values) != 1
        ):
            raise InternalServerError()

        documents.append(Document(
            doc_id=chunk_doc_id,
            file_name=file_names.pop(),
            file_type=file_types.pop(),
            num_chunks=len(metadatas),
            upload_time=upload_times.pop(),
            size_bytes=size_bytes_values.pop(),
        ))

    # Rebuild BM25 index after successful upload

    return DocumentListResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=sum(d.num_chunks for d in documents)
    )


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(
    doc_id: str,
    request: Request
):
    """
    Delete a document and associated data.

    Args:
        doc_id: UUID of document to delete
        request: FastAPI request for generating request_id

    Returns:
        DeleteResponse with deletion confirmation
    """
    # Generate request ID for logging correlation
    request_id = generate_request_id()

    try:
        # Delete from ChromaDB via ChromaService
        from ..services.chroma_service import ChromaService
        from datetime import datetime, timezone

        chroma_service = ChromaService()

        # Identify chunks strictly by persisted doc_id metadata
        matches = chroma_service.collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )

        if not matches["ids"]:
            raise DocumentNotFoundError()

        # file_name must be consistent across the document's chunks
        file_names = {m.get("file_name") for m in matches["metadatas"] if m}
        if len(file_names) != 1 or None in file_names:
            raise DeleteFailedError()
        file_name = file_names.pop()

        # Delete only this document's chunk IDs
        delete_result = chroma_service.collection.delete(ids=matches["ids"])

        # Verify that no matching records remain before reporting success
        remaining = chroma_service.collection.get(where={"doc_id": doc_id})
        if remaining["ids"]:
            raise DeletePartialError()

        # Actual number of deleted records from ChromaDB's DeleteResult
        chunks_deleted = (
            delete_result["deleted"]
            if isinstance(delete_result, dict) and "deleted" in delete_result
            else len(matches["ids"])
        )

        # Rebuild BM25 index
        get_retriever().rebuild_index()

        return DeleteResponse(
            doc_id=doc_id,
            file_name=file_name,
            chunks_deleted=chunks_deleted,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

    except DocumentNotFoundError:
        raise
    except DeletePartialError:
        raise
    except DeleteFailedError:
        raise
    except Exception as e:
        raise DeleteFailedError() from e


def register_router(app):
    """Register the documents router with the main app."""
    app.include_router(router)


# If running directly
if __name__ == "__main__":
    import uvicorn
    from ..main import app
    uvicorn.run(app, host="0.0.0.0", port=8000)