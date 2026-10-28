"""
Format-aware document processing for multi-format RAG ingestion.

Dispatches by file extension to per-format extractors that produce a common
segment representation, then reuses the existing PDF chunking logic so every
format flows through the same embedding / storage / retrieval / generation
pipeline (no separate RAG pipelines).

Segment: {"text": str, "location": dict} where location may contain:
    {"page": int}                      - PDF
    {"section": "A > B"}               - DOCX / Markdown (heading path)
    {"sheet": "Sales", "rows": "2-41"} - XLSX
    {"rows": "2-41"}                   - CSV
    {}                                 - TXT

Chunk (returned to ChromaService): {"text", "file_name", "file_type",
"chunk_index", "segment"} plus flattened location keys (only those that
actually exist - nothing is fabricated).
"""

import csv
import io
import re
from pathlib import Path
from typing import Any, Dict, List

from backend.app.exceptions import ExtractionFailedError
from backend.app.services.pdf_processor import process_pdf, split_into_chunks

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx")

CHUNK_SIZE = 512          # matches the existing PDF defaults
CHUNK_OVERLAP = 50        # matches the existing PDF defaults
ROWS_PER_SEGMENT = 40     # tabular rows serialized per segment before chunking

# Explicit safety limit for spreadsheets (per sheet / per CSV file).
# Exceeding it raises a clear error - data is NEVER silently truncated.
MAX_SPREADSHEET_ROWS_PER_SHEET = 5000


def _decode_text(raw: bytes) -> str:
    """Decode text bytes with a sensible encoding fallback chain."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ExtractionFailedError(
        "Could not decode text file (tried utf-8 and latin-1 encodings)"
    )


def process_document(file_path: str, original_filename: str) -> List[Dict[str, Any]]:
    """
    Extract and chunk a document of any supported format.

    Args:
        file_path: Path to the saved upload on disk
        original_filename: Original filename (extension detection + stored
            as the chunk file_name)

    Returns:
        List of chunk dicts ready for ChromaService.add_chunks()

    Raises:
        ExtractionFailedError: Unsupported format, empty/unreadable file,
            or a spreadsheet exceeding the row limit
    """
    ext = Path(original_filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ExtractionFailedError(f"Unsupported file extension '{ext}'")

    if ext == ".pdf":
        # Existing PDF behavior, preserved exactly (page metadata included)
        try:
            chunks = process_pdf(file_path)
        except ValueError as e:
            raise ExtractionFailedError(f"PDF could not be parsed: {e}")
        for chunk in chunks:
            chunk["file_type"] = "pdf"
            chunk.setdefault("segment", chunk.get("page", 0))
        return chunks

    if ext == ".docx":
        segments = _extract_docx(file_path)
    elif ext == ".txt":
        segments = _extract_txt(file_path)
    elif ext == ".md":
        segments = _extract_md(file_path)
    elif ext == ".csv":
        segments = _extract_csv(file_path)
    else:  # .xlsx
        segments = _extract_xlsx(file_path)

    return _chunk_segments(segments, original_filename, ext)


def _chunk_segments(
    segments: List[Dict[str, Any]], original_filename: str, ext: str
) -> List[Dict[str, Any]]:
    """Chunk all segments with the shared splitter and flatten locations."""
    chunks: List[Dict[str, Any]] = []
    chunk_index = 0
    for seg_idx, segment in enumerate(segments):
        pieces = split_into_chunks(segment["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for piece in pieces:
            if not piece.strip():
                continue
            chunk: Dict[str, Any] = {
                "text": piece,
                "file_name": original_filename,
                "file_type": ext.lstrip("."),
                "chunk_index": chunk_index,
                "segment": seg_idx,
            }
            for key, value in segment["location"].items():
                chunk[key] = value
            chunks.append(chunk)
            chunk_index += 1
    if not chunks:
        raise ExtractionFailedError("Document contains no extractable text")
    return chunks


def _extract_txt(file_path: str) -> List[Dict[str, Any]]:
    """Plain text: single segment, no location metadata."""
    raw = Path(file_path).read_bytes()
    text = _decode_text(raw)
    if not text.strip():
        raise ExtractionFailedError("Text file contains no readable text")
    return [{"text": text, "location": {}}]


def _extract_md(file_path: str) -> List[Dict[str, Any]]:
    """Markdown: segment by heading path; code fences are kept as text."""
    text = _decode_text(Path(file_path).read_bytes())
    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")
    segments: List[Dict[str, Any]] = []
    stack: List[tuple] = []
    lines: List[str] = []
    location: Dict[str, str] = {}
    in_fence = False

    def flush() -> None:
        block = "\n".join(lines).strip()
        if block:
            segments.append({"text": block, "location": dict(location)})
        lines.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            lines.append(line)
            continue
        match = None if in_fence else heading_re.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip() or f"Section {len(stack) + 1}"
            stack = [(lvl, t) for lvl, t in stack if lvl < level]
            stack.append((level, title))
            location = {"section": " > ".join(t for _, t in stack)}
        else:
            lines.append(line)
    flush()

    if not segments:
        raise ExtractionFailedError("Markdown file contains no readable text")
    return segments


def _extract_docx(file_path: str) -> List[Dict[str, Any]]:
    """Word documents: heading-bounded sections with heading-path location."""
    try:
        import docx  # python-docx
    except ImportError:
        raise ExtractionFailedError("python-docx is not installed")
    try:
        document = docx.Document(file_path)
    except Exception as e:
        raise ExtractionFailedError(f"Could not read DOCX file: {e}")

    segments: List[Dict[str, Any]] = []
    stack: List[tuple] = []
    texts: List[str] = []
    location: Dict[str, str] = {}

    def flush() -> None:
        block = "\n".join(t for t in texts if t.strip())
        if block.strip():
            segments.append({"text": block, "location": dict(location)})
        texts.clear()

    for para in document.paragraphs:
        style_name = (para.style.name if para.style is not None else "") or ""
        text = para.text.strip()
        if style_name.lower().startswith("heading"):
            flush()
            level_match = re.search(r"\d", style_name)
            level = int(level_match.group()) if level_match else 1
            title = text or f"Section {len(stack) + 1}"
            stack = [(lvl, t) for lvl, t in stack if lvl < level]
            stack.append((level, title))
            location = {"section": " > ".join(t for _, t in stack)}
        elif text:
            texts.append(para.text)
    flush()

    # Extract tables after paragraphs. Tables are independent block-level
    # elements in DOCX; they are not part of the heading-bounded paragraph
    # flow, so they get their own sheet/rows location metadata.
    table_index = 0
    for table in document.tables:
        table_index += 1
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(c.strip() for c in cells):
                rows.append(cells)
        if len(rows) < 2:
            # Skip empty or header-only tables - nothing to retrieve.
            continue
        header = rows[0]
        data_rows = rows[1:]
        table_segments = _tabular_segments(header, data_rows, sheet_label=f"Table {table_index}")
        segments.extend(table_segments)

    if not segments:
        raise ExtractionFailedError("DOCX file contains no readable text")
    return segments


def _tabular_segments(
    header: List[str], data_rows: List[List[str]], sheet_label: str = None
) -> List[Dict[str, Any]]:
    """
    Serialize tabular rows into text segments with column context.

    Each line reads: "Row 12: Region: EMEA | Revenue: 4200 | ..."
    so retrieved chunks stay understandable on their own. Segments cover
    ROWS_PER_SEGMENT rows and carry a "rows" range location (plus "sheet"
    for Excel worksheets).
    """
    segments: List[Dict[str, Any]] = []
    for start in range(0, len(data_rows), ROWS_PER_SEGMENT):
        batch = data_rows[start:start + ROWS_PER_SEGMENT]
        lines = []
        for offset, row in enumerate(batch):
            row_number = start + offset + 2  # +2: header occupies row 1
            cells = [
                f"{header[i] if i < len(header) else f'column_{i + 1}'}: "
                f"{row[i].strip() if i < len(row) else ''}"
                for i in range(max(len(header), len(row)))
            ]
            lines.append(f"Row {row_number}: " + " | ".join(cells))
        location = {"rows": f"{start + 2}-{start + 1 + len(batch)}"}
        if sheet_label:
            location["sheet"] = sheet_label
        segments.append({"text": "\n".join(lines), "location": location})
    return segments


def _extract_csv(file_path: str) -> List[Dict[str, Any]]:
    """CSV: header-aware row serialization with row-range location."""
    text = _decode_text(Path(file_path).read_bytes())
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as e:
        raise ExtractionFailedError(f"Could not parse CSV file: {e}")
    if not rows or not any(cell.strip() for row in rows for cell in row):
        raise ExtractionFailedError("CSV file contains no data")

    header = [
        h.strip() or f"column_{i + 1}" for i, h in enumerate(rows[0])
    ]
    data_rows = [r for r in rows[1:] if any(c.strip() for c in r)]
    if not data_rows:
        raise ExtractionFailedError("CSV file has a header but no data rows")
    if len(data_rows) > MAX_SPREADSHEET_ROWS_PER_SHEET:
        raise ExtractionFailedError(
            f"CSV has {len(data_rows)} data rows, exceeding the limit of "
            f"{MAX_SPREADSHEET_ROWS_PER_SHEET} rows per file. "
            "Split the file into smaller parts and retry."
        )
    return _tabular_segments(header, data_rows)


def _extract_xlsx(file_path: str) -> List[Dict[str, Any]]:
    """Excel: per-sheet header-aware serialization with sheet+row location."""
    try:
        import openpyxl
    except ImportError:
        raise ExtractionFailedError("openpyxl is not installed")
    try:
        workbook = openpyxl.load_workbook(
            file_path, read_only=True, data_only=True
        )
    except Exception as e:
        raise ExtractionFailedError(f"Could not read XLSX file: {e}")

    segments: List[Dict[str, Any]] = []
    try:
        for sheet in workbook.worksheets:
            row_iterator = sheet.iter_rows(values_only=True)
            try:
                header_row = next(row_iterator)
            except StopIteration:
                continue  # empty worksheet - skip
            header = [
                str(h).strip() if h is not None and str(h).strip()
                else f"column_{i + 1}"
                for i, h in enumerate(header_row)
            ]
            data_rows: List[List[str]] = []
            for row in row_iterator:
                cells = ["" if c is None else str(c) for c in row]
                if any(c.strip() for c in cells):
                    data_rows.append(cells)
            if not data_rows:
                continue  # header-only sheet - skip
            if len(data_rows) > MAX_SPREADSHEET_ROWS_PER_SHEET:
                raise ExtractionFailedError(
                    f"Worksheet '{sheet.title}' has {len(data_rows)} data rows, "
                    f"exceeding the limit of {MAX_SPREADSHEET_ROWS_PER_SHEET} "
                    "rows per sheet. Split the data and retry."
                )
            segments.extend(
                _tabular_segments(header, data_rows, sheet_label=sheet.title)
            )
    finally:
        workbook.close()

    if not segments:
        raise ExtractionFailedError("XLSX file contains no readable data")
    return segments