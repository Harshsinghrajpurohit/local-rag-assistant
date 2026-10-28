"""
Structured query filtering for spreadsheet data.

Provides deterministic row-level filtering for numeric comparison queries
(e.g., "stipend >= 40000 AND duration <= 3") applied to tabular rows
extracted from CSV/Excel documents.

All functions are pure and side-effect free.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class FilterOutcome(Enum):
    MATCH = "match"
    NO_MATCH = "no_match"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Condition:
    column: str
    operator: str
    value: float


def parse_conditions(query: str) -> List[Condition]:
    conditions: List[Condition] = []
    parts = re.split(r"\s+AND\s+", query, flags=re.IGNORECASE)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        condition = _parse_single_condition(part)
        if condition is not None:
            conditions.append(condition)
    return conditions


def _parse_single_condition(part: str) -> Optional[Condition]:
    for op_symbol, op_name in (
        (">=", ">="),
        ("<=", "<="),
        ("==", "=="),
        (">", ">"),
        ("<", "<"),
        ("=", "=="),
    ):
        pattern = re.compile(
            r"^(.*?)\s*" + re.escape(op_symbol) + r"\s*(.*?)\s*$",
            re.IGNORECASE,
        )
        match = pattern.match(part)
        if match:
            column = match.group(1).strip()
            value_str = match.group(2).strip()
            operator = op_name
            break
    else:
        for word_op, op_name in (("at least", ">="), ("at most", "<=")):
            pattern = re.compile(
                r"^(.*?)\s+" + re.escape(word_op) + r"\s+(.*?)\s*$",
                re.IGNORECASE,
            )
            match = pattern.match(part)
            if match:
                column = match.group(1).strip()
                value_str = match.group(2).strip()
                operator = op_name
                break
        else:
            return None

    column = re.sub(
        r"^(what|which|how|show|find|get|list|are|is)\s+",
        "",
        column,
        flags=re.IGNORECASE,
    ).strip()
    if not column:
        return None

    value = parse_numeric_value(value_str)
    if value is None:
        return None

    return Condition(column=column, operator=operator, value=value)


def parse_numeric_value(text: str) -> Optional[float]:
    if not text:
        return None
    cleaned = text.replace(chr(8377), "").replace("$", "").replace(",", "").strip()
    if "\u2013" in cleaned or "-" in cleaned:
        return None
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def evaluate_condition(row_value: str, condition: Condition) -> FilterOutcome:
    numeric_value = parse_numeric_value(row_value)
    if numeric_value is None:
        return FilterOutcome.UNKNOWN
    if condition.operator == ">=":
        return FilterOutcome.MATCH if numeric_value >= condition.value else FilterOutcome.NO_MATCH
    if condition.operator == "<=":
        return FilterOutcome.MATCH if numeric_value <= condition.value else FilterOutcome.NO_MATCH
    if condition.operator == ">":
        return FilterOutcome.MATCH if numeric_value > condition.value else FilterOutcome.NO_MATCH
    if condition.operator == "<":
        return FilterOutcome.MATCH if numeric_value < condition.value else FilterOutcome.NO_MATCH
    if condition.operator == "==":
        return FilterOutcome.MATCH if numeric_value == condition.value else FilterOutcome.NO_MATCH
    return FilterOutcome.UNKNOWN


def parse_tabular_segment(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not re.match(r"^\s*Row\s+\d+\s*:", text, re.IGNORECASE):
        return []
    records: List[Dict[str, Any]] = []
    file_name = metadata.get("file_name", "unknown")
    sheet = metadata.get("sheet")
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        row_match = re.match(r"^Row\s+(\d+)\s*:\s*(.*)$", line, re.IGNORECASE)
        if not row_match:
            continue
        row_number = int(row_match.group(1))
        cells_text = row_match.group(2).strip()
        cells = cells_text.split(" | ")
        record: Dict[str, Any] = {
            "_file_name": file_name,
            "_sheet": sheet,
            "_row_number": row_number,
        }
        for cell in cells:
            cell = cell.strip()
            if not cell:
                continue
            if ": " in cell:
                col_name, col_value = cell.split(": ", 1)
                record[col_name.strip()] = col_value.strip()
            else:
                record[f"cell_{len(record)}"] = cell
        records.append(record)
    return records


def filter_structured_rows(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    conditions: List[Condition],
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    matching_documents: List[str] = []
    matching_metadatas: List[Dict[str, Any]] = []
    matching_ids: List[str] = []
    for doc_text, metadata, doc_id in zip(documents, metadatas, ids):
        records = parse_tabular_segment(doc_text, metadata)
        for record in records:
            all_match = True
            any_unknown = False
            for condition in conditions:
                row_value = None
                for key, val in record.items():
                    if key.startswith("_"):
                        continue
                    if key.lower() == condition.column.lower():
                        row_value = val
                        break
                if row_value is None:
                    all_match = False
                    break
                outcome = evaluate_condition(row_value, condition)
                if outcome == FilterOutcome.UNKNOWN:
                    any_unknown = True
                    break
                if outcome == FilterOutcome.NO_MATCH:
                    all_match = False
                    break
            if not all_match or any_unknown:
                continue
            row_metadata = dict(metadata)
            row_metadata["_file_name"] = record.get("_file_name", "unknown")
            row_metadata["_sheet"] = record.get("_sheet")
            row_metadata["_row_number"] = record.get("_row_number")
            cells = []
            for key, val in record.items():
                if key.startswith("_"):
                    continue
                cells.append(f"{key}: {val}")
            row_text = f"Row {record.get('_row_number', '?')}: " + " | ".join(cells)
            matching_documents.append(row_text)
            matching_metadatas.append(row_metadata)
            matching_ids.append(f"{doc_id}_row{record.get('_row_number', '?')}")
    return matching_documents, matching_metadatas, matching_ids

