"""
Phase 4 — Reproducible synthetic evaluation dataset for the RAG-Q&A document assistant.

This module defines a version-controlled benchmark dataset with synthetic documents
and evaluation cases. All content is synthetic; no private or personal information
is included. Documents are created in-memory (never written to real chroma_db).

Each evaluation case defines:
- question: The user question
- category: One of exact_keyword, semantic_paraphrase, structured_fact, summary,
              multi_part, multi_document, missing_evidence, conflicting_facts,
              distractor
- expected_facts: Facts that should appear in a correct answer
- sufficient_evidence: Whether the corpus contains enough evidence to answer
- relevant_doc_ids: Document IDs that contain relevant evidence
- distractor_doc_ids: Document IDs that are irrelevant (should NOT be retrieved)
- min_recall_k: Minimum recall expected at k=1,3,5 for retrieval metrics
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class EvalDocument:
    """A synthetic document for the benchmark."""
    doc_id: str
    filename: str
    content: str
    file_format: str  # "txt", "md", "csv", "xlsx", "docx", "pdf"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalCase:
    """A single evaluation case."""
    case_id: str
    question: str
    category: str
    expected_facts: List[str]
    sufficient_evidence: bool
    relevant_doc_ids: List[str]
    distractor_doc_ids: List[str] = field(default_factory=list)
    min_recall_at_1: float = 0.0
    min_recall_at_3: float = 0.0
    min_recall_at_5: float = 0.0
    min_mrr: float = 0.0


# ============================================================================
# SYNTHETIC DOCUMENTS
# ============================================================================

BENCHMARK_DOCUMENTS: List[EvalDocument] = [
    # ---------------------------------------------------------------
    # Document 1: Company policy handbook (TXT)
    # ---------------------------------------------------------------
    EvalDocument(
        doc_id="doc_policy",
        filename="employee_handbook.txt",
        file_format="txt",
        content=(
            "Acme Corporation Employee Handbook\n"
            "===============================\n\n"
            "Section 1: Leave Policy\n"
            "Employees are entitled to 20 days of paid annual leave per calendar year.\n"
            "Sick leave is accrued at 1 day per month, up to 12 days per year.\n"
            "Parental leave is 12 weeks for primary caregivers and 2 weeks for secondary.\n"
            "Unpaid leave requires manager approval and HR review.\n\n"
            "Section 2: Expense Reimbursement\n"
            "All business expenses must be submitted within 30 days of the expense date.\n"
            "Meal expenses are capped at $75 per day.\n"
            "Travel expenses require original receipts for amounts over $25.\n"
            "Mileage reimbursement is $0.58 per mile for personal vehicle use.\n\n"
            "Section 3: Remote Work Policy\n"
            "Employees may work remotely up to 3 days per week with manager approval.\n"
            "Core collaboration hours are 10:00 AM to 3:00 PM local time.\n"
            "Equipment stipend of $500 is available for home office setup.\n"
        ),
    ),

    # ---------------------------------------------------------------
    # Document 2: Product specifications (Markdown)
    # ---------------------------------------------------------------
    EvalDocument(
        doc_id="doc_specs",
        filename="product_specs.md",
        file_format="md",
        content=(
            "# QuantumEdge QX-500 Product Specifications\n\n"
            "## Overview\n"
            "The QuantumEdge QX-500 is an enterprise-grade data processing appliance\n"
            "designed for high-throughput analytics workloads.\n\n"
            "## Technical Specifications\n"
            "- Processor: 32-core Xeon Platinum 8480+\n"
            "- Memory: 512 GB DDR5 ECC\n"
            "- Storage: 15.36 TB NVMe SSD (2x 7.68 TB)\n"
            "- Network: 2x 100 GbE SFP28, 4x 25 GbE\n"
            "- Power: 750W redundant PSU\n"
            "- Dimensions: 2U rackmount\n"
            "- Weight: 28.5 kg\n\n"
            "## Performance Benchmarks\n"
            "Sequential read: 7.2 GB/s\n"
            "Sequential write: 6.8 GB/s\n"
            "Random 4K read: 1.4M IOPS\n"
            "Random 4K write: 950K IOPS\n\n"
            "## Compliance\n"
            "ISO 27001, SOC 2 Type II, GDPR, HIPAA\n"
        ),
    ),

    # ---------------------------------------------------------------
    # Document 3: Quarterly results (CSV)
    # ---------------------------------------------------------------
    EvalDocument(
        doc_id="doc_quarterly",
        filename="quarterly_results.csv",
        file_format="csv",
        content=(
            "Quarter,Revenue,Profit,Employees,Region\n"
            "Q1-2024,12500000,1850000,142,North America\n"
            "Q2-2024,13800000,2100000,148,North America\n"
            "Q3-2024,14200000,2250000,153,North America\n"
            "Q4-2024,15100000,2400000,160,North America\n"
            "Q1-2025,16500000,2750000,168,North America\n"
            "Q2-2025,17200000,2900000,175,North America\n"
            "Q1-2024,9800000,1200000,89,Europe\n"
            "Q2-2024,10500000,1450000,92,Europe\n"
            "Q3-2024,11200000,1600000,95,Europe\n"
            "Q4-2024,11900000,1750000,98,Europe\n"
        ),
    ),

    # ---------------------------------------------------------------
    # Document 4: Project timeline (Markdown)
    # ---------------------------------------------------------------
    EvalDocument(
        doc_id="doc_timeline",
        filename="project_timeline.md",
        file_format="md",
        content=(
            "# Aurora Project Timeline\n\n"
            "## Phase 1: Discovery (Jan 2024 - Mar 2024)\n"
            "Requirements gathering and stakeholder interviews completed.\n"
            "Budget allocated: $2.1 million.\n\n"
            "## Phase 2: Development (Apr 2024 - Sep 2024)\n"
            "Core platform development in progress.\n"
            "Team: 12 engineers, 4 designers, 2 product managers.\n\n"
            "## Phase 3: Testing (Oct 2024 - Dec 2024)\n"
            "Quality assurance and user acceptance testing.\n"
            "Bug count: 47 critical, 123 minor.\n\n"
            "## Phase 4: Launch (Q1 2025)\n"
            "Public release scheduled for March 15, 2025.\n"
            "Marketing budget: $500,000.\n"
        ),
    ),

    # ---------------------------------------------------------------
    # Document 5: Irrelevant distractor (TXT)
    # ---------------------------------------------------------------
    EvalDocument(
        doc_id="doc_distractor",
        filename="weather_forecast.txt",
        file_format="txt",
        content=(
            "Regional Weather Forecast\n"
            "========================\n\n"
            "Monday: Partly cloudy, high 72F, low 55F\n"
            "Tuesday: Rain likely, high 68F, low 60F\n"
            "Wednesday: Sunny, high 75F, low 58F\n"
            "Thursday: Thunderstorms, high 70F, low 62F\n"
            "Friday: Clear skies, high 78F, low 55F\n\n"
            "This document contains no information about company policies,\n"
            "products, or personnel. It is a distractor document for testing.\n"
        ),
    ),

    # ---------------------------------------------------------------
    # Document 6: Conflicting policy document (TXT)
    # ---------------------------------------------------------------
    EvalDocument(
        doc_id="doc_policy_v2",
        filename="employee_handbook_v2.txt",
        file_format="txt",
        content=(
            "Acme Corporation Employee Handbook (Updated v2)\n"
            "================================================\n\n"
            "Section 1: Leave Policy\n"
            "Employees are entitled to 25 days of paid annual leave per calendar year.\n"
            "Sick leave is accrued at 1.5 days per month, up to 18 days per year.\n"
            "Parental leave is 16 weeks for primary caregivers and 4 weeks for secondary.\n"
        ),
    ),

    # ---------------------------------------------------------------
    # Document 7: Employee directory (CSV)
    # ---------------------------------------------------------------
    EvalDocument(
        doc_id="doc_directory",
        filename="employee_directory.csv",
        file_format="csv",
        content=(
            "Name,Role,Department,Start Date,Office\n"
            "Alice Chen,Senior Engineer,Engineering,2022-03-15,New York\n"
            "Bob Wilson,Product Manager,Product,2021-07-10,San Francisco\n"
            "Carol Davis,HR Manager,Human Resources,2020-11-02,Chicago\n"
            "David Kim,Sales Executive,Sales,2023-01-20,Austin\n"
            "Eva Martinez,Designer,Design,2022-09-05,Remote\n"
        ),
    ),
]

# ============================================================================
# EVALUATION CASES
# ============================================================================

BENCHMARK_CASES: List[EvalCase] = [
    # ------------------------------------------------------------------
    # Exact keyword questions
    # ------------------------------------------------------------------
    EvalCase(
        case_id="KW-001",
        question="How many days of paid annual leave are employees entitled to?",
        category="exact_keyword",
        expected_facts=["20 days", "paid annual leave", "per calendar year"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_policy"],
        distractor_doc_ids=["doc_distractor"],
        min_recall_at_1=0.0,
        min_recall_at_3=1.0,
        min_recall_at_5=1.0,
        min_mrr=0.3,
    ),
    EvalCase(
        case_id="KW-002",
        question="What is the mileage reimbursement rate?",
        category="exact_keyword",
        expected_facts=["$0.58", "per mile", "personal vehicle"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_policy"],
        distractor_doc_ids=["doc_distractor"],
        min_recall_at_1=0.0,
        min_recall_at_3=1.0,
        min_recall_at_5=1.0,
        min_mrr=0.3,
    ),
    EvalCase(
        case_id="KW-003",
        question="What is the Q3-2024 revenue for North America?",
        category="exact_keyword",
        expected_facts=["14200000", "Q3-2024", "North America"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_quarterly"],
        distractor_doc_ids=["doc_distractor"],
        min_recall_at_1=0.4,
        min_recall_at_3=0.7,
        min_recall_at_5=1.0,
        min_mrr=0.4,
    ),

    # ------------------------------------------------------------------
    # Semantic paraphrase questions
    # ------------------------------------------------------------------
    EvalCase(
        case_id="SP-001",
        question="What is the yearly vacation allowance for staff?",
        category="semantic_paraphrase",
        expected_facts=["20 days", "paid annual leave"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_policy"],
        distractor_doc_ids=["doc_distractor"],
        min_recall_at_1=0.3,
        min_recall_at_3=0.6,
        min_recall_at_5=0.9,
        min_mrr=0.3,
    ),
    EvalCase(
        case_id="SP-002",
        question="How much can I spend on meals per day when traveling for work?",
        category="semantic_paraphrase",
        expected_facts=["$75", "per day", "meal expenses", "capped"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_policy"],
        distractor_doc_ids=["doc_distractor"],
        min_recall_at_1=0.3,
        min_recall_at_3=0.6,
        min_recall_at_5=0.9,
        min_mrr=0.3,
    ),
    EvalCase(
        case_id="SP-003",
        question="When is the Aurora project expected to launch publicly?",
        category="semantic_paraphrase",
        expected_facts=["March 15, 2025", "Q1 2025", "launch"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_timeline"],
        distractor_doc_ids=["doc_distractor"],
        min_recall_at_1=0.0,
        min_recall_at_3=0.0,
        min_recall_at_5=0.0,
        min_mrr=0.0,
    ),

    # ------------------------------------------------------------------
    # Structured fact questions (table/CSV data)
    # ------------------------------------------------------------------
    EvalCase(
        case_id="SF-001",
        question="What is Alice Chen's role and department?",
        category="structured_fact",
        expected_facts=["Senior Engineer", "Engineering", "Alice Chen"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_directory"],
        distractor_doc_ids=["doc_distractor"],
        min_recall_at_1=0.5,
        min_recall_at_3=0.8,
        min_recall_at_5=1.0,
        min_mrr=0.4,
    ),
    EvalCase(
        case_id="SF-002",
        question="What is the Q4-2024 profit for North America?",
        category="structured_fact",
        expected_facts=["2400000", "Q4-2024", "North America"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_quarterly"],
        distractor_doc_ids=["doc_distractor"],
        min_recall_at_1=0.4,
        min_recall_at_3=0.7,
        min_recall_at_5=1.0,
        min_mrr=0.4,
    ),

    # ------------------------------------------------------------------
    # Summary questions (multiple facts from one document)
    # ------------------------------------------------------------------
    EvalCase(
        case_id="SM-001",
        question="What are the key technical specifications of the QuantumEdge QX-500?",
        category="summary",
        expected_facts=["32-core Xeon Platinum 8480+", "512 GB DDR5 ECC", "15.36 TB NVMe SSD"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_specs"],
        distractor_doc_ids=["doc_distractor"],
        min_recall_at_1=0.2,
        min_recall_at_3=0.5,
        min_recall_at_5=0.8,
        min_mrr=0.2,
    ),
    EvalCase(
        case_id="SM-002",
        question="What is the remote work policy at Acme Corporation?",
        category="summary",
        expected_facts=["3 days per week", "manager approval", "10:00 AM to 3:00 PM", "$500"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_policy"],
        distractor_doc_ids=["doc_distractor"],
        min_recall_at_1=0.2,
        min_recall_at_3=0.5,
        min_recall_at_5=0.8,
        min_mrr=0.2,
    ),

    # ------------------------------------------------------------------
    # Multi-part questions
    # ------------------------------------------------------------------
    EvalCase(
        case_id="MP-001",
        question="What are the leave and expense policies? Include annual leave days, sick leave accrual, and meal expense cap.",
        category="multi_part",
        expected_facts=["20 days", "1 day per month", "$75", "per day"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_policy"],
        distractor_doc_ids=["doc_distractor"],
        min_recall_at_1=0.2,
        min_recall_at_3=0.4,
        min_recall_at_5=0.7,
        min_mrr=0.2,
    ),

    # ------------------------------------------------------------------
    # Multi-document questions
    # ------------------------------------------------------------------
    EvalCase(
        case_id="MD-001",
        question="How many employees work in North America in Q2-2025, and what is their profit?",
        category="multi_document",
        expected_facts=["175", "2900000", "Q2-2025", "North America"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_quarterly"],
        distractor_doc_ids=["doc_distractor"],
        min_recall_at_1=0.3,
        min_recall_at_3=0.6,
        min_recall_at_5=0.9,
        min_mrr=0.3,
    ),

    # ------------------------------------------------------------------
    # Missing evidence questions (should abstain)
    # ------------------------------------------------------------------
    EvalCase(
        case_id="ME-001",
        question="What is the company's dress code policy?",
        category="missing_evidence",
        expected_facts=[],
        sufficient_evidence=False,
        relevant_doc_ids=[],
        distractor_doc_ids=["doc_policy", "doc_distractor"],
        min_recall_at_1=0.0,
        min_recall_at_3=0.0,
        min_recall_at_5=0.0,
        min_mrr=0.0,
    ),
    EvalCase(
        case_id="ME-002",
        question="What is the CEO's salary?",
        category="missing_evidence",
        expected_facts=[],
        sufficient_evidence=False,
        relevant_doc_ids=[],
        distractor_doc_ids=["doc_directory", "doc_policy"],
        min_recall_at_1=0.0,
        min_recall_at_3=0.0,
        min_recall_at_5=0.0,
        min_mrr=0.0,
    ),

    # ------------------------------------------------------------------
    # Conflicting facts questions
    # ------------------------------------------------------------------
    EvalCase(
        case_id="CF-001",
        question="How many days of paid annual leave does Acme offer?",
        category="conflicting_facts",
        expected_facts=["20 days", "25 days"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_policy", "doc_policy_v2"],
        distractor_doc_ids=["doc_distractor"],
        min_recall_at_1=0.3,
        min_recall_at_3=0.6,
        min_recall_at_5=1.0,
        min_mrr=0.3,
    ),

    # ------------------------------------------------------------------
    # Distractor resistance questions
    # ------------------------------------------------------------------
    EvalCase(
        case_id="DR-001",
        question="What is the weather forecast for Tuesday?",
        category="distractor",
        expected_facts=["Rain likely", "68F", "60F"],
        sufficient_evidence=True,
        relevant_doc_ids=["doc_distractor"],
        distractor_doc_ids=["doc_policy", "doc_specs"],
        min_recall_at_1=0.0,
        min_recall_at_3=0.0,
        min_recall_at_5=1.0,
        min_mrr=0.2,
    ),
]

# ============================================================================
# METADATA
# ============================================================================

BENCHMARK_METADATA = {
    "name": "RAG-Q&A Synthetic Benchmark",
    "version": "1.0.0",
    "total_documents": len(BENCHMARK_DOCUMENTS),
    "total_cases": len(BENCHMARK_CASES),
    "categories": sorted(set(c.category for c in BENCHMARK_CASES)),
    "description": (
        "Synthetic evaluation dataset for measuring retrieval and generation "
        "quality in the RAG-Q&A document assistant. All documents and questions "
        "are synthetic; no personal or private information is included."
    ),
}

# Category counts for reporting
CATEGORY_COUNTS: Dict[str, int] = {}
for case in BENCHMARK_CASES:
    CATEGORY_COUNTS[case.category] = CATEGORY_COUNTS.get(case.category, 0) + 1

SUFFICIENT_EVIDENCE_COUNT = sum(1 for c in BENCHMARK_CASES if c.sufficient_evidence)
INSUFFICIENT_EVIDENCE_COUNT = sum(1 for c in BENCHMARK_CASES if not c.sufficient_evidence)