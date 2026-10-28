"""
Answer generator service using Ollama LLM.
Generates grounded answers from retrieved passages with source citations.
"""

from typing import Dict, List, Any, Optional
import requests
import json
import logging
import string
from datetime import datetime, timezone

from backend.app.config import Config
from backend.app.exceptions import OllamaUnavailableError

logger = logging.getLogger(__name__)

# Threshold below which retrieved evidence is considered insufficient
# and the system should proactively abstain rather than generate an answer.
# Fraction of query terms that must appear in retrieved documents.
EVIDENCE_SUFFICIENCY_THRESHOLD = 0.25

# Punctuation extension: string.punctuation does NOT include ₹ (rupee sign).
# Add it so ₹40,000 → 40000 in both query and document normalization.
_PUNCTUATION_EXTRA = string.punctuation + "₹"

# Common English function words (stopwords) used instead of a length filter
# so short technical tokens (c, r, ai, ml, js, ci, cd) are preserved.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "were", "will", "with", "i", "you", "we",
    "they", "this", "that", "these", "those", "am",
}


class AnswerGenerator:
    """Generates answers grounded in retrieved documents using Ollama."""

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
    ):
        """
        Initialize answer generator.

        Args:
            base_url: Ollama API base URL (default: from config)
            model: LLM model name (default: Config.LLM_MODEL)
            max_tokens: Maximum tokens for answer generation (default: Config.LLM_MAX_TOKENS)
            temperature: Temperature for LLM sampling (default: Config.LLM_TEMPERATURE)
        """
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.model = model or Config.LLM_MODEL
        self.generate_url = f"{self.base_url}/api/generate"
        self.max_tokens = max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS
        self.temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE

    def generate(
        self,
        query: str,
        retriever_results: Dict[str, Any],
        max_sources: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate answer from retrieved passages.

        Args:
            query: User question
            retriever_results: Output from HybridRetriever.retrieve()
                Can have either nested structure:
                {
                    "ids": [...],
                    "documents": [...],
                    "metadatas": [...],
                    "scores": {"bm25": [...], "semantic": [...], "rrf": [...]}
                }
                OR flat structure (actual output from HybridRetriever):
                {
                    "ids": [...],
                    "documents": [...],
                    "metadatas": [...],
                    "distances": [...],
                    "bm25_scores": [...],
                    "rrf_scores": [...]
                }
            max_sources: Maximum number of source passages to include (default: 5)

        Returns:
            {
                "answer": "Generated answer text",
                "sources": [
                    {
                        "page": 1,  # PDFs only; None for page-less formats
                        "file": "document.pdf",
                        "excerpt": "Relevant passage from document",
                        "score": 0.85,
                        "location": null  # e.g. 'Sheet: Sales, rows 2-41' for non-PDF
                    },
                    ...
                ],
                "confidence": 0.85,
                "tokens_used": 150,
                "model": "qwen3:4b-instruct",
                "timestamp": "2026-10-13T09:06:33Z"
            }

        Raises:
            ValueError: If retriever_results is empty or malformed
            requests.RequestException: If Ollama API call fails
        """
        # Validate retriever results
        if not retriever_results or not retriever_results.get("documents"):
            return self._insufficient_context_response(query)

        # Limit to max_sources
        num_sources = min(len(retriever_results["documents"]), max_sources)
        documents = retriever_results["documents"][:num_sources]
        metadatas = retriever_results["metadatas"][:num_sources]

        # Handle both nested and flat score structures
        rrf_scores = self._extract_rrf_scores(retriever_results, num_sources)

        # Build context from passages
        context = self._format_context(documents, metadatas)

        # Extract sources from passages (independent of evidence sufficiency).
        # Source extraction happens before the abstention check so that
        # source truncation can be verified independently of LLM generation
        # and answer abstention. The answer_text is not yet available here;
        # _extract_sources accepts it but does not require it for the excerpt.
        sources = self._extract_sources(documents, metadatas, rrf_scores, "")

        # --- Proactive abstention: check evidence sufficiency before calling LLM ---
        # Abstain when the fraction of query terms found in retrieved documents
        # falls below EVIDENCE_SUFFICIENCY_THRESHOLD. Queries with partial but
        # insufficient overlap proceed to generation with proportionally low
        # confidence (retrieval_signal * evidence_sufficiency).
        evidence_sufficiency = self._compute_evidence_sufficiency(documents, query)
        if (
            evidence_sufficiency < EVIDENCE_SUFFICIENCY_THRESHOLD
            and query
            and self._tokenize_query(query)
        ):
            return self._insufficient_evidence_response(
                query, evidence_sufficiency, sources
            )

        # Generate answer
        prompt = self._build_prompt(query, context)
        answer_text, tokens_used = self._call_ollama(prompt)

        # Re-extract sources with actual answer text for relevance checking
        sources = self._extract_sources(
            documents, metadatas, rrf_scores, answer_text
        )

        # Calculate confidence from RRF scores + evidence sufficiency
        confidence = self._calculate_confidence(rrf_scores, documents, query)

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": confidence,
            "tokens_used": tokens_used,
            "model": self.model,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    def _extract_rrf_scores(
        self, retriever_results: Dict[str, Any], num_sources: int
    ) -> List[float]:
        """
        Extract RRF scores from retriever results.
        Handles both nested ('scores': {'rrf': [...]}) and flat ('rrf_scores': [...]) formats.
        """
        # Try nested format first
        scores_dict = retriever_results.get("scores", {})
        if isinstance(scores_dict, dict) and "rrf" in scores_dict:
            return scores_dict["rrf"][:num_sources]

        # Fall back to flat format
        if "rrf_scores" in retriever_results:
            return retriever_results["rrf_scores"][:num_sources]

        # No RRF scores found, return empty
        return []

    def _insufficient_context_response(self, query: str) -> Dict[str, Any]:
        """Return structured response when no passages retrieved."""
        return {
            "answer": (
                "I don't have enough context to answer this question. "
                "Please upload relevant documents or rephrase your question."
            ),
            "sources": [],
            "confidence": 0.0,
            "tokens_used": 0,
            "model": self.model,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    def _insufficient_evidence_response(
        self, query: str, evidence_sufficiency: float, sources: list = None
    ) -> Dict[str, Any]:
        """Return structured response when retrieved docs exist but none are relevant.

        Used when documents are retrieved but evidence_sufficiency is 0.0,
        indicating the retrieved content does not address the query at all.

        Args:
            query: Original user query
            evidence_sufficiency: Computed evidence sufficiency score (0.0)
            sources: Retrieved source citations (extracted before abstention)
        """
        return {
            "answer": (
                "I don't have enough information to answer this question based on "
                "the provided sources. The retrieved documents do not contain "
                "relevant evidence for: " + query
            ),
            "sources": sources if sources is not None else [],
            "confidence": 0.0,
            "tokens_used": 0,
            "model": self.model,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    def _format_context(
        self, documents: List[str], metadatas: List[Dict[str, Any]]
    ) -> str:
        """
        Format retrieved passages as context for LLM.

        Args:
            documents: List of passage texts
            metadatas: List of metadata dictionaries

        Returns:
            Formatted context string with source attribution
        """
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(documents, metadatas), 1):
            file_name = metadata.get("file_name", "unknown")
            page = metadata.get("page")
            location = self._format_location(metadata)
            if page is not None:
                loc_str = f"Page {page}"
            elif location:
                loc_str = location
            else:
                loc_str = "location not recorded"
            context_parts.append(f"Source {i} ({loc_str}, {file_name}):\n{doc}")

        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build prompt for LLM.

        Uses system role to enforce grounding in passages.

        Args:
            query: User question
            context: Formatted passage context

        Returns:
            Complete prompt string
        """
        system_prompt = (
            "You are a helpful AI assistant that answers questions based on provided documents. "
            "Answer only using information from the provided sources. "
            "If the answer is not in the sources, say you don't have enough information. "
            "Be concise and accurate. Do not make up or infer information not explicitly stated. "
            "If the provided sources contain conflicting information, explicitly identify "
            "the conflicting values and attribute them to their respective sources. "
            "Do not arbitrarily prefer one version or silently merge conflicts."
        )

        user_message = f"""Based on the following documents:

{context}

Please answer this question: {query}

Provide a clear, concise answer based only on the provided sources."""

        # Format for Ollama
        prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"

        return prompt

    def _call_ollama(self, prompt: str) -> tuple[str, int]:
        """
        Call Ollama API to generate answer.

        Args:
            prompt: Complete prompt string

        Returns:
            (generated_answer, tokens_used)

        Raises:
            OllamaUnavailableError: If Ollama is unreachable, times out, or
                returns a 5xx response (mapped to HTTP 503)
            RuntimeError: For any other API failure (e.g. HTTP 4xx)
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Ollama only honors generation parameters inside "options"
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            response = requests.post(self.generate_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            answer = result.get("response", "").strip()
            tokens = result.get("eval_count", 0) + result.get("prompt_eval_count", 0)

            return answer, tokens

        except requests.exceptions.ConnectionError as e:
            logger.warning("Ollama unreachable at %s: %s", self.base_url, e)
            raise OllamaUnavailableError() from e
        except requests.exceptions.Timeout as e:
            logger.warning("Ollama request timed out: %s", e)
            raise OllamaUnavailableError() from e
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and status >= 500:
                logger.warning("Ollama returned server error %s: %s", status, e)
                raise OllamaUnavailableError() from e
            raise RuntimeError(f"Ollama API error: {str(e)}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {str(e)}") from e

    def _extract_sources(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        scores: List[float],
        answer_text: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract source citations from retrieved passages.

        Args:
            documents: List of passage texts
            metadatas: List of metadata dictionaries
            scores: List of RRF scores
            answer_text: Generated answer (for relevance checking)

        Returns:
            List of source dictionaries with page, file, excerpt, score
        """
        sources = []
        for doc, metadata, score in zip(documents, metadatas, scores):
            # Limit excerpt to first 150 characters
            excerpt = doc[:150] + "..." if len(doc) > 150 else doc

            source = {
                # Never fabricate page numbers for page-less formats
                "page": metadata.get("page"),
                "file": metadata.get("file_name", "unknown"),
                "excerpt": excerpt,
                "score": float(score) if score else 0.0,
                "location": self._format_location(metadata),
            }
            sources.append(source)

        return sources

    @staticmethod
    def _format_location(metadata: Dict[str, Any]) -> Optional[str]:
        """
        Build a human-readable source location from chunk metadata.

        Uses only metadata keys that actually exist (never fabricates):
        sheet name, section heading path, and/or row range. Returns None
        when no location metadata is present (PDFs cite page instead).
        """
        parts = []
        if metadata.get("sheet"):
            parts.append(f"Sheet: {metadata['sheet']}")
        if metadata.get("section"):
            parts.append(f"Section: {metadata['section']}")
        if metadata.get("rows"):
            parts.append(f"rows {metadata['rows']}")
        return ", ".join(parts) if parts else None

    @staticmethod
    def _strip_plural_suffix(term: str) -> str:
        """Conservatively strip common English plural suffixes.

        Only strips when the remaining stem is still meaningful (> 2 chars),
        avoiding over-stemming of words that naturally end in 's' or 'es'
        (e.g., 'bus', 'class', 'gas').
        """
        for suffix in ("es", "s"):
            if term.endswith(suffix):
                candidate = term[: -len(suffix)]
                if len(candidate) > 2:
                    return candidate
        return term

    def _compute_evidence_sufficiency(
        self, documents: Optional[List[str]], query: Optional[str]
    ) -> float:
        """Compute fraction of query terms found in retrieved document text.

        Uses word-boundary matching (document split into a word set) plus
        conservative singular/plural normalization so that e.g. "projects"
        in the query can match "Project" in the documents.

        Returns 1.0 when no query or documents are provided (no signal).
        Returns 0.0 when query terms exist but none appear in the documents.
        """
        if not documents or not query:
            return 1.0
        query_terms = self._tokenize_query(query)
        if not query_terms:
            return 1.0
        doc_text = " ".join(documents).lower()
        doc_text = doc_text.translate(str.maketrans("", "", _PUNCTUATION_EXTRA))
        doc_words = set(t for t in doc_text.split() if t not in STOPWORDS)
        stemmed_queries = [self._strip_plural_suffix(t) for t in query_terms]
        stemmed_doc_words = {self._strip_plural_suffix(w) for w in doc_words}
        matched_terms = sum(1 for term in stemmed_queries if term in stemmed_doc_words)
        return matched_terms / len(query_terms)

    def _calculate_confidence(
        self,
        scores: List[float],
        documents: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> float:
        """
        Calculate calibrated confidence from RRF scores and evidence sufficiency.

        Confidence combines two signals:
          1. Retrieval signal — average RRF score calibrated to 0.0-1.0.
          2. Evidence sufficiency — fraction of query terms found in retrieved docs.

        Final confidence = retrieval_signal * evidence_sufficiency.

        There is no artificial floor: when evidence_sufficiency is 0 (no query
        terms found in retrieved documents), confidence is 0.0 regardless of
        retrieval signal strength. This prevents assigning nonzero confidence
        to answers generated from irrelevant retrieved documents.

        This ensures:
          - No retrieved docs → confidence = 0.0
          - Retrieved but irrelevant docs → evidence_sufficiency ≈ 0 → confidence ≈ 0
          - Well-matched docs with strong RRF scores → high confidence

        Args:
            scores: List of RRF scores from retriever
            documents: Retrieved document texts (for evidence sufficiency check)
            query: Original user query (for evidence sufficiency check)

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not scores:
            return 0.0

        # Signal 1: RRF score strength (0.0-1.0).
        # RRF scores are typically 0.0-0.1; calibrate so that RRF ≈ 0.03 → ~0.6,
        # RRF ≈ 0.06 → ~0.9. Empirically derived from production score ranges.
        avg_rrf = sum(scores) / len(scores)
        retrieval_signal = min(avg_rrf * 19, 1.0)

        # Signal 2: Evidence sufficiency — do retrieved docs contain query terms?
        # Uses the same word-boundary + suffix-stripping normalization as
        # _compute_evidence_sufficiency for consistency.
        evidence_sufficiency = 1.0  # default when no query/documents provided
        if documents and query:
            query_terms = self._tokenize_query(query)
            if query_terms:
                doc_text = " ".join(documents).lower()
                doc_text = doc_text.translate(str.maketrans("", "", _PUNCTUATION_EXTRA))
                doc_words = set(t for t in doc_text.split() if t not in STOPWORDS)
                stemmed_queries = [
                    self._strip_plural_suffix(t) for t in query_terms
                ]
                stemmed_doc_words = {self._strip_plural_suffix(w) for w in doc_words}
                matched_terms = sum(
                    1 for term in stemmed_queries if term in stemmed_doc_words
                )
                evidence_sufficiency = matched_terms / len(query_terms)

        # Combined confidence: retrieval signal discounted by evidence sufficiency.
        # No artificial floor: confidence approaches 0 when evidence is missing.
        confidence = retrieval_signal * evidence_sufficiency

        return float(confidence)

    @staticmethod
    def _tokenize_query(query: str) -> List[str]:
        """Simple tokenization for evidence sufficiency check.

        Strips punctuation (including ₹) and filters stopwords so that
        short technical tokens (c, r, ai, ml, js, ci, cd) are preserved.
        """
        text = query.translate(str.maketrans("", "", _PUNCTUATION_EXTRA)).lower()
        return [t for t in text.split() if t not in STOPWORDS]
