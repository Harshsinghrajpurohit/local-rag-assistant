"""
Hybrid retriever combining BM25 (sparse) + semantic (dense) search with RRF fusion.
Provides better ranking by combining exact keyword matches with semantic meaning.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from backend.app.services.chroma_service import ChromaService
from backend.app.services.embeddings import EmbeddingsService


class HybridRetriever:
    """Combines BM25 keyword search + dense semantic search using Reciprocal Rank Fusion."""

    def __init__(
        self,
        chroma_service: ChromaService,
        embeddings_service: Optional[EmbeddingsService] = None,
        k_rrf: int = 60,
    ):
        """
        Initialize hybrid retriever.

        Args:
            chroma_service: ChromaService instance for dense semantic search
            embeddings_service: Optional EmbeddingsService (unused but kept for consistency)
            k_rrf: Constant in RRF formula 1/(k+rank), default 60 (empirically robust)
        """
        self.chroma = chroma_service
        self.embeddings_service = embeddings_service
        self.k_rrf = k_rrf

        # BM25 index
        self.bm25_index = None
        self.doc_ids = []  # Mapping rank position to ChromaDB ID
        self.corpus_texts = []  # Original texts for reference

        # Build BM25 index from existing ChromaDB documents
        self._build_bm25_index()

    def _build_bm25_index(self):
        """
        Build BM25 index from all documents in ChromaDB.
        Call on startup and after adding new documents.
        """
        # Fetch all documents from ChromaDB
        all_docs = self.chroma.collection.get(include=["documents"])

        if not all_docs["ids"]:
            # Empty database
            self.bm25_index = None
            self.doc_ids = []
            self.corpus_texts = []
            return

        self.doc_ids = all_docs["ids"]
        self.corpus_texts = all_docs["documents"]

        # Tokenize corpus: simple lowercase + split
        # For future: could use nltk.PorterStemmer or spacy for better stemming
        tokenized_corpus = [self._tokenize(text) for text in self.corpus_texts]

        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase, remove punctuation, and split on whitespace.
        Extensible for future stemming/lemmatization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        import string
        # Remove punctuation and convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        return text.split()

    def _bm25_search(
        self, query_text: str, top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Search using BM25 keyword matching.

        Args:
            query_text: Query string
            top_k: Number of results to return

        Returns:
            List of (document_id, bm25_score) tuples, sorted by score descending.
            Documents with a non-positive BM25 score are excluded.
        """
        if self.bm25_index is None or not self.doc_ids:
            return []

        # Tokenize query the same way as corpus
        query_tokens = self._tokenize(query_text)

        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(query_tokens)

        # Get indices of top-k highest scores
        top_indices = np.argsort(scores)[-top_k:][::-1]

        # Only documents with a positive BM25 score carry keyword signal.
        # Zero/negative scores (no matching terms, or non-positive idf in
        # tiny corpora) are excluded so they cannot pollute RRF fusion or
        # report meaningless scores to callers.
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append((self.doc_ids[idx], score))

        return results

    def _filter_bm25_by_where(
        self, bm25_results: List[Tuple[str, float]], where: Dict
    ) -> List[Tuple[str, float]]:
        """
        Keep only BM25 candidates whose IDs match a ChromaDB metadata filter.

        Delegates where-clause evaluation to ChromaDB itself, so the BM25 leg
        and the dense leg apply identical filter semantics. Reads the existing
        chunk metadata only - no schema change and no persistent-data writes.

        Args:
            bm25_results: (document_id, bm25_score) tuples from _bm25_search
            where: ChromaDB where filter (e.g. {"file_name": {"$eq": "..."}})

        Returns:
            Filtered (document_id, bm25_score) tuples in original BM25 order
        """
        matching_ids = set(
            self.chroma.collection.get(where=where, include=["metadatas"])["ids"]
        )
        return [
            (doc_id, score)
            for doc_id, score in bm25_results
            if doc_id in matching_ids
        ]

    def _dense_search(
        self, query_text: str, top_k: int = 20, where: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Search using dense semantic embeddings (ChromaDB).

        Args:
            query_text: Query string
            top_k: Number of results to return
            where: Optional metadata filter for ChromaDB

        Returns:
            List of (document_id, similarity_score) tuples, sorted by similarity descending
        """
        results = self.chroma.query(query_text, n_results=top_k, where=where)

        # ChromaDB returns nested lists and distances (lower = more similar)
        # Convert distances to similarities (negate) and pair with IDs
        if not results["ids"] or not results["ids"][0]:
            return []

        return [
            (doc_id, -distance)
            for doc_id, distance in zip(results["ids"][0], results["distances"][0])
        ]

    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion: combine two ranked lists into one.

        Formula: RRF_score(doc) = sum(1 / (k + rank) for each ranker)

        Args:
            bm25_results: List of (doc_id, score) from BM25
            dense_results: List of (doc_id, score) from dense search
            k: Constant in RRF formula (default 60, empirically robust)

        Returns:
            List of (doc_id, rrf_score) sorted by RRF score descending.
            Ties (equal RRF scores) are ordered deterministically by:
              (1) dense-search rank, best (lowest) first;
              (2) BM25 rank, best (lowest) first;
              (3) doc_id lexicographically (final stability guarantee).
            RRF score remains the primary sort key, so relevance ordering
            is preserved and dict insertion order no longer affects results.
        """
        rrf_scores: Dict[str, float] = {}
        bm25_ranks: Dict[str, int] = {}
        dense_ranks: Dict[str, int] = {}

        # BM25 contributions
        for rank, (doc_id, _) in enumerate(bm25_results, 1):
            bm25_ranks[doc_id] = rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # Dense contributions (deduplicated)
        for rank, (doc_id, _) in enumerate(dense_results, 1):
            dense_ranks[doc_id] = rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # Deterministic ordering: RRF score descending (primary), then
        # dense rank, then BM25 rank, then doc_id as the final tie-break.
        def _sort_key(
            item: Tuple[str, float]
        ) -> Tuple[float, float, float, str]:
            doc_id, score = item
            return (
                -score,
                dense_ranks.get(doc_id, float("inf")),
                bm25_ranks.get(doc_id, float("inf")),
                doc_id,
            )

        return sorted(rrf_scores.items(), key=_sort_key)

    def retrieve(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Hybrid retrieval: combine BM25 and dense search with RRF fusion.

        Args:
            query_text: User query
            n_results: Final number of results to return
            where: Optional metadata filter (applied to both the dense
                search and the BM25 candidates before fusion)

        Returns:
            Dict with keys:
                - ids: Document IDs
                - documents: Document texts
                - metadatas: Metadata for each document
                - distances: Semantic distances (cosine)
                - bm25_scores: BM25 relevance scores
                - rrf_scores: Final RRF fusion scores

        Example:
            results = retriever.retrieve("What is RAG?", n_results=5)
            for doc, meta, score in zip(results["documents"], results["metadatas"], results["rrf_scores"]):
                print(f"[{score:.4f}] {doc[:100]}... (page {meta['page']})")
        """
        # Retrieve more before fusion to reduce ranking noise
        fetch_k = max(n_results * 2, 20)

        # Parallel-style fetching (sequential for now, could be async)
        bm25_results = self._bm25_search(query_text, top_k=fetch_k)
        if where is not None:
            # Apply the same metadata filter to the BM25 leg as to the
            # dense leg, so filtered-out documents cannot re-enter via fusion.
            bm25_results = self._filter_bm25_by_where(bm25_results, where)
        dense_results = self._dense_search(query_text, top_k=fetch_k, where=where)

        # Fuse rankings using RRF
        fused_results = self._rrf_fusion(bm25_results, dense_results, k=self.k_rrf)

        # Get top-n final results
        top_doc_ids = [doc_id for doc_id, _ in fused_results[:n_results]]

        if not top_doc_ids:
            # No results
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
                "bm25_scores": [],
                "rrf_scores": [],
            }

        # Fetch full documents and metadata from ChromaDB
        batch_results = self.chroma.collection.get(
            ids=top_doc_ids,
            include=["documents", "metadatas"],
        )

        # Build response in fused order
        id_to_batch_idx = {doc_id: i for i, doc_id in enumerate(batch_results["ids"])}

        response = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
            "bm25_scores": [],
            "rrf_scores": [],
        }

        # Create mapping of doc_id to scores from both rankers
        bm25_score_map = {doc_id: score for doc_id, score in bm25_results}
        dense_score_map = {doc_id: score for doc_id, score in dense_results}
        rrf_score_map = {doc_id: score for doc_id, score in fused_results}

        for doc_id, rrf_score in fused_results[:n_results]:
            if doc_id in id_to_batch_idx:
                idx = id_to_batch_idx[doc_id]

                response["ids"].append(doc_id)
                response["documents"].append(batch_results["documents"][idx])
                response["metadatas"].append(batch_results["metadatas"][idx])
                # Use negative of dense similarity (since we converted distances to similarities)
                response["distances"].append(dense_score_map.get(doc_id, 1.0))
                response["bm25_scores"].append(bm25_score_map.get(doc_id, 0.0))
                response["rrf_scores"].append(rrf_score)

        return response

    def rebuild_index(self):
        """
        Rebuild BM25 index from ChromaDB.
        Call this after adding new documents via ChromaService.add_chunks().
        """
        self._build_bm25_index()

    def get_index_stats(self) -> Dict[str, Any]:
        """Return statistics about the hybrid index."""
        return {
            "bm25_corpus_size": len(self.doc_ids) if self.bm25_index else 0,
            "k_rrf": self.k_rrf,
            "index_built": self.bm25_index is not None,
        }
