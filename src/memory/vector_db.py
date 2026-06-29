"""
Hybrid RAG engine combining dense vector search (ChromaDB) with sparse
BM25 lexical retrieval, fused via Reciprocal Rank Fusion (RRF), and
re-ranked with a cross-encoder model for high-precision results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chromadb
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import (
    BM25_REBUILD_THRESHOLD,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    RAG_TOP_K,
    RELEVANCE_THRESHOLD,
    RERANKER_MODEL,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

RRF_K = 60  # Constant used in Reciprocal Rank Fusion scoring.


@dataclass
class SearchResult:
    """A single search result returned by the hybrid retrieval pipeline."""

    document: str
    score: float
    source: str  # e.g. "hybrid_rrf", "reranker"

    def __repr__(self) -> str:
        return (
            f"SearchResult(score={self.score:.4f}, source={self.source!r}, "
            f"document={self.document[:80]!r}{'…' if len(self.document) > 80 else ''})"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class HybridRAGEngine:
    """Dense + sparse hybrid retrieval with cross-encoder re-ranking.

    Pipeline
    --------
    1. **Dense retrieval** – ChromaDB vector similarity search using a
       SentenceTransformer bi-encoder.
    2. **Sparse retrieval** – BM25 (Okapi) keyword search over the same
       corpus.
    3. **RRF fusion** – Reciprocal Rank Fusion merges the two ranked
       lists into a single candidate set.
    4. **Cross-encoder re-ranking** – A cross-encoder scores each
       (query, candidate) pair for fine-grained relevance.
    5. **Threshold filter** – Only results with a cross-encoder score
       ≥ ``RELEVANCE_THRESHOLD`` are returned.
    """

    def __init__(self) -> None:
        logger.info(
            "Initialising HybridRAGEngine (chroma={}, embedder={}, reranker={})",
            CHROMA_PATH,
            EMBEDDING_MODEL,
            RERANKER_MODEL,
        )

        # -- Vector store --
        self.chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma.get_or_create_collection(
            name="receipt_memory",
        )

        # -- Models --
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # -- BM25 state --
        self.corpus: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._docs_since_rebuild: int = 0

        # Hydrate BM25 from existing ChromaDB documents.
        self._load_existing_corpus()

        logger.info(
            "HybridRAGEngine ready – {} documents in corpus", len(self.corpus)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_existing_corpus(self) -> None:
        """Load all persisted documents into the BM25 index."""
        try:
            results = self.collection.get()
            if results and results["documents"]:
                self.corpus = list(results["documents"])
                self.tokenized_corpus = [doc.split() for doc in self.corpus]
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                self._docs_since_rebuild = 0
                logger.debug(
                    "Loaded {} existing documents into BM25 index",
                    len(self.corpus),
                )
            else:
                logger.debug("No existing documents found in ChromaDB")
        except Exception as e:
            logger.warning("Failed to load existing corpus into BM25: {}", e)
            self.bm25 = None

    def _rebuild_bm25_if_needed(self, *, force: bool = False) -> None:
        """Rebuild the BM25 index when enough new documents have accumulated.

        The index is rebuilt every ``BM25_REBUILD_THRESHOLD`` insertions
        (configurable) to avoid the O(n) cost on every single add.
        """
        if force or self._docs_since_rebuild >= BM25_REBUILD_THRESHOLD:
            if not self.tokenized_corpus:
                self.bm25 = None
                return
            t0 = time.perf_counter()
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                "BM25 index rebuilt ({} docs) in {:.1f} ms",
                len(self.tokenized_corpus),
                elapsed_ms,
            )
            self._docs_since_rebuild = 0

    def _rrf_fuse(
        self,
        ranked_lists: List[List[str]],
        k: int = RRF_K,
    ) -> Dict[str, float]:
        """Merge multiple ranked lists via Reciprocal Rank Fusion.

        Parameters
        ----------
        ranked_lists:
            Each inner list is an ordered sequence of document strings
            (most relevant first).
        k:
            The RRF constant (default 60).

        Returns
        -------
        dict mapping document text → fused RRF score.
        """
        scores: Dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
        return scores

    def _rerank(
        self,
        query: str,
        candidates: List[str],
    ) -> List[SearchResult]:
        """Score candidates with the cross-encoder and sort descending.

        Parameters
        ----------
        query:
            The user query.
        candidates:
            Candidate document texts from the RRF stage.

        Returns
        -------
        Sorted list of ``SearchResult`` with cross-encoder scores.
        """
        if not candidates:
            return []

        pairs = [(query, doc) for doc in candidates]
        t0 = time.perf_counter()
        ce_scores: np.ndarray = self.reranker.predict(pairs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results = [
            SearchResult(document=doc, score=float(score), source="reranker")
            for doc, score in zip(candidates, ce_scores)
        ]
        results.sort(key=lambda r: r.score, reverse=True)

        logger.debug(
            "Cross-encoder re-ranked {} candidates in {:.1f} ms | scores: {}",
            len(results),
            elapsed_ms,
            [f"{r.score:.4f}" for r in results],
        )
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_memory(self, doc_id: str, text: str, metadata: dict) -> None:
        """Persist a document into both ChromaDB and the BM25 corpus.

        Parameters
        ----------
        doc_id:
            Unique identifier for the document.
        text:
            Full text content to store and index.
        metadata:
            Arbitrary metadata dict attached to the ChromaDB record.
        """
        embedding = self.embedder.encode(text).tolist()
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

        self.corpus.append(text)
        self.tokenized_corpus.append(text.split())
        self._docs_since_rebuild += 1

        # Rebuild BM25 only when the threshold is crossed, or when the
        # index doesn't exist yet (first document).
        needs_initial_build = self.bm25 is None
        self._rebuild_bm25_if_needed(force=needs_initial_build)

        logger.info(
            "Added memory '{}' ({} chars) – docs_since_rebuild={}",
            doc_id,
            len(text),
            self._docs_since_rebuild,
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        metadata_filter: Optional[dict] = None,
    ) -> List[SearchResult]:
        """Run the full hybrid retrieval + re-ranking pipeline.

        Parameters
        ----------
        query:
            Natural language query from the user.
        top_k:
            Maximum number of results to return.
        metadata_filter:
            Optional ChromaDB ``where`` filter dict.

        Returns
        -------
        List of ``SearchResult`` sorted by cross-encoder relevance,
        filtered by ``RELEVANCE_THRESHOLD``.  Returns a single
        ``SearchResult`` with a helpful message if nothing relevant is
        found.
        """
        t_start = time.perf_counter()
        logger.info("Hybrid search: query={!r}, top_k={}", query, top_k)

        if self.bm25 is None or len(self.corpus) == 0:
            logger.warning("Search attempted on empty corpus")
            return [
                SearchResult(
                    document="No receipts found in memory.",
                    score=0.0,
                    source="empty_corpus",
                )
            ]

        # --- Stage 1: Dense retrieval (ChromaDB) -------------------------
        query_emb = self.embedder.encode(query).tolist()
        chroma_kwargs: dict = {
            "query_embeddings": [query_emb],
            "n_results": top_k,
        }
        if metadata_filter:
            chroma_kwargs["where"] = metadata_filter

        try:
            vec_docs: List[str] = self.collection.query(**chroma_kwargs)[
                "documents"
            ][0]
        except Exception as e:
            logger.warning(
                "ChromaDB metadata filter failed ({}), retrying without filter",
                e,
            )
            vec_docs = self.collection.query(
                query_embeddings=[query_emb], n_results=top_k
            )["documents"][0]

        # --- Stage 2: Sparse retrieval (BM25) -----------------------------
        bm25_scores = self.bm25.get_scores(query.split())
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_docs: List[str] = [self.corpus[i] for i in top_bm25_indices]

        # --- Stage 3: RRF fusion ------------------------------------------
        rrf_scores = self._rrf_fuse([vec_docs, bm25_docs])
        rrf_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [doc for doc, _ in rrf_ranked[:top_k]]

        logger.debug(
            "RRF fusion produced {} unique candidates from {} dense + {} sparse",
            len(rrf_scores),
            len(vec_docs),
            len(bm25_docs),
        )

        # --- Stage 4: Cross-encoder re-ranking ----------------------------
        reranked = self._rerank(query, candidates)

        # --- Stage 5: Relevance threshold ---------------------------------
        filtered = [r for r in reranked if r.score >= RELEVANCE_THRESHOLD]

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Search complete in {:.1f} ms – {} results above threshold "
            "(threshold={}, pre-filter={})",
            elapsed_ms,
            len(filtered),
            RELEVANCE_THRESHOLD,
            len(reranked),
        )

        if not filtered:
            logger.info(
                "All {} candidates scored below RELEVANCE_THRESHOLD={}",
                len(reranked),
                RELEVANCE_THRESHOLD,
            )
            return [
                SearchResult(
                    document=(
                        "No sufficiently relevant receipts found. "
                        "Try rephrasing your query or broadening the search."
                    ),
                    score=0.0,
                    source="below_threshold",
                )
            ]

        return filtered

    def search_text(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        metadata_filter: Optional[dict] = None,
    ) -> List[str]:
        """Convenience wrapper returning plain document strings.

        This preserves backward-compatibility with callers that expect
        ``List[str]`` from the search method.
        """
        results = self.hybrid_search(
            query, top_k=top_k, metadata_filter=metadata_filter
        )
        return [r.document for r in results]