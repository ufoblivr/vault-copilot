"""
RAG Evaluation Runner.

Measures recall@k, MRR, and precision across 4 retrieval strategies:
1. Vector-only (ChromaDB dense embeddings)
2. BM25-only (sparse keyword matching)
3. Hybrid (RRF fusion of dense + sparse)
4. Hybrid + Cross-Encoder Re-ranker

Usage:
    python -m eval.run_eval
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k retrieved."""
    if not relevant_ids:
        return 1.0  # vacuously true
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of top-k retrieved that are relevant."""
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / k


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of the first relevant result."""
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# ID-matching helper
# ---------------------------------------------------------------------------

def _match_doc_id(document_text: str, corpus: List[dict]) -> str:
    """Find the corpus ID that matches a retrieved document text."""
    for entry in corpus:
        if entry["text"] == document_text:
            return entry["id"]
    # Fuzzy fallback: check if the document is a substring or vice versa
    for entry in corpus:
        if entry["text"] in document_text or document_text in entry["text"]:
            return entry["id"]
    return "unknown"


# ---------------------------------------------------------------------------
# Evaluation strategies
# ---------------------------------------------------------------------------

def _evaluate_vector_only(engine, queries, corpus, top_k: int = 5) -> Dict:
    """Evaluate using only ChromaDB dense vector search."""
    from sentence_transformers import SentenceTransformer
    all_recall3, all_recall5, all_mrr, all_prec3 = [], [], [], []

    embedder = engine.embedder

    for entry in queries:
        query_emb = embedder.encode(entry.query).tolist()
        try:
            results = engine.collection.query(
                query_embeddings=[query_emb], n_results=top_k
            )
            docs = results["documents"][0]
        except Exception:
            docs = []

        retrieved_ids = [_match_doc_id(d, corpus) for d in docs]
        all_recall3.append(recall_at_k(retrieved_ids, entry.relevant_doc_ids, 3))
        all_recall5.append(recall_at_k(retrieved_ids, entry.relevant_doc_ids, 5))
        all_mrr.append(mrr(retrieved_ids, entry.relevant_doc_ids))
        all_prec3.append(precision_at_k(retrieved_ids, entry.relevant_doc_ids, 3))

    return {
        "recall@3": float(np.mean(all_recall3)),
        "recall@5": float(np.mean(all_recall5)),
        "mrr": float(np.mean(all_mrr)),
        "precision@3": float(np.mean(all_prec3)),
    }


def _evaluate_bm25_only(engine, queries, corpus, top_k: int = 5) -> Dict:
    """Evaluate using only BM25 sparse keyword search."""
    all_recall3, all_recall5, all_mrr, all_prec3 = [], [], [], []

    if engine.bm25 is None:
        return {"recall@3": 0, "recall@5": 0, "mrr": 0, "precision@3": 0}

    for entry in queries:
        scores = engine.bm25.get_scores(entry.query.split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        docs = [engine.corpus[i] for i in top_indices]
        retrieved_ids = [_match_doc_id(d, corpus) for d in docs]

        all_recall3.append(recall_at_k(retrieved_ids, entry.relevant_doc_ids, 3))
        all_recall5.append(recall_at_k(retrieved_ids, entry.relevant_doc_ids, 5))
        all_mrr.append(mrr(retrieved_ids, entry.relevant_doc_ids))
        all_prec3.append(precision_at_k(retrieved_ids, entry.relevant_doc_ids, 3))

    return {
        "recall@3": float(np.mean(all_recall3)),
        "recall@5": float(np.mean(all_recall5)),
        "mrr": float(np.mean(all_mrr)),
        "precision@3": float(np.mean(all_prec3)),
    }


def _evaluate_hybrid(engine, queries, corpus, top_k: int = 5, use_reranker: bool = True) -> Dict:
    """Evaluate using the full hybrid pipeline (optionally with re-ranker)."""
    all_recall3, all_recall5, all_mrr, all_prec3 = [], [], [], []

    # Temporarily adjust reranker behavior if needed
    original_threshold = None
    if not use_reranker:
        # To skip re-ranker, we set threshold very low
        import src.memory.vector_db as vdb
        original_threshold = vdb.RELEVANCE_THRESHOLD
        vdb.RELEVANCE_THRESHOLD = -999.0

    try:
        for entry in queries:
            results = engine.hybrid_search(entry.query, top_k=top_k)
            docs = [r.document for r in results]
            retrieved_ids = [_match_doc_id(d, corpus) for d in docs]

            all_recall3.append(recall_at_k(retrieved_ids, entry.relevant_doc_ids, 3))
            all_recall5.append(recall_at_k(retrieved_ids, entry.relevant_doc_ids, 5))
            all_mrr.append(mrr(retrieved_ids, entry.relevant_doc_ids))
            all_prec3.append(precision_at_k(retrieved_ids, entry.relevant_doc_ids, 3))
    finally:
        if original_threshold is not None:
            import src.memory.vector_db as vdb
            vdb.RELEVANCE_THRESHOLD = original_threshold

    return {
        "recall@3": float(np.mean(all_recall3)),
        "recall@5": float(np.mean(all_recall5)),
        "mrr": float(np.mean(all_mrr)),
        "precision@3": float(np.mean(all_prec3)),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evaluation():
    """Execute the full RAG evaluation pipeline."""
    from eval.rag_benchmark import SYNTHETIC_CORPUS, BENCHMARK_QUERIES

    # Setup: use a temp directory for ChromaDB
    tmp_chroma = tempfile.mkdtemp(prefix="vault_eval_chroma_")

    # Patch config before importing the engine
    import src.config as cfg
    original_chroma = cfg.CHROMA_PATH
    cfg.CHROMA_PATH = tmp_chroma

    # Also set a low threshold for evaluation so we get all results
    original_threshold = cfg.RELEVANCE_THRESHOLD
    cfg.RELEVANCE_THRESHOLD = -10.0  # Accept everything during eval

    try:
        from src.memory.vector_db import HybridRAGEngine

        print("=" * 60)
        print("  RAG Evaluation Runner — Vault Copilot")
        print("=" * 60)
        print()

        # 1. Build the index
        print(f"Loading corpus: {len(SYNTHETIC_CORPUS)} documents...")
        # Need to patch module-level imports too
        import src.memory.vector_db as vdb
        vdb.CHROMA_PATH = tmp_chroma
        vdb.BM25_REBUILD_THRESHOLD = 1  # Rebuild after every doc for eval

        engine = HybridRAGEngine()
        for doc in SYNTHETIC_CORPUS:
            engine.add_memory(doc["id"], doc["text"], doc["metadata"])

        print(f"Index built: {len(engine.corpus)} documents, BM25 ready: {engine.bm25 is not None}")
        print(f"Queries: {len(BENCHMARK_QUERIES)}")
        print()

        # 2. Run evaluations
        results = {}

        print("Evaluating: Vector-only (ChromaDB)...")
        t0 = time.perf_counter()
        results["Vector-only"] = _evaluate_vector_only(engine, BENCHMARK_QUERIES, SYNTHETIC_CORPUS)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

        print("Evaluating: BM25-only...")
        t0 = time.perf_counter()
        results["BM25-only"] = _evaluate_bm25_only(engine, BENCHMARK_QUERIES, SYNTHETIC_CORPUS)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

        print("Evaluating: Hybrid (RRF) without re-ranker...")
        t0 = time.perf_counter()
        # For hybrid without reranker, set threshold very low
        vdb.RELEVANCE_THRESHOLD = -999.0
        results["Hybrid (RRF)"] = _evaluate_hybrid(engine, BENCHMARK_QUERIES, SYNTHETIC_CORPUS, use_reranker=True)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

        print("Evaluating: Hybrid + Cross-Encoder Re-ranker...")
        vdb.RELEVANCE_THRESHOLD = -10.0  # Still accept everything but now with real reranking
        t0 = time.perf_counter()
        results["Hybrid + Re-ranker"] = _evaluate_hybrid(engine, BENCHMARK_QUERIES, SYNTHETIC_CORPUS, use_reranker=True)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

        # 3. Print results table
        print()
        print("=" * 70)
        print(f"{'Strategy':<25} {'Recall@3':>10} {'Recall@5':>10} {'MRR':>10} {'Prec@3':>10}")
        print("-" * 70)
        for strategy, metrics in results.items():
            print(
                f"{strategy:<25} "
                f"{metrics['recall@3']:>10.4f} "
                f"{metrics['recall@5']:>10.4f} "
                f"{metrics['mrr']:>10.4f} "
                f"{metrics['precision@3']:>10.4f}"
            )
        print("=" * 70)

        # 4. Save results
        eval_dir = Path(__file__).parent
        results_path = eval_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        # 5. Per-category breakdown
        print("\n--- Per-Category Breakdown (Hybrid + Re-ranker) ---")
        categories = {}
        for entry in BENCHMARK_QUERIES:
            if entry.category not in categories:
                categories[entry.category] = {"queries": [], "recall3": []}
            r = engine.hybrid_search(entry.query, top_k=5)
            docs = [ri.document for ri in r]
            retrieved_ids = [_match_doc_id(d, SYNTHETIC_CORPUS) for d in docs]
            r3 = recall_at_k(retrieved_ids, entry.relevant_doc_ids, 3)
            categories[entry.category]["recall3"].append(r3)

        for cat, data in sorted(categories.items()):
            avg_r3 = np.mean(data["recall3"])
            print(f"  {cat:<15}: Recall@3 = {avg_r3:.4f} ({len(data['recall3'])} queries)")

    finally:
        # Cleanup
        cfg.CHROMA_PATH = original_chroma
        cfg.RELEVANCE_THRESHOLD = original_threshold
        if os.path.exists(tmp_chroma):
            shutil.rmtree(tmp_chroma, ignore_errors=True)


if __name__ == "__main__":
    run_evaluation()
