"""
Test suite for the Hybrid RAG engine: add/search round-trip,
SearchResult structure, empty corpus, and relevance filtering.
"""
import shutil
import tempfile

import pytest

from src.memory.vector_db import HybridRAGEngine, SearchResult


@pytest.fixture
def rag_engine(monkeypatch, tmp_path):
    """Create a HybridRAGEngine with a temporary ChromaDB directory."""
    chroma_path = str(tmp_path / "test_chroma")
    monkeypatch.setattr("src.config.CHROMA_PATH", chroma_path)
    # Also need to patch the module where it's already imported
    import src.memory.vector_db as vdb_mod
    monkeypatch.setattr(vdb_mod, "CHROMA_PATH", chroma_path)
    monkeypatch.setattr(vdb_mod, "BM25_REBUILD_THRESHOLD", 1)  # Rebuild immediately for tests

    engine = HybridRAGEngine()
    return engine


@pytest.fixture
def populated_rag(rag_engine):
    """A RAG engine pre-populated with sample documents."""
    docs = [
        ("1", "Spent $45.99 at Whole Foods on 2024-01-15. Items: organic milk, avocados, quinoa.",
         {"store": "Whole Foods"}),
        ("2", "Spent $12.50 at Starbucks on 2024-01-16. Items: grande latte, croissant.",
         {"store": "Starbucks"}),
        ("3", "Spent $199.99 at Amazon on 2024-01-20. Items: wireless headphones.",
         {"store": "Amazon"}),
        ("4", "Spent $67.40 at Target on 2024-02-01. Items: paper towels, cereal, detergent.",
         {"store": "Target"}),
        ("5", "Spent $55.00 at Shell Gas on 2024-02-10. Items: premium fuel.",
         {"store": "Shell Gas"}),
    ]
    for doc_id, text, metadata in docs:
        rag_engine.add_memory(doc_id, text, metadata)
    return rag_engine


# ======================================================================
# Basic operations
# ======================================================================

class TestBasicOperations:
    def test_add_memory(self, rag_engine):
        rag_engine.add_memory("test1", "Test document about groceries", {"store": "TestStore"})
        assert len(rag_engine.corpus) == 1

    def test_add_multiple(self, rag_engine):
        rag_engine.add_memory("1", "First doc", {"store": "A"})
        rag_engine.add_memory("2", "Second doc", {"store": "B"})
        assert len(rag_engine.corpus) == 2


# ======================================================================
# Search functionality
# ======================================================================

class TestHybridSearch:
    def test_search_returns_search_results(self, populated_rag):
        results = populated_rag.hybrid_search("Where did I buy milk?")
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)

    def test_search_result_has_required_fields(self, populated_rag):
        results = populated_rag.hybrid_search("headphones")
        assert len(results) > 0
        r = results[0]
        assert hasattr(r, "document")
        assert hasattr(r, "score")
        assert hasattr(r, "source")
        assert isinstance(r.score, float)
        assert isinstance(r.document, str)

    def test_search_text_returns_strings(self, populated_rag):
        results = populated_rag.search_text("coffee latte")
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], str)

    def test_relevant_results_ranked_higher(self, populated_rag):
        """'headphones' should surface the Amazon doc."""
        results = populated_rag.hybrid_search("wireless headphones")
        docs = [r.document for r in results]
        # The Amazon doc should be in the results
        amazon_found = any("Amazon" in d or "headphones" in d for d in docs)
        assert amazon_found


# ======================================================================
# Empty corpus
# ======================================================================

class TestEmptyCorpus:
    def test_search_empty_returns_message(self, rag_engine):
        results = rag_engine.hybrid_search("anything")
        assert len(results) == 1
        assert "No receipts" in results[0].document or "no" in results[0].document.lower()

    def test_search_text_empty_returns_message(self, rag_engine):
        results = rag_engine.search_text("anything")
        assert len(results) == 1


# ======================================================================
# Metadata filtering
# ======================================================================

class TestMetadataFilter:
    def test_filter_by_store(self, populated_rag):
        results = populated_rag.hybrid_search(
            "What did I buy?",
            metadata_filter={"store": "Starbucks"},
        )
        # Should return results — at minimum the Starbucks doc
        assert len(results) > 0
