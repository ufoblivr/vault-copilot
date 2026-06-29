"""
Test suite for SQL safety: AST-based validation, read-only execution,
row limits, and receipt insertion.
"""
import sqlite3

import pytest

from src.memory.sqlite_db import (
    validate_sql,
    execute_sql,
    insert_receipt,
    check_duplicate_hash,
    insert_receipt_hash,
)


# ======================================================================
# validate_sql — Valid queries that MUST pass
# ======================================================================

class TestValidSQLQueries:
    """Queries that should be accepted by the AST validator."""

    def test_simple_select(self):
        ok, _ = validate_sql("SELECT * FROM receipts")
        assert ok

    def test_select_with_where(self):
        ok, _ = validate_sql("SELECT store, total FROM receipts WHERE total > 10")
        assert ok

    def test_select_with_like(self):
        ok, _ = validate_sql("SELECT * FROM receipts WHERE store LIKE '%Whole%'")
        assert ok

    def test_select_with_aggregation(self):
        ok, _ = validate_sql("SELECT category, SUM(total) FROM receipts GROUP BY category")
        assert ok

    def test_select_with_order_and_limit(self):
        ok, _ = validate_sql("SELECT * FROM receipts ORDER BY total DESC LIMIT 5")
        assert ok

    def test_select_with_join(self):
        ok, _ = validate_sql(
            "SELECT r.store, i.name FROM receipts r JOIN items i ON r.id = i.receipt_id"
        )
        assert ok

    def test_select_count(self):
        ok, _ = validate_sql("SELECT COUNT(*) FROM receipts")
        assert ok

    def test_select_with_subquery_in_where(self):
        """Subqueries in WHERE clause should be allowed (they're still SELECTs)."""
        ok, _ = validate_sql(
            "SELECT * FROM receipts WHERE total > (SELECT AVG(total) FROM receipts)"
        )
        assert ok

    def test_select_with_having(self):
        ok, _ = validate_sql(
            "SELECT category, AVG(total) FROM receipts GROUP BY category HAVING AVG(total) > 20"
        )
        assert ok

    def test_keyword_in_string_literal_not_blocked(self):
        """'DROP' inside a LIKE string must NOT trigger the blocker."""
        ok, _ = validate_sql("SELECT * FROM receipts WHERE store LIKE '%DROPBOX%'")
        assert ok

    def test_keyword_delete_in_string_literal(self):
        """'DELETE' inside a string literal must pass."""
        ok, _ = validate_sql("SELECT * FROM receipts WHERE store LIKE '%DELETE%'")
        assert ok


# ======================================================================
# validate_sql — Dangerous queries that MUST be blocked
# ======================================================================

class TestBlockedSQLQueries:
    """Queries that must be rejected by the AST validator."""

    def test_drop_table(self):
        ok, reason = validate_sql("DROP TABLE receipts")
        assert not ok
        assert "SELECT" in reason or "Drop" in reason

    def test_delete_from(self):
        ok, reason = validate_sql("DELETE FROM receipts WHERE id = 1")
        assert not ok

    def test_update(self):
        ok, reason = validate_sql("UPDATE receipts SET total = 0")
        assert not ok

    def test_insert(self):
        ok, reason = validate_sql("INSERT INTO receipts VALUES (1, 'x', 'y', 1.0, 'z')")
        assert not ok

    def test_alter_table(self):
        ok, reason = validate_sql("ALTER TABLE receipts ADD COLUMN foo TEXT")
        assert not ok

    def test_create_table(self):
        ok, reason = validate_sql("CREATE TABLE evil (id INTEGER)")
        assert not ok

    def test_sqlite_master_access(self):
        ok, reason = validate_sql("SELECT * FROM sqlite_master")
        assert not ok
        assert "sqlite_master" in reason

    def test_sqlite_schema_access(self):
        ok, reason = validate_sql("SELECT * FROM sqlite_schema")
        assert not ok
        assert "sqlite_schema" in reason

    def test_load_extension(self):
        ok, reason = validate_sql("SELECT load_extension('evil.so')")
        assert not ok
        assert "load_extension" in reason

    def test_union_select(self):
        ok, reason = validate_sql(
            "SELECT * FROM receipts UNION SELECT sql FROM sqlite_master"
        )
        assert not ok

    def test_multi_statement(self):
        ok, reason = validate_sql("SELECT 1; DROP TABLE receipts")
        assert not ok

    def test_multi_statement_semicolon(self):
        ok, reason = validate_sql("SELECT * FROM receipts; DELETE FROM receipts")
        assert not ok

    def test_empty_query(self):
        ok, reason = validate_sql("")
        assert not ok

    def test_garbage_input(self):
        ok, reason = validate_sql("THIS IS NOT SQL AT ALL")
        assert not ok

    def test_intersect(self):
        ok, reason = validate_sql(
            "SELECT * FROM receipts INTERSECT SELECT * FROM items"
        )
        assert not ok

    def test_except_clause(self):
        ok, reason = validate_sql(
            "SELECT * FROM receipts EXCEPT SELECT * FROM items"
        )
        assert not ok


# ======================================================================
# execute_sql — Integration tests
# ======================================================================

class TestExecuteSQL:
    """Integration tests for safe query execution."""

    def test_execute_valid_query(self, populated_db):
        results = execute_sql("SELECT * FROM receipts")
        assert isinstance(results, list)
        assert len(results) > 0
        assert "store" in results[0]

    def test_execute_blocked_query_raises(self, tmp_db):
        with pytest.raises(ValueError, match="Execution blocked"):
            execute_sql("DROP TABLE receipts")

    def test_execute_returns_dicts(self, populated_db):
        results = execute_sql("SELECT store, total FROM receipts WHERE total > 100")
        assert isinstance(results, list)
        for row in results:
            assert isinstance(row, dict)
            assert "store" in row
            assert "total" in row

    def test_row_limit_enforced(self, populated_db):
        """Queries without LIMIT should get one appended."""
        results = execute_sql("SELECT * FROM receipts")
        # The SQL_MAX_ROWS default is 1000, and we only have 12 rows,
        # so this just verifies no error is thrown
        assert isinstance(results, list)

    def test_execute_empty_result(self, tmp_db):
        results = execute_sql("SELECT * FROM receipts WHERE total > 999999")
        assert results == []


# ======================================================================
# insert_receipt
# ======================================================================

class TestInsertReceipt:
    """Tests for parameterized receipt insertion."""

    def test_insert_and_retrieve(self, tmp_db):
        receipt_id = insert_receipt(
            store="TestMart",
            date="2024-03-01",
            total=42.50,
            category="groceries",
            items=["milk", "bread", "eggs"],
        )
        assert isinstance(receipt_id, int)
        assert receipt_id > 0

        results = execute_sql(f"SELECT * FROM receipts WHERE id = {receipt_id}")
        assert len(results) == 1
        assert results[0]["store"] == "TestMart"
        assert results[0]["total"] == 42.50

    def test_insert_items_linked(self, tmp_db):
        receipt_id = insert_receipt(
            store="Store", date="2024-01-01", total=10.0,
            category="test", items=["item_a", "item_b"],
        )
        items = execute_sql(
            f"SELECT * FROM items WHERE receipt_id = {receipt_id}"
        )
        assert len(items) == 2
        names = {i["name"] for i in items}
        assert names == {"item_a", "item_b"}

    def test_insert_no_items(self, tmp_db):
        receipt_id = insert_receipt(
            store="EmptyStore", date="2024-01-01", total=0.0,
            category="other", items=[],
        )
        assert receipt_id > 0


# ======================================================================
# Duplicate hash detection
# ======================================================================

class TestDuplicateHash:
    """Tests for perceptual hash deduplication helpers."""

    def test_no_duplicate_initially(self, tmp_db):
        assert check_duplicate_hash("abc123") is False

    def test_insert_and_detect_duplicate(self, tmp_db):
        rid = insert_receipt("S", "2024-01-01", 1.0, "c", [])
        insert_receipt_hash("abc123", rid)
        assert check_duplicate_hash("abc123") is True

    def test_different_hash_not_duplicate(self, tmp_db):
        rid = insert_receipt("S", "2024-01-01", 1.0, "c", [])
        insert_receipt_hash("abc123", rid)
        assert check_duplicate_hash("xyz789") is False

    def test_duplicate_hash_insert_raises(self, tmp_db):
        rid = insert_receipt("S", "2024-01-01", 1.0, "c", [])
        insert_receipt_hash("duplicate", rid)
        with pytest.raises(sqlite3.IntegrityError):
            insert_receipt_hash("duplicate", rid)
