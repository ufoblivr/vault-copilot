"""
SQLite database layer for Vault Copilot financial ledger.

Provides:
- Schema initialization (receipts, items, receipt_hashes tables)
- AST-based SQL query validation via sqlglot
- Read-only query execution with safety limits and audit logging
- Receipt insertion with parameterized queries
- Perceptual-hash-based duplicate receipt detection
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

import pandas as pd
import sqlglot
from sqlglot import exp
from loguru import logger

from src.config import DB_PATH, SQL_MAX_ROWS, SQL_QUERY_TIMEOUT_MS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_connection() -> sqlite3.Connection:
    """Open a read-write connection to the database.

    Returns:
        sqlite3.Connection: A standard read-write connection.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"PRAGMA busy_timeout = {SQL_QUERY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _readonly_connection() -> sqlite3.Connection:
    """Open a **read-only** connection to the database.

    Uses the SQLite URI filename format with ``mode=ro`` so that any
    accidental write attempt raises an ``OperationalError``.

    Returns:
        sqlite3.Connection: A read-only connection.
    """
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute(f"PRAGMA busy_timeout = {SQL_QUERY_TIMEOUT_MS}")
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

_SCHEMA_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS receipts (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        store    TEXT    NOT NULL,
        date     TEXT    NOT NULL,
        total    REAL    NOT NULL,
        category TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS items (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        receipt_id INTEGER NOT NULL,
        name       TEXT    NOT NULL,
        FOREIGN KEY (receipt_id) REFERENCES receipts (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS receipt_hashes (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        phash      TEXT    UNIQUE NOT NULL,
        receipt_id INTEGER NOT NULL,
        FOREIGN KEY (receipt_id) REFERENCES receipts (id) ON DELETE CASCADE
    )
    """,
]


def init_sqlite() -> None:
    """Create the financial-ledger schema if it does not already exist.

    Tables created:
    - **receipts** – header-level receipt data (store, date, total, category).
    - **items** – line items linked to a receipt via ``receipt_id``.
    - **receipt_hashes** – perceptual hashes for duplicate detection.
    """
    conn = _write_connection()
    try:
        cursor = conn.cursor()
        for stmt in _SCHEMA_STATEMENTS:
            cursor.execute(stmt)
        conn.commit()
        logger.info("SQLite schema initialised successfully at {}", DB_PATH)
    except sqlite3.Error as exc:
        logger.error("Failed to initialise SQLite schema: {}", exc)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# AST-based SQL validation
# ---------------------------------------------------------------------------

# System tables that must never be queried.
_BLOCKED_TABLES: frozenset[str] = frozenset({"sqlite_master", "sqlite_schema"})

# Dangerous function names (case-insensitive comparison).
_BLOCKED_FUNCTIONS: frozenset[str] = frozenset({"load_extension"})

# AST node types representing DDL / DML mutations.
_BLOCKED_NODE_TYPES: tuple[type[exp.Expression], ...] = (
    exp.Drop,
    exp.Delete,
    exp.Update,
    exp.Insert,
    exp.Create,
    exp.Alter,
)


def validate_sql(query: str) -> tuple[bool, str]:
    """Validate an SQL string using AST analysis via *sqlglot*.

    The function parses the query into an AST and applies the following
    safety rules:

    1. Exactly **one** statement is allowed.
    2. The statement must be a ``SELECT``.
    3. No DDL / DML sub-expressions (DROP, DELETE, UPDATE, INSERT, CREATE,
       ALTER TABLE).
    4. No references to internal SQLite tables (``sqlite_master``,
       ``sqlite_schema``).
    5. No calls to dangerous functions (e.g. ``load_extension``).
    6. No ``UNION`` operations that could leak data from other tables.

    Args:
        query: The raw SQL string to validate.

    Returns:
        A ``(is_valid, reason)`` tuple.  When *is_valid* is ``True``,
        *reason* is an empty string.
    """
    # --- Parse ---------------------------------------------------------------
    try:
        statements = sqlglot.parse(query, dialect="sqlite")
    except sqlglot.errors.ParseError as exc:
        return False, f"SQL parse error: {exc}"

    # Filter out any ``None`` entries that sqlglot may yield for whitespace.
    statements = [s for s in statements if s is not None]

    # Rule 1 – exactly one statement.
    if len(statements) != 1:
        return False, f"Expected exactly 1 statement, got {len(statements)}"

    root = statements[0]

    # Rule 2 – must be a SELECT.
    if not isinstance(root, exp.Select):
        return False, f"Only SELECT statements are allowed, got {type(root).__name__}"

    # --- Walk the AST --------------------------------------------------------
    for node in root.walk():
        # Rule 3 – no DDL/DML nodes.
        if isinstance(node, _BLOCKED_NODE_TYPES):
            return False, f"Blocked operation detected: {type(node).__name__}"

        # Rule 4 – no system tables.
        if isinstance(node, exp.Table):
            table_name = (node.name or "").lower()
            if table_name in _BLOCKED_TABLES:
                return False, f"Access to system table '{table_name}' is blocked"

        # Rule 5 – no dangerous functions.
        if isinstance(node, (exp.Anonymous, exp.Func)):
            func_name: str = ""
            if isinstance(node, exp.Anonymous):
                func_name = (node.name or "").lower()
            elif hasattr(node, "sql_name"):
                func_name = node.sql_name().lower()
            elif hasattr(node, "key"):
                func_name = node.key.lower()

            if func_name in _BLOCKED_FUNCTIONS:
                return False, f"Blocked function call: {func_name}"

        # Rule 6 – no UNION operations.
        if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
            return False, f"Set operations ({type(node).__name__}) are not allowed"

    return True, ""


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def _ensure_limit(query: str) -> str:
    """Append a ``LIMIT`` clause if the query does not already contain one.

    The check is performed on the parsed AST so that ``LIMIT`` inside
    string literals or comments is not mistakenly detected.

    Args:
        query: The validated SQL string.

    Returns:
        The query string, potentially with an appended LIMIT clause.
    """
    try:
        statements = sqlglot.parse(query, dialect="sqlite")
        root = statements[0]
        if root is not None and not root.find(exp.Limit):
            return f"{query.rstrip().rstrip(';')} LIMIT {SQL_MAX_ROWS}"
    except sqlglot.errors.ParseError:
        # Validation already passed; just append textually as a fallback.
        upper = query.upper()
        if "LIMIT" not in upper:
            return f"{query.rstrip().rstrip(';')} LIMIT {SQL_MAX_ROWS}"
    return query


def execute_sql(query: str) -> list[dict[str, Any]]:
    """Execute a validated, read-only SQL query and return the results.

    The function:
    1. Validates the query via :func:`validate_sql`.
    2. Ensures a ``LIMIT`` clause is present.
    3. Opens a **read-only** database connection.
    4. Executes the query and converts the result to a list of dicts.
    5. Logs an audit trail (query text, outcome, row count, latency).

    Args:
        query: The SQL query string to execute.

    Returns:
        A list of dictionaries, one per result row (column-name → value).

    Raises:
        ValueError: If the query fails validation.
        sqlite3.Error: On database-level errors.
    """
    # --- Validate ------------------------------------------------------------
    is_valid, reason = validate_sql(query)
    if not is_valid:
        logger.warning(
            "SQL validation failed | query={!r} | reason={}",
            query,
            reason,
        )
        raise ValueError(f"Execution blocked: {reason}")

    # --- Ensure LIMIT --------------------------------------------------------
    safe_query = _ensure_limit(query)

    # --- Execute -------------------------------------------------------------
    start = time.perf_counter()
    conn = _readonly_connection()
    try:
        df = pd.read_sql_query(safe_query, conn)
        result: list[dict[str, Any]] = df.to_dict(orient="records")
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "SQL executed | rows={} | elapsed_ms={:.1f} | query={!r}",
            len(result),
            elapsed_ms,
            safe_query,
        )
        return result
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error(
            "SQL execution failed | elapsed_ms={:.1f} | query={!r} | error={}",
            elapsed_ms,
            safe_query,
            exc,
        )
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Receipt insertion
# ---------------------------------------------------------------------------

def insert_receipt(
    store: str,
    date: str,
    total: float,
    category: str,
    items: list[str],
) -> int:
    """Insert a new receipt with its line items into the database.

    Uses parameterized queries to prevent SQL injection.

    Args:
        store: The merchant / store name.
        date: The receipt date (ISO-8601 string recommended).
        total: The total amount on the receipt.
        category: Spending category (e.g. ``"groceries"``).
        items: A list of item-name strings belonging to this receipt.

    Returns:
        The ``id`` of the newly inserted receipt row.

    Raises:
        sqlite3.Error: On any database error (transaction is rolled back).
    """
    conn = _write_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO receipts (store, date, total, category) VALUES (?, ?, ?, ?)",
            (store, date, total, category),
        )
        receipt_id: int = cursor.lastrowid  # type: ignore[assignment]

        if items:
            cursor.executemany(
                "INSERT INTO items (receipt_id, name) VALUES (?, ?)",
                [(receipt_id, item_name) for item_name in items],
            )

        conn.commit()
        logger.info(
            "Inserted receipt id={} | store={!r} | items={}",
            receipt_id,
            store,
            len(items),
        )
        return receipt_id
    except sqlite3.Error as exc:
        conn.rollback()
        logger.error("Failed to insert receipt: {}", exc)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Duplicate-hash helpers
# ---------------------------------------------------------------------------

def check_duplicate_hash(phash: str) -> bool:
    """Check whether a perceptual hash already exists in ``receipt_hashes``.

    Args:
        phash: The perceptual hash string to look up.

    Returns:
        ``True`` if a matching hash is found (i.e. the receipt is a
        probable duplicate), ``False`` otherwise.
    """
    conn = _readonly_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM receipt_hashes WHERE phash = ? LIMIT 1",
            (phash,),
        )
        row = cursor.fetchone()
        is_duplicate = row is not None
        logger.debug("Duplicate-hash check | phash={!r} | duplicate={}", phash, is_duplicate)
        return is_duplicate
    except sqlite3.Error as exc:
        logger.error("Duplicate-hash lookup failed: {}", exc)
        raise
    finally:
        conn.close()


def insert_receipt_hash(phash: str, receipt_id: int) -> None:
    """Store a perceptual hash linked to a receipt.

    Args:
        phash: The perceptual hash string.
        receipt_id: The ``id`` of the associated receipt row.

    Raises:
        sqlite3.IntegrityError: If the hash already exists (UNIQUE constraint).
        sqlite3.Error: On other database errors.
    """
    conn = _write_connection()
    try:
        conn.execute(
            "INSERT INTO receipt_hashes (phash, receipt_id) VALUES (?, ?)",
            (phash, receipt_id),
        )
        conn.commit()
        logger.info(
            "Stored receipt hash | phash={!r} | receipt_id={}",
            phash,
            receipt_id,
        )
    except sqlite3.IntegrityError:
        logger.warning(
            "Duplicate hash insertion attempted | phash={!r} | receipt_id={}",
            phash,
            receipt_id,
        )
        raise
    except sqlite3.Error as exc:
        conn.rollback()
        logger.error("Failed to insert receipt hash: {}", exc)
        raise
    finally:
        conn.close()