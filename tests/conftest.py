"""
Shared test fixtures for the Vault Copilot test suite.
"""
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Database fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Create a temporary SQLite database and patch DB_PATH to point to it."""
    db_path = str(tmp_path / "test_finance.db")
    monkeypatch.setattr("src.config.DB_PATH", db_path)

    # Also patch the module-level reference in every module that did
    # `from src.config import DB_PATH` (creates a local copy)
    for mod_path in ("src.memory.sqlite_db", "src.agent.tools"):
        try:
            mod = __import__(mod_path, fromlist=["DB_PATH"])
            monkeypatch.setattr(mod, "DB_PATH", db_path)
        except (ImportError, AttributeError):
            pass

    from src.memory.sqlite_db import init_sqlite
    init_sqlite()

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def populated_db(tmp_db):
    """A tmp_db pre-populated with sample receipt data."""
    conn = sqlite3.connect(tmp_db)
    c = conn.cursor()

    receipts = [
        ("Whole Foods", "2024-01-15", 45.99, "groceries"),
        ("Starbucks", "2024-01-16", 12.50, "coffee"),
        ("Whole Foods", "2024-01-22", 52.30, "groceries"),
        ("Amazon", "2024-01-20", 199.99, "electronics"),
        ("Starbucks", "2024-01-23", 8.75, "coffee"),
        ("Target", "2024-02-01", 67.40, "groceries"),
        ("Starbucks", "2024-02-05", 11.20, "coffee"),
        ("Shell Gas", "2024-02-10", 55.00, "gas"),
        ("Whole Foods", "2024-02-15", 48.10, "groceries"),
        ("Netflix", "2024-01-01", 15.99, "entertainment"),
        ("Netflix", "2024-02-01", 15.99, "entertainment"),
        ("Netflix", "2024-03-01", 15.99, "entertainment"),
    ]

    items_data = {
        1: ["organic milk", "avocados", "quinoa"],
        2: ["grande latte", "croissant"],
        3: ["salmon", "broccoli", "rice"],
        4: ["wireless headphones"],
        5: ["cappuccino"],
    }

    for store, date, total, category in receipts:
        c.execute(
            "INSERT INTO receipts (store, date, total, category) VALUES (?, ?, ?, ?)",
            (store, date, total, category),
        )

    for receipt_id, items in items_data.items():
        for item in items:
            c.execute(
                "INSERT INTO items (receipt_id, name) VALUES (?, ?)",
                (receipt_id, item),
            )

    conn.commit()
    conn.close()
    return tmp_db


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_receipts():
    """Returns a list of sample receipt dicts for testing."""
    return [
        {"store": "Whole Foods", "date": "2024-01-15", "total": 45.99, "category": "groceries",
         "items": [{"name": "organic milk"}, {"name": "avocados"}]},
        {"store": "Starbucks", "date": "2024-01-16", "total": 12.50, "category": "coffee",
         "items": [{"name": "grande latte"}]},
        {"store": "Amazon", "date": "2024-01-20", "total": 199.99, "category": "electronics",
         "items": [{"name": "wireless headphones"}]},
        {"store": "Target", "date": "2024-02-01", "total": 67.40, "category": "groceries",
         "items": [{"name": "paper towels"}, {"name": "cereal"}]},
        {"store": "Shell Gas", "date": "2024-02-10", "total": 55.00, "category": "gas",
         "items": [{"name": "fuel"}]},
    ]


# ---------------------------------------------------------------------------
# Mock LLM pipeline
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_pipeline():
    """A mock HuggingFace text-generation pipeline that returns plausible outputs."""
    def pipeline_fn(prompt: str, **kwargs):
        # Route queries → return tool tag
        if "Route" in prompt or "route" in prompt:
            return [{"generated_text": "<TOOL:SQL>"}]

        # SQL generation → return a valid SELECT
        if "SQLite" in prompt or "SELECT" in prompt:
            return [{"generated_text": "* FROM receipts LIMIT 10"}]

        # JSON extraction → return structured receipt
        if "Extract" in prompt or "JSON" in prompt:
            return [{"generated_text": '"store": "TestStore", "date": "2024-01-01", "total": 25.99, "category": "food", "items": ["burger", "fries"]}'}]

        # Planning/reasoning
        if "planning" in prompt.lower() or "tool" in prompt.lower():
            return [{"generated_text": '{"reasoning": "Need SQL data", "tool": "SQL", "tool_input": ""}'}]

        # Default synthesizing response
        return [{"generated_text": "Based on the data, your total spending is $100.00."}]

    return pipeline_fn


# ---------------------------------------------------------------------------
# Temp image fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def test_image(tmp_path):
    """Creates a simple test image file for OCR testing."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (200, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "WHOLE FOODS", fill="black")
    draw.text((10, 30), "Milk  $3.99", fill="black")
    draw.text((10, 50), "Total: $3.99", fill="black")

    path = str(tmp_path / "test_receipt.png")
    img.save(path)
    return path
