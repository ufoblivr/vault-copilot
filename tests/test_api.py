"""
API integration tests using FastAPI TestClient with mocked AI models.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_copilot():
    """Create a mock FinancialCopilot that doesn't load real models."""
    copilot = MagicMock()
    copilot.chat.return_value = {
        "response": "Your total spending is $500.00",
        "execution_trace": [
            {"step_number": 1, "tool_selected": "SQL",
             "tool_input": "totals", "tool_output": "data",
             "reasoning": "Need SQL", "latency_ms": 100}
        ],
        "total_latency_ms": 150.0,
        "steps_taken": 1,
    }
    copilot.rag = MagicMock()
    copilot.pipe = MagicMock()
    return copilot


@pytest.fixture
def mock_ocr():
    """Create a mock OCRPipeline."""
    ocr = MagicMock()
    ocr.process_image.return_value = {
        "store": "TestMart", "date": "2024-01-01", "total": 25.99,
        "category": "groceries", "items": [{"name": "milk"}],
        "ocr_confidence": 0.9, "low_confidence": False,
        "extraction_failed": False, "phash": "abc123",
        "raw_ocr_text": "TESTMART MILK 25.99",
    }
    return ocr


@pytest.fixture
def client(mock_copilot, mock_ocr, tmp_db, monkeypatch):
    """Create a TestClient with mocked models."""
    import src.api.main as api_module

    # Patch the globals that lifespan sets
    monkeypatch.setattr(api_module, "_copilot", mock_copilot)
    monkeypatch.setattr(api_module, "_ocr_pipeline", mock_ocr)
    # Disable auth for tests
    monkeypatch.setattr("src.config.API_KEY", "")

    # Avoid running lifespan (which loads real models)
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    api_module.app.router.lifespan_context = noop_lifespan

    with TestClient(api_module.app) as c:
        yield c


@pytest.fixture
def auth_client(mock_copilot, mock_ocr, tmp_db, monkeypatch):
    """Create a TestClient with authentication enabled."""
    import src.api.main as api_module

    monkeypatch.setattr(api_module, "_copilot", mock_copilot)
    monkeypatch.setattr(api_module, "_ocr_pipeline", mock_ocr)
    monkeypatch.setattr("src.config.API_KEY", "test-secret-key")
    # Also patch it where it's already imported
    monkeypatch.setattr(api_module, "API_KEY", "test-secret-key")

    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    api_module.app.router.lifespan_context = noop_lifespan

    with TestClient(api_module.app) as c:
        yield c


# ======================================================================
# Health check
# ======================================================================

class TestHealthCheck:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "version" in data

    def test_health_no_auth_required(self, auth_client):
        """Health check should work without API key."""
        resp = auth_client.get("/api/health")
        assert resp.status_code == 200


# ======================================================================
# Chat endpoint
# ======================================================================

class TestChatEndpoint:
    def test_chat_valid_query(self, client):
        resp = client.post("/api/chat", json={"query": "How much did I spend?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "execution_trace" in data
        assert isinstance(data["response"], str)

    def test_chat_empty_query_rejected(self, client):
        resp = client.post("/api/chat", json={"query": ""})
        assert resp.status_code == 422  # Validation error

    def test_chat_missing_query_rejected(self, client):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_chat_with_session_id(self, client):
        resp = client.post(
            "/api/chat",
            json={"query": "test query", "session_id": "my-session"},
        )
        assert resp.status_code == 200


# ======================================================================
# Upload endpoint
# ======================================================================

class TestUploadEndpoint:
    def test_upload_valid_image(self, client, mock_ocr, monkeypatch):
        """Upload a valid JPEG should return 200."""
        # Patch dedup check to return False
        monkeypatch.setattr("src.api.main.check_duplicate_hash", lambda h: False)
        monkeypatch.setattr("src.api.main.insert_receipt", lambda **kw: 1)
        monkeypatch.setattr("src.api.main.insert_receipt_hash", lambda h, r: None)

        from PIL import Image
        import io
        img = Image.new("RGB", (100, 100), "white")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = client.post(
            "/api/upload",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "extracted_data" in data

    def test_upload_non_image_rejected(self, client):
        resp = client.post(
            "/api/upload",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400


# ======================================================================
# Authentication
# ======================================================================

class TestAuthentication:
    def test_chat_without_key_rejected(self, auth_client):
        resp = auth_client.post("/api/chat", json={"query": "test"})
        assert resp.status_code == 401

    def test_chat_with_wrong_key_rejected(self, auth_client):
        resp = auth_client.post(
            "/api/chat",
            json={"query": "test"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_chat_with_correct_key_succeeds(self, auth_client):
        resp = auth_client.post(
            "/api/chat",
            json={"query": "test"},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert resp.status_code == 200

    def test_upload_without_key_rejected(self, auth_client):
        from PIL import Image
        import io
        img = Image.new("RGB", (100, 100), "white")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = auth_client.post(
            "/api/upload",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )
        assert resp.status_code == 401


# ======================================================================
# Analytics endpoint
# ======================================================================

class TestAnalyticsEndpoint:
    def test_analytics_returns_200(self, client, populated_db, monkeypatch):
        """Note: analytics creates its own FinancialIntelligence instance."""
        resp = client.get("/api/analytics")
        assert resp.status_code == 200

    def test_analytics_returns_dict(self, client, populated_db, monkeypatch):
        resp = client.get("/api/analytics")
        data = resp.json()
        assert isinstance(data, dict)


# ======================================================================
# Request ID middleware
# ======================================================================

class TestRequestID:
    def test_response_has_request_id_header(self, client):
        resp = client.get("/api/health")
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) > 0
