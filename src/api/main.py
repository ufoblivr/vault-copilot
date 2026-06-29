"""
Production-grade FastAPI application for Vault Copilot.
Features: lifespan management, API key auth, streaming chat,
async upload processing, health checks, request tracing.
"""
import os
import uuid
import shutil
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from src.config import (
    ALLOWED_ORIGINS, API_KEY, MAX_UPLOAD_SIZE_MB, UPLOAD_DIR,
)
from src.memory.sqlite_db import (
    init_sqlite, insert_receipt, insert_receipt_hash, check_duplicate_hash,
)

# ---------------------------------------------------------------------------
# Global references (set during lifespan)
# ---------------------------------------------------------------------------
_copilot = None
_ocr_pipeline = None


def _get_copilot():
    if _copilot is None:
        raise HTTPException(status_code=503, detail="Copilot not initialized")
    return _copilot


def _get_ocr():
    if _ocr_pipeline is None:
        raise HTTPException(status_code=503, detail="OCR pipeline not initialized")
    return _ocr_pipeline


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _copilot, _ocr_pipeline
    from src.agent.graph import FinancialCopilot
    from src.ocr.pipeline import OCRPipeline

    logger.info("Starting Vault Copilot API")
    logger.info("Initializing database tables")
    init_sqlite()

    logger.info("Loading AI models (this may take a moment)")
    _copilot = FinancialCopilot()
    _ocr_pipeline = OCRPipeline()
    logger.info("All models loaded — API ready")

    yield  # Application runs

    # Shutdown
    logger.info("Shutting down Vault Copilot API")
    _copilot = None
    _ocr_pipeline = None


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vault Copilot API",
    description="Privacy-first financial AI copilot with multi-step reasoning",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
async def verify_api_key(request: Request):
    """
    API key authentication via X-API-Key header.
    Disabled when VAULT_API_KEY is empty (dev mode).
    """
    if not API_KEY:
        return  # Auth disabled in dev mode

    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        logger.warning("Auth failure from {}", request.client.host if request.client else "unknown")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    with logger.contextualize(request_id=request_id):
        logger.info("{} {}", request.method, request.url.path)
        response = await call_next(request)
        latency = (time.perf_counter() - start) * 1000
        logger.info(
            "{} {} -> {} ({:.0f}ms)",
            request.method, request.url.path, response.status_code, latency,
        )
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default="default", max_length=100)


class ChatResponse(BaseModel):
    response: str
    execution_trace: list = Field(default_factory=list)
    total_latency_ms: float = 0.0
    steps_taken: int = 0


class UploadResponse(BaseModel):
    message: str
    extracted_data: dict = Field(default_factory=dict)
    receipt_id: Optional[int] = None
    duplicate: bool = False


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    database_initialized: bool
    version: str = "2.0.0"


class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return HealthResponse(
        status="healthy" if _copilot else "starting",
        models_loaded=_copilot is not None and _ocr_pipeline is not None,
        database_initialized=True,
    )


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Chat"],
)
async def chat_endpoint(request: ChatRequest):
    """
    Multi-step agentic chat. The agent reasons across SQL, RAG, and analytics
    tools to build a comprehensive answer.
    """
    copilot = _get_copilot()
    try:
        result = copilot.chat(request.query, session_id=request.session_id)
        return ChatResponse(**result)
    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/chat/stream",
    dependencies=[Depends(verify_api_key)],
    tags=["Chat"],
)
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events.
    Sends execution trace steps as they happen, then the final response.
    """
    import json
    import asyncio

    copilot = _get_copilot()

    async def event_generator():
        try:
            # Send a "thinking" event
            yield f"data: {json.dumps({'type': 'status', 'content': 'Agent is reasoning...'})}\n\n"
            await asyncio.sleep(0)

            # Run the agent (synchronous, so we wrap it)
            result = copilot.chat(request.query, session_id=request.session_id)

            # Stream the execution trace
            for step in result.get("execution_trace", []):
                yield f"data: {json.dumps({'type': 'trace', 'content': step})}\n\n"
                await asyncio.sleep(0)

            # Stream the final response
            yield f"data: {json.dumps({'type': 'response', 'content': result['response']})}\n\n"

            # Send completion event
            yield f"data: {json.dumps({'type': 'done', 'total_latency_ms': result.get('total_latency_ms', 0)})}\n\n"

        except Exception as e:
            logger.exception("Streaming chat error")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post(
    "/api/upload",
    response_model=UploadResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Upload"],
)
async def upload_endpoint(file: UploadFile = File(...)):
    """
    Upload and process a receipt image. Extracts data via OCR,
    validates with Pydantic, stores in dual-memory system (SQL + vector).
    Includes deduplication and confidence scoring.
    """
    copilot = _get_copilot()
    ocr = _get_ocr()

    # --- Validate file ---
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    # Read file content and check size
    content = await file.read()
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / 1024 / 1024:.1f}MB). Max: {MAX_UPLOAD_SIZE_MB}MB.",
        )

    # --- Save temporarily ---
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        # 1. Extract and validate
        data = ocr.process_image(file_path, copilot.pipe)

        # 2. Check for extraction failure
        if data.get("extraction_failed"):
            logger.warning(
                "OCR extraction failed for {}: {}",
                file.filename, data.get("error", "unknown"),
            )
            return UploadResponse(
                message="Extraction failed. The image could not be processed reliably.",
                extracted_data=data,
                duplicate=False,
            )

        # 3. Check for duplicates via perceptual hash
        phash = data.get("phash", "")
        if phash and check_duplicate_hash(phash):
            logger.info("Duplicate receipt detected: phash={}", phash)
            return UploadResponse(
                message="Duplicate receipt detected. This image has already been processed.",
                extracted_data=data,
                duplicate=True,
            )

        # 4. Insert into SQLite
        items_list = data.get("items", [])
        item_names = [
            item.get("name", "Unknown") if isinstance(item, dict) else str(item)
            for item in items_list
        ]
        receipt_id = insert_receipt(
            store=data.get("store", "Unknown"),
            date=data.get("date", "Unknown"),
            total=float(data.get("total", 0.0)),
            category=data.get("category", "Unknown"),
            items=item_names,
        )

        # 5. Store perceptual hash
        if phash:
            insert_receipt_hash(phash, receipt_id)

        # 6. Store in vector memory
        items_str = ", ".join(item_names)
        mem_text = (
            f"Spent ${data.get('total', 0)} at {data.get('store', 'Unknown')} "
            f"on {data.get('date', 'Unknown')}. Items: {items_str}."
        )
        copilot.rag.add_memory(
            str(receipt_id),
            mem_text,
            {"store": str(data.get("store", "Unknown"))},
        )

        logger.info(
            "Receipt vaulted: id={}, store={}, total={}",
            receipt_id, data.get("store"), data.get("total"),
        )

        return UploadResponse(
            message="Receipt safely vaulted and image permanently deleted.",
            extracted_data=data,
            receipt_id=receipt_id,
        )

    finally:
        # PRIVACY: Delete PII immediately after processing
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug("Deleted uploaded file: {}", safe_filename)


@app.get(
    "/api/analytics",
    dependencies=[Depends(verify_api_key)],
    tags=["Analytics"],
)
async def analytics_endpoint():
    """
    Returns structured financial analytics data for the dashboard.
    Bypasses the LLM — pure deterministic computation.
    """
    from src.agent.tools import FinancialIntelligence

    intel = FinancialIntelligence()
    result = intel.analyze_spending()
    if isinstance(result, dict):
        return result
    return {"report": result}