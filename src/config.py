"""
Centralized configuration for Vault Copilot.
All settings loaded from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("VAULT_DB_PATH", str(BASE_DIR / "finance.db"))
CHROMA_PATH = os.getenv("VAULT_CHROMA_PATH", str(BASE_DIR / "chroma_db"))
UPLOAD_DIR = os.getenv("VAULT_UPLOAD_DIR", str(BASE_DIR / "raw_receipts"))

# --- AI Models ---
LLM_MODEL = os.getenv("VAULT_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
EMBEDDING_MODEL = os.getenv("VAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("VAULT_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Agent ---
MAX_AGENT_STEPS = int(os.getenv("VAULT_MAX_AGENT_STEPS", "3"))
LLM_MAX_NEW_TOKENS = int(os.getenv("VAULT_LLM_MAX_TOKENS", "150"))
LLM_TEMPERATURE = float(os.getenv("VAULT_LLM_TEMPERATURE", "0.1"))

# --- RAG ---
RAG_TOP_K = int(os.getenv("VAULT_RAG_TOP_K", "5"))
RELEVANCE_THRESHOLD = float(os.getenv("VAULT_RELEVANCE_THRESHOLD", "0.3"))
BM25_REBUILD_THRESHOLD = int(os.getenv("VAULT_BM25_REBUILD_THRESHOLD", "100"))

# --- Security ---
API_KEY = os.getenv("VAULT_API_KEY", "")  # Empty = auth disabled (dev mode)
MAX_UPLOAD_SIZE_MB = int(os.getenv("VAULT_MAX_UPLOAD_SIZE_MB", "10"))
ALLOWED_ORIGINS = os.getenv(
    "VAULT_ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
).split(",")

# --- SQL Safety ---
SQL_QUERY_TIMEOUT_MS = int(os.getenv("VAULT_SQL_TIMEOUT_MS", "5000"))
SQL_MAX_ROWS = int(os.getenv("VAULT_SQL_MAX_ROWS", "1000"))

# --- OCR ---
OCR_CONFIDENCE_THRESHOLD = float(os.getenv("VAULT_OCR_CONFIDENCE_THRESHOLD", "0.4"))
DUPLICATE_HASH_THRESHOLD = int(os.getenv("VAULT_DUPLICATE_HASH_THRESHOLD", "10"))
OCR_MAX_TEXT_LENGTH = int(os.getenv("VAULT_OCR_MAX_TEXT_LENGTH", "500"))  # Truncate OCR text for LLM

# --- Logging ---
LOG_LEVEL = os.getenv("VAULT_LOG_LEVEL", "INFO")
LOG_JSON = os.getenv("VAULT_LOG_JSON", "false").lower() == "true"

# --- API ---
API_HOST = os.getenv("VAULT_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("VAULT_API_PORT", "8000"))
