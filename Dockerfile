# ============================================================================
# Vault Copilot — Multi-stage Dockerfile
# ============================================================================
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for EasyOCR and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage: Dependencies
# ---------------------------------------------------------------------------
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage: Application
# ---------------------------------------------------------------------------
FROM deps AS app

# Create non-root user for security
RUN groupadd -r vault && useradd -r -g vault -d /app vault

# Copy application code
COPY src/ src/
COPY eval/ eval/
COPY .env.example .env.example

# Create data directories
RUN mkdir -p raw_receipts chroma_db logs \
    && chown -R vault:vault /app

USER vault

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Default: run the API server
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
