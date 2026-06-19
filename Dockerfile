# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.10-slim

# System libraries required by OpenCV (headless) and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install PyTorch CPU-only first (own layer → cached on rebuilds) ───────────
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining Python dependencies ─────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source only (checkpoint lives on GCS, not in the image) ──
COPY src/        ./src/
# Uncomment the line below only for local testing without GCS:
# COPY experiments/ ./experiments/

# Run from src/ so relative imports (configs/, skp/) resolve correctly
WORKDIR /app/src

# ── Runtime config ────────────────────────────────────────────────────────────
ENV USE_GPU=0
# Cloud Run injects $PORT at runtime (default 8080)
ENV PORT=8080

# Single worker: Cloud Run scales via instances, not OS threads
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1
