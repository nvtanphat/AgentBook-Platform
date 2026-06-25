# ============================================================
# Stage 1: Builder — install deps into virtualenv
# ============================================================
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch is installed first from a dedicated index so it doesn't drag a
# mismatched CUDA build via transitive deps.
#   CPU machine  → default cpu index (lean ~200 MB, no GPU libs)
#   GPU machine  → pass --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
#                  (cu121 wheels bundle the CUDA runtime; RTX 40xx/4060 compatible)
# docker-compose.gpu.yml sets this automatically.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    torch torchvision --index-url ${TORCH_INDEX_URL}

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================================
# Stage 2: Runtime — lean image, no build tools
# ============================================================
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser -d /app appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend

COPY backend/src /app/backend/src
COPY config /app/config
RUN mkdir -p /app/data/raw /app/data/processed \
    && chown -R appuser:appuser /app

WORKDIR /app/backend
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--limit-concurrency", "50", "--timeout-keep-alive", "30"]
