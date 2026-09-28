# ==============================================================================
# AesFix Multi-Stage Production Dockerfile
#
# Why a multi-stage CPU-optimized build is chosen:
# 1. Reduces final image footprint from ~8GB (CUDA full) down to <1.2GB.
# 2. Tier-A deterministic edits run in <25ms on CPU, making CPU inference
#    dramatically more cost-effective (>85% gross margin at <$0.003/photo).
# 3. Non-root user execution prevents container privilege escalation attacks.
# ==============================================================================

# Stage 1: Build & Dependency Resolution
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /install

# Install build-essential for C-extensions (scipy, kornia, libraw)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU wheels first to avoid downloading heavy CUDA binaries
RUN pip install --prefix=/install/wheels torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining python packages into the prefix
COPY requirements.txt .
RUN pip install --prefix=/install/wheels -r requirements.txt \
    uvicorn[standard] \
    prometheus-client \
    mcp

# ------------------------------------------------------------------------------
# Stage 2: Minimal Production Runtime
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    ENVIRONMENT=production \
    PATH="/usr/local/bin:$PATH"

WORKDIR /app

# Install minimal system runtime libraries for OpenCV, image codecs & healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-compiled wheels from builder stage
COPY --from=builder /install/wheels /usr/local

# Create non-root dedicated service user
RUN groupadd -g 1001 aesfix && \
    useradd -u 1001 -g aesfix -s /bin/bash -m aesfix && \
    mkdir -p /app/results /app/cache && \
    chown -R aesfix:aesfix /app

# Copy application source code with correct ownership
COPY --chown=aesfix:aesfix aesfix/ /app/aesfix/
COPY --chown=aesfix:aesfix pyproject.toml /app/
COPY --chown=aesfix:aesfix docs/ /app/docs/

# Switch to unprivileged execution user
USER aesfix

EXPOSE 8000

# Integrated liveness check verifying FastAPI gateway availability
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Launch the production FastAPI application factory
CMD ["uvicorn", "aesfix.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
