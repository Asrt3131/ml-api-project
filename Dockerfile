# ==========================================================
# Stage 1: Builder — install dependencies
# ==========================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Build essentials (needed only if a wheel must be compiled)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ==========================================================
# Stage 2: Runtime — lean, non-root image
# ==========================================================
FROM python:3.11-slim AS runtime

# Create a non-root user
RUN groupadd --system appuser && useradd --system --gid appuser appuser

WORKDIR /app

# Copy only the installed Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/model.pkl \
    ENVIRONMENT=production \
    PORT=8080 \
    LOG_LEVEL=INFO

USER appuser
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]