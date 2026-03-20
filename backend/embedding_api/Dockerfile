# ------------------------------------------------------------------ #
# Stage 1: builder — install Python dependencies                       #
# ------------------------------------------------------------------ #
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools required for C-extension packages (e.g. asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
  && rm -rf /var/lib/apt/lists/*

COPY embedding_api/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ------------------------------------------------------------------ #
# Stage 2: runner — lean production image                              #
# ------------------------------------------------------------------ #
FROM python:3.12-slim AS runner

WORKDIR /app

# Copy only the installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy shared package and application code
COPY shared/ /app/backend/shared/
COPY embedding_api/ /app/backend/embedding_api/

# Create top-level backend package marker so imports work
RUN touch /app/backend/__init__.py

# Run as non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["uvicorn", "backend.embedding_api.main:app", "--host", "0.0.0.0", "--port", "8001"]
