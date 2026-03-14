FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY embedding_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared package and application code
COPY shared/ /app/backend/shared/
COPY embedding_api/ /app/backend/embedding_api/

# Create top-level backend package marker so imports work
RUN touch /app/backend/__init__.py

# Ensure backend package is importable
ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["uvicorn", "backend.embedding_api.main:app", "--host", "0.0.0.0", "--port", "8001"]
