# ─── Build stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# ─── Runtime environment ─────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Default: run the FastAPI server for HF Spaces
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
