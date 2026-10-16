# Production Dockerfile for RAG Application (CPU-Optimized)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install minimal system utilities for compilation and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .

# Install dependencies (fastembed uses ONNX runtime, no PyTorch/CUDA needed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ ./app/
COPY .env.example .env

# Create volume mount points with correct permissions
RUN mkdir -p /app/data/documents /app/metadata

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to verify Streamlit responsiveness
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start Streamlit application
CMD ["streamlit", "run", "app/streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
