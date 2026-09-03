FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
# Limit OpenMP/BLAS thread pools to 1 to stay well under 512 MB RAM
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# Install system dependencies including ffmpeg for audio extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download fastembed model files during image build so runtime startup has zero download overhead
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2', threads=1)"

# Copy application source code
COPY . .

EXPOSE 8000

# Use 1 worker on 512 MB instances (FastAPI + Granian async event loop handles concurrent I/O)
CMD ["granian", "--interface", "asgi", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
