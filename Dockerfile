FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch to keep image small)
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    transformers>=4.40.0 \
    accelerate>=1.1.0 \
    sentencepiece>=0.2.0 \
    pythainlp>=5.0.0 \
    rapidfuzz>=3.6.0 \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.27.0" \
    pyyaml>=6.0

# Copy source code
COPY src/ src/

# Copy model weights
COPY results/final/ model/

ENV NER_MODEL_DIR=/app/model
ENV NER_DEVICE=cpu
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
