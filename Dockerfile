FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY claire_rag/ claire_rag/
COPY config/ config/

# Install CPU-only torch first to avoid pulling CUDA libraries
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install the package (torch already satisfied, pip won't re-download it)
RUN pip install --no-cache-dir .

# Pre-download HuggingFace models at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

ENTRYPOINT ["python", "-m", "claire_rag.ask"]
CMD ["--help"]
