# CLAIRE-RAG-Docker

Containerized distribution of CLAIRE-RAG -- a retrieval-augmented generation pipeline for cybersecurity knowledge bases (CVE, CWE, CAPEC, ATT&CK, NICE, DCWF).

## Prerequisites

- Docker (with Docker Compose)
- OpenAI API key

## Quick Start

```bash
git clone https://github.com/jkirc001/CLAIRE-RAG-Docker.git
cd CLAIRE-RAG-Docker
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
./scripts/fetch-vectorstore.sh
./ask "What is CWE-79?"
```

The first `./ask` starts the server container (models load once, ~10s). Subsequent queries are fast (~2s local pipeline + OpenAI API latency).

## How It Works

The RAG pipeline runs as a persistent server inside a single Docker container. Models load once on startup and stay in memory between queries.

1. **Retriever** -- embeds your query with `all-MiniLM-L6-v2`, searches ChromaDB for top-50 candidates
2. **Ranker** -- re-ranks candidates with cross-encoder `ms-marco-MiniLM-L-6-v2`, returns top-10
3. **Generator** -- builds a prompt with the question and retrieved chunks, calls OpenAI for the answer

The ChromaDB vectorstore is pre-built with all six datasets and mounted from the host into the container. HuggingFace models are baked into the Docker image at build time.

## CLI Options

```
./ask "question"                  # Ask a question
./ask "question" --k 5            # Return top 5 chunks
./ask "question" --dataset CWE    # Filter by dataset
./ask "question" --show-chunks    # Show retrieved chunks
./ask "question" --no-ranker      # Disable cross-encoder re-ranking
```

## Managing the Server

```bash
docker compose up -d              # Start the server manually
docker compose down               # Stop the server
docker compose logs               # View server logs
```

The `./ask` script auto-starts the server if it's not running. To stop it when done, use `docker compose down`.

## Building Locally

If you want to build the Docker image locally instead of pulling from GHCR:

```bash
docker compose build
```

The `docker-compose.override.yml` file (gitignored) enables local builds. To create one:

```yaml
services:
  app:
    build: .
    image: claire-rag-app:local
```

## Image Size

The Docker image is approximately 2-4 GB due to PyTorch (CPU-only) and HuggingFace models. This is expected.

## Related

- [CLAIRE-RAG](https://github.com/jkirc001/CLAIRE-RAG) -- main research repository
- [CLAIRE-KG-Docker](https://github.com/jkirc001/CLAIRE-KG-Docker) -- knowledge graph variant
