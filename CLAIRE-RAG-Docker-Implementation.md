# CLAIRE-RAG-Docker Implementation Document

## 1. Project Overview

CLAIRE-RAG-Docker is a containerized distribution of the CLAIRE-RAG system, a standard Retrieval-Augmented Generation (RAG) pipeline for cybersecurity knowledge bases. It serves as a baseline for comparative evaluation against CLAIRE-KG (knowledge graph approach) and CLAIRE-DirectLLM (direct LLM approach).

The system retrieves relevant text chunks from a pre-built ChromaDB vectorstore, re-ranks them using a cross-encoder model, and generates answers using OpenAI's API. It is not intended for production cybersecurity analysis -- testing showed a 0% automated pass rate and 5% human-validated pass rate on cross-framework queries, compared to 100%/95% for CLAIRE-KG.

### Repository

- GitHub: https://github.com/jkirc001/CLAIRE-RAG-Docker
- Source pinned to commit `c818497684af556004cf115790b1f66ebf196bb8` of the main [CLAIRE-RAG](https://github.com/jkirc001/CLAIRE-RAG) repository.

---

## 2. Development Environment

### Hardware

- **Machine:** MacBook Pro 16-inch, 2019 ([specs](https://support.apple.com/en-us/111932))
- **CPU:** 2.3 GHz 8-Core Intel Core i9
- **Graphics:** AMD Radeon Pro 5500M 4 GB, Intel UHD Graphics 630 1536 MB
- **Memory:** 32 GB 2667 MHz DDR4
- **OS:** macOS Sequoia 15.7.4 (Darwin 24.6.0)

### Software

- **Docker:** Docker Desktop for Mac with Docker Compose v2
- **Python (host):** 3.14.3 (used only for the `./ask` wrapper script's JSON formatting)
- **Python (container):** 3.11-slim (Debian Trixie base)
- **Build system:** hatchling
- **Package manager (source repo):** uv
- **Container registry:** GitHub Container Registry (GHCR) at `ghcr.io/jkirc001/claire-rag-app`
- **CI/CD:** GitHub Actions

---

## 3. Architecture

### 3.1 RAG Pipeline (3 Components)

1. **Retriever** -- Embeds the user's query using the `all-MiniLM-L6-v2` sentence-transformer model (384-dimensional embeddings), then performs a cosine similarity search against the ChromaDB vectorstore to retrieve the top-50 candidate chunks.

2. **Ranker** -- Re-ranks the 50 candidates using the `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder model. The cross-encoder scores each (query, chunk) pair for relevance and returns the top-10 chunks. This is more accurate than bi-encoder similarity alone.

3. **Generator** -- Constructs a prompt containing the user's question and the top-10 retrieved chunks, then sends it to OpenAI (default: `gpt-4o-mini`) to generate the final answer.

### 3.2 Server Architecture

The system runs as a persistent FastAPI server inside a single Docker container. This was a deliberate decision to avoid the ~11-second cold start that occurred when running the CLI per invocation (PyTorch and HuggingFace model deserialization on every call).

- **Server process:** uvicorn running `claire_rag.server:app` on port 8000
- **Model lifecycle:** Embedding model and cross-encoder load once during server startup via the FastAPI lifespan context manager. They remain in memory for the lifetime of the container.
- **Endpoints:**
  - `GET /health` -- Returns `{"status": "ok"}` for healthchecks
  - `POST /ask` -- Accepts JSON with question and parameters, returns the answer and optionally retrieved chunks

### 3.3 Data Flow

```
User -> ./ask script -> curl POST /ask -> FastAPI server -> Retriever (embed + ChromaDB query)
                                                         -> Ranker (cross-encoder re-rank)
                                                         -> Generator (OpenAI API call)
                                                         -> JSON response -> ./ask script -> stdout
```

---

## 4. Package Structure

```
CLAIRE-RAG-Docker/
  claire_rag/                    # Application source (copied from main repo)
    __init__.py                  # Package init, version 0.1.0
    ask.py                       # Original CLI (typer-based, retained for reference)
    index.py                     # Index build CLI (not used in Docker distribution)
    server.py                    # FastAPI server (added for Docker distribution)
    corpus/
      __init__.py
      build.py                   # Document builder from raw datasets
      chunking.py                # Text chunking with tiktoken (768 tokens, 100 overlap)
      models.py                  # Document dataclass
    datasets/
      __init__.py
      attack.py                  # MITRE ATT&CK dataset loader
      capec.py                   # CAPEC dataset loader
      cve.py                     # CVE dataset loader
      cwe.py                     # CWE dataset loader
      dcwf.py                    # DCWF dataset loader
      nice.py                    # NICE dataset loader
    embeddings/
      __init__.py
      service.py                 # SentenceTransformer embedding (all-MiniLM-L6-v2)
    llm/
      __init__.py
      client.py                  # OpenAI client with config management
      prompts.py                 # Prompt construction (question + context chunks)
      service.py                 # Orchestrates retrieve -> rank -> generate
    ranking/
      __init__.py
      service.py                 # Cross-encoder ranker (ms-marco-MiniLM-L-6-v2)
    retrieval/
      __init__.py
      service.py                 # Vector search and retrieve-and-rank pipeline
    vector_store/
      __init__.py
      store.py                   # ChromaDB PersistentClient wrapper
  config/
    settings.yaml                # LLM settings, ranker config, paths
    models.yaml                  # Allowed models, evaluation model
  scripts/
    fetch-vectorstore.sh         # Downloads vectorstore from GitHub Release
  .github/
    workflows/
      docker-publish.yml         # Multi-arch GHCR build on release
  ask                            # User-facing wrapper script
  Dockerfile
  docker-compose.yml
  docker-compose.override.yml    # Local build override (gitignored)
  pyproject.toml
  .env.example
  .gitignore
  .dockerignore
  VERSION                        # Pinned source commit SHA
  LICENSE                        # GPL-3.0
  README.md
```

---

## 5. Dependencies and Library Choices

### 5.1 Core Dependencies

| Library | Version Constraint | Purpose | Decision Rationale |
|---------|-------------------|---------|-------------------|
| `torch` | `>=2.0.0,<2.2.0` (installed as 2.1.2) | PyTorch tensor operations for ML models | Required by sentence-transformers. Pinned to 2.1.x for compatibility with transformers <4.40.0. |
| `sentence-transformers` | `>=2.2.0,<3.0.0` | Embedding model (`all-MiniLM-L6-v2`) and cross-encoder (`ms-marco-MiniLM-L-6-v2`) | Standard library for bi-encoder and cross-encoder models. |
| `transformers` | `>=4.30.0,<4.40.0` | HuggingFace model infrastructure | Pinned upper bound for torch 2.1.2 compatibility. |
| `numpy` | `<2.0.0` | Numerical operations | Pinned to 1.x for torch 2.1.x compatibility. NumPy 2.0 introduced breaking changes. |
| `chromadb` | `>=0.4.0` | Vector database for storing and querying document embeddings | File-based (SQLite + HNSW index), no separate server needed. |
| `openai` | `>=1.0.0` | OpenAI API client for LLM generation | v1.0+ API with chat completions. |
| `tiktoken` | `>=0.5.0` | Token counting for chunk size management | Uses `cl100k_base` encoding (GPT-3.5/GPT-4 tokenizer). |
| `pydantic` | `>=2.0.0` | Data validation for API request/response models | Used by FastAPI and for Pydantic BaseModel schemas. |
| `pyyaml` | `>=6.0` | YAML config file parsing | Reads settings.yaml and models.yaml. |
| `python-dotenv` | `>=1.0.0` | Environment variable loading from .env files | Loads OPENAI_API_KEY. |
| `typer` | `>=0.20.0` | CLI framework for the original ask/index commands | Retained from source repo; the CLI still works inside the container. |
| `fastapi` | `>=0.100.0` | HTTP server framework | Added for persistent server mode. Lightweight, async-capable. |
| `uvicorn` | `>=0.20.0` | ASGI server for FastAPI | Production-grade server for running the FastAPI app. |

### 5.2 Build Dependencies

| Tool | Purpose |
|------|---------|
| `hatchling` | Python build backend (PEP 517) |
| `gcc`, `g++` | C/C++ compilers required for building native extensions during pip install |

### 5.3 CPU-Only PyTorch

PyTorch was installed using the CPU-only index (`https://download.pytorch.org/whl/cpu`) to avoid downloading CUDA libraries. This reduced the torch package from ~2GB (with CUDA) to ~200MB. GPU acceleration is not needed because:

- The embedding model (`all-MiniLM-L6-v2`) processes single queries at inference time, not batches
- The cross-encoder re-ranks at most 50 chunks per query
- Both operations complete in <2 seconds on CPU

### 5.4 HuggingFace Models

Two models are pre-downloaded at Docker image build time:

| Model | Type | Size | Purpose |
|-------|------|------|---------|
| `sentence-transformers/all-MiniLM-L6-v2` | Bi-encoder | ~80MB | Query embedding (384 dimensions) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder | ~80MB | Re-ranking retrieved chunks |

Pre-downloading avoids:
- Network calls at runtime
- HuggingFace rate limiting
- Dependency on HuggingFace availability

---

## 6. Data

### 6.1 Vectorstore

The ChromaDB vectorstore is pre-built from six cybersecurity datasets:

| Dataset | Description |
|---------|-------------|
| CVE | Common Vulnerabilities and Exposures |
| CWE | Common Weakness Enumeration |
| CAPEC | Common Attack Pattern Enumeration and Classification |
| ATTACK | MITRE ATT&CK framework |
| NICE | NICE Cybersecurity Workforce Framework |
| DCWF | DoD Cyber Workforce Framework |

The vectorstore was built by running `index build` in the main CLAIRE-RAG repository. Documents were chunked using tiktoken (`cl100k_base` encoding) with a chunk size of 768 tokens and 100-token overlap, then embedded with `all-MiniLM-L6-v2` and stored in ChromaDB.

### 6.2 Vectorstore Files

The vectorstore directory contains:

| File | Size | Purpose |
|------|------|---------|
| `chroma.sqlite3` | 256 MB | ChromaDB metadata and document storage |
| `e4f4063e-d93e-4a10-a9a9-f92fc90492c4/data_level0.bin` | 71 MB | HNSW index level 0 data |
| `e4f4063e-d93e-4a10-a9a9-f92fc90492c4/header.bin` | 100 B | HNSW index header |
| `e4f4063e-d93e-4a10-a9a9-f92fc90492c4/index_metadata.pickle` | 2.5 MB | Index metadata |
| `e4f4063e-d93e-4a10-a9a9-f92fc90492c4/length.bin` | 173 KB | HNSW length data |
| `e4f4063e-d93e-4a10-a9a9-f92fc90492c4/link_lists.bin` | 374 KB | HNSW link lists |
| **Total** | **~330 MB** | |

### 6.3 Vectorstore Distribution

The vectorstore is hosted as a GitHub Release asset (`v0.1.0-data`) as a gzip-compressed tarball (`vectorstore.tar.gz`, 176 MB compressed). Users download it with `scripts/fetch-vectorstore.sh`.

Decision: The vectorstore is not baked into the Docker image because:
- It would increase image size by 330 MB on every pull
- It changes independently of the application code
- Mounting from the host allows inspection and replacement without rebuilding

### 6.4 Vectorstore Mount

The vectorstore is mounted as a read-write Docker volume (`./vectorstore:/app/vectorstore`). Read-write access is required because ChromaDB's `PersistentClient` uses SQLite WAL (Write-Ahead Logging) mode, which needs write access even for read-only queries. The original plan specified read-only mounting (`:ro`), but this was changed during implementation when ChromaDB failed to open the database.

---

## 7. Docker Implementation

### 7.1 Base Image

`python:3.11-slim` (Debian Trixie) was chosen because:
- Python 3.11 is required for torch 2.1.x compatibility (torch <2.2.0 does not support Python 3.13+)
- The `-slim` variant minimizes image size while still including essential system libraries
- Debian base provides stable apt package management for gcc/g++

### 7.2 Build Process

The Dockerfile follows this sequence:

1. Install system dependencies (gcc, g++) for native extension compilation
2. Copy `pyproject.toml`, `README.md`, source code, and config
3. Install CPU-only torch 2.1.2 from PyTorch's CPU index (avoids CUDA)
4. Install the package and remaining dependencies via `pip install .`
5. Pre-download HuggingFace models by importing and instantiating them
6. Set the default command to run uvicorn on port 8000

### 7.3 Image Size

The final Docker image is approximately 2.93 GB, broken down roughly as:

- Python 3.11-slim base: ~150 MB
- gcc/g++ and system deps: ~277 MB
- PyTorch (CPU-only): ~200 MB
- sentence-transformers + transformers + dependencies: ~500 MB
- ChromaDB + dependencies: ~200 MB
- HuggingFace models (cached): ~160 MB
- Other Python packages: ~100 MB
- Layers and overhead: remainder

### 7.4 Multi-Architecture Support

The CI/CD pipeline builds for both `linux/amd64` and `linux/arm64`. ARM64 support is achieved via QEMU emulation on GitHub Actions. This allows the image to run natively on:

- Linux x86_64 servers
- Apple Silicon Macs (M1/M2/M3/M4) via Docker Desktop

### 7.5 Docker Compose Configuration

The `docker-compose.yml` defines a single service:

- **Image:** `ghcr.io/jkirc001/claire-rag-app:latest`
- **Port:** 8000:8000 (FastAPI server)
- **Environment:** Loaded from `.env` file (contains `OPENAI_API_KEY`)
- **Volume:** `./vectorstore:/app/vectorstore` (read-write)
- **Healthcheck:** Python urllib request to `http://localhost:8000/health`, 10s interval, 30s start period

A `docker-compose.override.yml` (gitignored) is provided for local development builds that adds `build: .` and overrides the image tag.

---

## 8. CI/CD Pipeline

### 8.1 GitHub Actions Workflow

The workflow (`docker-publish.yml`) triggers on GitHub Release publish events. It:

1. Checks out the repository
2. Sets up QEMU (for ARM64 cross-compilation)
3. Sets up Docker Buildx (multi-platform builder)
4. Logs into GHCR using `GITHUB_TOKEN`
5. Extracts image tags from the release (semver + `latest`)
6. Builds and pushes the multi-arch image with GitHub Actions cache

### 8.2 Image Tags

- `ghcr.io/jkirc001/claire-rag-app:latest` -- latest release
- `ghcr.io/jkirc001/claire-rag-app:<version>` -- specific release (e.g., `0.2.0`)

### 8.3 GHCR Package Visibility

The GHCR package was initially created as private (GitHub default). It was manually changed to public via the package settings page to allow unauthenticated pulls.

---

## 9. Wrapper Script (`./ask`)

The `./ask` bash script provides the user-facing interface. It:

1. Loads environment variables from `.env`
2. Validates `OPENAI_API_KEY` is set
3. Validates the vectorstore exists
4. Checks if the server is running (via `/health` endpoint)
5. If not running, starts it with `docker compose up -d` and waits for healthy
6. Parses CLI flags (`--k`, `--dataset`, `--no-ranker`, `--show-chunks`)
7. Constructs a JSON payload and POSTs to `http://localhost:8000/ask`
8. Extracts and displays the answer from the JSON response

### 9.1 Supported Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--k <n>` | 10 | Number of chunks to return after ranking |
| `--dataset <name>` | (none) | Filter by dataset: CVE, CWE, CAPEC, ATTACK, NICE, DCWF |
| `--no-ranker` | false | Disable cross-encoder re-ranking |
| `--show-chunks` | false | Display retrieved chunks in output |

---

## 10. Configuration

### 10.1 settings.yaml

| Setting | Value | Description |
|---------|-------|-------------|
| `llm.provider` | `openai` | LLM provider |
| `llm.model` | `gpt-4o-mini` | Default model for development mode |
| `llm.temperature` | `0.2` | Low temperature for consistent answers |
| `llm.max_tokens` | `2048` | Maximum response tokens |
| `ranker.enabled` | `true` | Cross-encoder re-ranking on by default |
| `ranker.model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Ranker model |
| `ranker.retrieve_k` | `50` | Candidates to retrieve before ranking |
| `ranker.rank_k` | `10` | Results to return after ranking |

### 10.2 models.yaml

- **Allowed models:** gpt-4o, gpt-4o-mini, gpt-4.1-mini
- **Evaluation model:** gpt-4o (used when `--eval` flag is passed)

### 10.3 Config Path Resolution

Configuration files are resolved relative to the package location using `Path(__file__).parent.parent.parent / "config"`. With the package installed at `/app/claire_rag/`, this resolves to `/app/config/`. The config directory is copied into the image at build time.

---

## 11. Decisions Made During Implementation

### 11.1 Vectorstore as Tarball (Not Just SQLite File)

The original plan specified hosting only `chroma.sqlite3` as the release asset. During implementation, it was discovered that ChromaDB also stores HNSW index data in a UUID-named subdirectory (~77 MB). Both are required. The release asset was changed to a tarball of the entire `vectorstore/` directory.

### 11.2 Read-Write Volume Mount (Not Read-Only)

The plan specified mounting the vectorstore as read-only (`:ro`). ChromaDB's `PersistentClient` uses SQLite WAL mode, which requires write access even for queries. The mount was changed to read-write.

### 11.3 Server Mode (Not CLI-Per-Invocation)

The original design ran `python -m claire_rag.ask` per invocation via `docker compose run --rm`. Each call took ~11 seconds due to PyTorch and HuggingFace model deserialization. A FastAPI server mode was added (Issue #1) that loads models once on startup, reducing per-query time to ~2 seconds. The `./ask` wrapper was rewritten to curl the server instead of running docker compose run.

### 11.4 CPU-Only PyTorch

The plan suggested using `--index-url https://download.pytorch.org/whl/cpu` for torch. This was implemented, reducing the torch package from ~2 GB (with CUDA) to ~200 MB. GPU is unnecessary for single-query inference.

### 11.5 Python 3.11 (Not 3.13)

The plan noted that `torch>=2.0.0,<2.2.0` may not support Python 3.13. Python 3.11 was chosen for guaranteed compatibility.

### 11.6 Single Container (Not Multi-Service)

Unlike CLAIRE-KG-Docker (which runs Neo4j as a separate container), CLAIRE-RAG-Docker uses a single container. ChromaDB is file-based and runs in-process, so no database service is needed.

### 11.7 GHCR Package Visibility

GitHub creates GHCR packages as private by default, even for public repositories. The package had to be manually changed to public via the GitHub web UI. The `gh` CLI could not do this because the authentication token lacked the `read:packages` scope.

### 11.8 ARM64 via QEMU (Not Native Runners)

Multi-arch builds use QEMU emulation on GitHub's x86_64 runners rather than native ARM64 runners. This is slower (~17 minutes total build time) but does not require special runner configuration.

### 11.9 No `--rm` Flag Tradeoff

Initial implementation used `docker compose run --rm` to avoid container accumulation. When `--rm` was removed to test reuse, orphan container warnings appeared on every invocation. The server mode eliminated this tradeoff entirely -- the container runs persistently and no `docker compose run` is used.

---

## 12. Performance

### 12.1 Server Startup

- Model loading (embedding + cross-encoder): ~10 seconds
- This happens once when the container starts

### 12.2 Per-Query Performance (Server Mode)

| Component | Time |
|-----------|------|
| Query embedding (all-MiniLM-L6-v2) | ~100 ms |
| ChromaDB vector search (top-50) | ~200 ms |
| Cross-encoder re-ranking (50 chunks) | ~800 ms |
| OpenAI API call | 2-5 s (variable, network-dependent) |
| **Total (local pipeline)** | **~1.2 s** |
| **Total (with OpenAI)** | **~3-6 s** |

### 12.3 Cold Start Comparison

| Mode | First Query | Second Query |
|------|-------------|--------------|
| CLI per invocation (`docker compose run`) | ~11 s | ~11 s |
| Server mode (after startup) | ~2.6 s | ~1.8 s |

---

## 13. Releases

| Release | Purpose | Assets |
|---------|---------|--------|
| `v0.1.0-data` | Vectorstore data | `vectorstore.tar.gz` (176 MB) |
| `v0.1.0` | Initial release (CLI mode, pre-server) | Docker image (amd64 + arm64) |
| `v0.2.0` | Server mode | Docker image (amd64 + arm64) |

---

## 14. Git History

```
14a9822 Strengthen README warning about unreliable answers
2efbf3f Simplify README performance warning
72467f1 Add performance comparison warning to README
14a89d7 Add persistent server mode to eliminate cold start (#1)
e44784e Restore --rm to prevent orphan container accumulation
cd5e2b0 Remove --rm from ask wrapper to reuse containers
00636df Add multi-arch GHCR build (amd64 + arm64)
488d1bd Add build directive to docker-compose.yml
90cb927 Initial CLAIRE-RAG-Docker distribution
f590cf6 Initial commit
```

---

## 15. Files Copied from Source Repository

The following files were copied from the main CLAIRE-RAG repository (commit `c818497684af556004cf115790b1f66ebf196bb8`):

**Included:**
- `claire_rag/` -- Complete application source package
- `config/settings.yaml` -- Runtime configuration
- `config/models.yaml` -- Model configuration
- `pyproject.toml` -- Build configuration and dependencies
- `LICENSE` -- GPL-3.0

**Excluded:**
- `data/` -- Raw dataset files (not needed; vectorstore is pre-built)
- `artifacts/` -- Intermediate build artifacts
- `vectorstore/` -- Delivered separately via GitHub Release
- `docs/`, `tests/`, `scripts/` -- Not needed in Docker distribution
- `config/.env`, `config/.env.example` -- Replaced by root-level `.env`
- `.venv/` -- Virtual environment
- `uv.lock` -- uv lockfile (not used in Docker; pip installs from pyproject.toml)

**Added for Docker distribution:**
- `claire_rag/server.py` -- FastAPI server
- `Dockerfile`
- `docker-compose.yml`
- `docker-compose.override.yml`
- `ask` -- Wrapper script
- `scripts/fetch-vectorstore.sh`
- `.github/workflows/docker-publish.yml`
- `.env.example`, `.gitignore`, `.dockerignore`
- `VERSION`
- `README.md`
