"""FastAPI server for CLAIRE-RAG. Keeps models loaded between queries."""

import logging
import os
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from claire_rag.embeddings.service import _get_model
from claire_rag.ranking.service import get_ranker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup so queries are fast."""
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore")

    logger.info("Loading embedding model...")
    _get_model()
    logger.info("Loading cross-encoder ranker...")
    get_ranker()
    logger.info("Models loaded. Server ready.")
    yield


app = FastAPI(title="CLAIRE-RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    k: int = 10
    dataset: str | None = None
    use_ranker: bool = True
    retrieve_k: int | None = None
    rank_k: int | None = None
    show_chunks: bool = False
    stub: bool = False


class ChunkResponse(BaseModel):
    dataset: str
    source_id: str
    document_id: str
    text: str
    distance: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    chunks: list[ChunkResponse] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    vectorstore_path = Path("./vectorstore")
    if not vectorstore_path.exists():
        raise HTTPException(status_code=503, detail="Vectorstore not found.")

    from claire_rag.llm import answer_question, get_llm_client

    try:
        llm_client = get_llm_client(use_stub=req.stub)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM client error: {e}")

    result = answer_question(
        query=req.question,
        k=req.k,
        dataset=req.dataset,
        llm_client=llm_client,
        use_ranker=req.use_ranker,
        retrieve_k=req.retrieve_k,
        rank_k=req.rank_k,
    )

    chunks = []
    if req.show_chunks:
        for chunk in result["chunks"]:
            chunks.append(ChunkResponse(
                dataset=chunk.dataset,
                source_id=chunk.source_id,
                document_id=chunk.document_id,
                text=chunk.text,
                distance=chunk.metadata.get("distance"),
            ))

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        chunks=chunks,
    )
