"""CLI for building the RAG index."""

import logging
import sys
from pathlib import Path

import typer

from claire_rag.corpus.build import build_documents
from claire_rag.corpus.chunking import chunk_and_save_documents
from claire_rag.embeddings import embed_texts
from claire_rag.vector_store import get_vector_store

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def build(
    rebuild_documents: bool = typer.Option(
        False,
        "--rebuild",
        help="Rebuild documents from raw datasets instead of loading from artifacts",
    ),
    eval_mode: bool = typer.Option(
        False,
        "--eval",
        help="Enable evaluation mode (forces gpt-4o for LLM)",
    ),
) -> None:
    """
    Build the RAG index.

    This command:
    1. Loads or rebuilds all Document objects
    2. Chunks all documents into Chunk objects
    3. Embeds all chunks using the embedding model
    4. Upserts all chunks and embeddings into the vector store

    The index is stored in ./vectorstore by default.
    """
    if eval_mode:
        import os

        os.environ["CLAIRE_ENV"] = "evaluation"
        logger.info("Evaluation mode enabled. LLM model will be set to gpt-4o.")

    logger.info("Starting index build...")

    # Step 1: Load or rebuild documents
    if rebuild_documents:
        logger.info("Rebuilding documents from raw datasets...")
        documents = list(build_documents())
        logger.info(f"Built {len(documents):,} documents")
    else:
        # Try to load from artifacts/corpus/documents.jsonl
        documents_path = Path("artifacts/corpus/documents.jsonl")
        if documents_path.exists():
            logger.info(f"Loading documents from {documents_path}...")
            import json

            documents = []
            with open(documents_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        from claire_rag.corpus.models import Document

                        documents.append(
                            Document(
                                id=data["id"],
                                dataset=data["dataset"],
                                source_id=data["source_id"],
                                title=data["title"],
                                body=data["body"],
                                metadata=data.get("metadata", {}),
                            )
                        )
            logger.info(f"Loaded {len(documents):,} documents")
        else:
            logger.warning(
                f"Documents file not found at {documents_path}. Rebuilding from datasets..."
            )
            documents = list(build_documents())
            logger.info(f"Built {len(documents):,} documents")

    # Step 2: Chunk documents
    logger.info("Chunking documents...")
    from claire_rag.corpus.chunking import chunk_documents

    chunks = list(chunk_documents(documents))
    logger.info(f"Created {len(chunks):,} chunks")

    # Save chunks for debugging
    chunk_and_save_documents(documents)

    # Step 3: Embed chunks
    logger.info("Embedding chunks...")
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(chunk_texts, batch_size=32)
    logger.info(f"Generated {len(embeddings):,} embeddings")

    # Step 4: Upsert into vector store
    logger.info("Upserting chunks into vector store...")
    vector_store = get_vector_store()
    vector_store.upsert(chunks, embeddings)
    logger.info("Index build complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    app()
