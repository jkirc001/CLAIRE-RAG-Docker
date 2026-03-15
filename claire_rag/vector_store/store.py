"""Vector store implementation using Chroma DB."""

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Protocol

import chromadb
from chromadb.config import Settings

from claire_rag.corpus.chunking import Chunk

if TYPE_CHECKING:
    pass


class VectorStore(Protocol):
    """Protocol for vector store implementations."""

    def upsert(self, chunks: Iterable[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks and their embeddings into the vector store."""
        ...

    def query(
        self,
        query_embedding: list[float],
        k: int = 10,
        filter_dataset: str | None = None,
    ) -> list[Chunk]:
        """Query the vector store for similar chunks."""
        ...


class LocalVectorStore:
    """
    Local vector store implementation using Chroma DB.

    Stores chunks and their embeddings in a local Chroma DB instance.
    """

    def __init__(self, persist_directory: Path | str = "./vectorstore"):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist the Chroma DB data
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        # Use cosine distance for similarity search
        self.collection = self.client.get_or_create_collection(
            name="claire_rag_chunks",
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

    def upsert(self, chunks: Iterable[Chunk], embeddings: list[list[float]]) -> None:
        """
        Upsert chunks and their embeddings into the vector store.

        Args:
            chunks: Iterable of Chunk objects
            embeddings: List of embedding vectors (same order as chunks)
        """
        chunks_list = list(chunks)

        if len(chunks_list) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks_list)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        # Helper function to convert metadata values to ChromaDB-compatible types
        def sanitize_metadata_value(value):
            """Convert metadata values to types supported by ChromaDB (str, int, float, bool, None)."""
            if value is None:
                return None
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, list):
                # Convert lists to comma-separated strings
                return ", ".join(str(item) for item in value)
            # Convert any other type to string
            return str(value)
        
        # Batch upsert to avoid ChromaDB batch size limits (max ~5461)
        batch_size = 5000
        total_chunks = len(chunks_list)
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks_list[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            batch_ids = [chunk.id for chunk in batch_chunks]
            batch_texts = [chunk.text for chunk in batch_chunks]
            batch_metadatas = [
                {
                    "dataset": chunk.dataset,
                    "source_id": chunk.source_id,
                    "document_id": chunk.document_id,
                    **{k: sanitize_metadata_value(v) for k, v in chunk.metadata.items()},
                }
                for chunk in batch_chunks
            ]
            
            # Upsert batch into Chroma
            self.collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
            )

    def query(
        self,
        query_embedding: list[float],
        k: int = 10,
        filter_dataset: str | None = None,
    ) -> list[Chunk]:
        """
        Query the vector store for similar chunks.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_dataset: Optional dataset filter (e.g., "CVE", "CWE")

        Returns:
            List of Chunk objects, ordered by similarity (most similar first)
        """
        # Build where clause for filtering
        where = None
        if filter_dataset:
            where = {"dataset": filter_dataset}

        # Query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
        )

        # Convert results to Chunk objects
        chunks = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][i]
                text = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i] if "distances" in results else None

                # Reconstruct Chunk object
                chunk = Chunk(
                    id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    dataset=metadata.get("dataset", ""),
                    source_id=metadata.get("source_id", ""),
                    text=text,
                    metadata={
                        **{k: v for k, v in metadata.items() if k not in ["dataset", "source_id", "document_id"]},
                        "distance": distance,  # Include similarity distance
                    },
                )
                chunks.append(chunk)

        # Ensure deterministic ordering by sorting by distance, then by chunk ID
        # This handles cases where ChromaDB returns results with equal distances
        chunks.sort(key=lambda c: (c.metadata.get("distance", float("inf")), c.id))

        return chunks


def get_vector_store(persist_directory: Path | str = "./vectorstore") -> LocalVectorStore:
    """
    Factory function to get a vector store instance.

    Args:
        persist_directory: Directory to persist the Chroma DB data

    Returns:
        LocalVectorStore instance
    """
    return LocalVectorStore(persist_directory=persist_directory)

