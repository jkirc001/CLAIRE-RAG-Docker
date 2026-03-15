"""Retrieval service for RAG queries."""

from claire_rag.corpus.chunking import Chunk
from claire_rag.embeddings import embed_texts
from claire_rag.ranking import Ranker, rank_chunks
from claire_rag.vector_store import VectorStore, get_vector_store


def embed_query(query: str) -> list[float]:
    """
    Embed a query string using the same embedding model as document chunks.

    Args:
        query: Query string to embed

    Returns:
        Query embedding vector (list of floats)

    Example:
        >>> embedding = embed_query("buffer overflow vulnerability")
        >>> len(embedding)
        384
    """
    # Trim whitespace (basic preprocessing)
    query = query.strip()

    # Use the same embedding service as documents
    embeddings = embed_texts([query])

    if not embeddings or not embeddings[0]:
        # Return zero vector if embedding fails
        return [0.0] * 384

    return embeddings[0]


def retrieve(
    query: str,
    k: int = 10,
    dataset: str | None = None,
    vector_store: VectorStore | None = None,
) -> list[Chunk]:
    """
    Retrieve relevant chunks for a query.

    Args:
        query: Query string
        k: Number of chunks to retrieve (default: 10)
        dataset: Optional dataset filter (e.g., "CVE", "CWE")
        vector_store: Optional vector store instance. If None, uses default.

    Returns:
        List of Chunk objects, ordered by similarity (most similar first)

    Example:
        >>> chunks = retrieve("buffer overflow", k=5)
        >>> len(chunks)
        5
        >>> all(isinstance(c, Chunk) for c in chunks)
        True
    """
    # Get vector store if not provided
    if vector_store is None:
        vector_store = get_vector_store()

    # Embed the query
    query_embedding = embed_query(query)

    # Query the vector store
    chunks = vector_store.query(
        query_embedding=query_embedding,
        k=k,
        filter_dataset=dataset,
    )

    return chunks


def retrieve_and_rank(
    query: str,
    retrieve_k: int = 50,
    rank_k: int = 10,
    dataset: str | None = None,
    vector_store: VectorStore | None = None,
    ranker: Ranker | None = None,
    debug: bool = False,
) -> list[Chunk]:
    """
    Retrieve and re-rank chunks using a two-stage approach.

    This function:
    1. Retrieves a larger set of candidate chunks (retrieve_k) using vector search
    2. Re-ranks them using a cross-encoder model
    3. Returns the top rank_k chunks

    Args:
        query: Query string
        retrieve_k: Number of chunks to retrieve initially (default: 50)
        rank_k: Number of chunks to return after ranking (default: 10)
        dataset: Optional dataset filter (e.g., "CVE", "CWE")
        vector_store: Optional vector store instance. If None, uses default.
        ranker: Optional Ranker instance. If None, uses default singleton.
        debug: If True, enable verbose logging for ranker initialization.

    Returns:
        List of Chunk objects, ordered by relevance (most relevant first)

    Example:
        >>> chunks = retrieve_and_rank("buffer overflow", retrieve_k=50, rank_k=10)
        >>> len(chunks)
        10
        >>> all(isinstance(c, Chunk) for c in chunks)
        True
    """
    # Step 1: Retrieve candidate chunks using vector search
    candidate_chunks = retrieve(
        query=query,
        k=retrieve_k,
        dataset=dataset,
        vector_store=vector_store,
    )

    # Step 2: Re-rank using cross-encoder
    # If ranker not provided, get one with debug flag
    if ranker is None:
        from claire_rag.ranking import get_ranker
        ranker = get_ranker(debug=debug)
    
    ranked_chunks = rank_chunks(
        query=query,
        chunks=candidate_chunks,
        top_k=rank_k,
        ranker=ranker,
    )

    return ranked_chunks

