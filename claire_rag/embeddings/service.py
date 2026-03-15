"""Embedding service for text vectorization."""

from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from typing import Any

# Default model (per PRD requirements)
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Global model instance (lazy loaded)
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Get or create the embedding model instance (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(DEFAULT_MODEL)
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        batch_size: Batch size for processing (default: 32)

    Returns:
        List of embedding vectors (each is a list of floats)

    Example:
        >>> embeddings = embed_texts(["Hello world", "How are you?"])
        >>> len(embeddings)
        2
        >>> len(embeddings[0])
        384
    """
    if not texts:
        return []

    model = _get_model()

    # Filter out empty strings
    non_empty_texts = [text for text in texts if text.strip()]

    if not non_empty_texts:
        # Return empty embeddings for empty input
        return [[] for _ in texts]

    # Generate embeddings with batching
    embeddings = model.encode(
        non_empty_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # Convert numpy arrays to lists of floats
    embeddings_list = embeddings.tolist()

    # Handle case where some texts were empty
    if len(embeddings_list) < len(texts):
        # Reconstruct full list with empty embeddings for empty texts
        result = []
        text_idx = 0
        for text in texts:
            if text.strip():
                result.append(embeddings_list[text_idx])
                text_idx += 1
            else:
                result.append([])
        return result

    return embeddings_list

