"""Ranking service for re-ranking retrieved chunks using cross-encoder models."""

import logging
from typing import TYPE_CHECKING

import torch
from sentence_transformers import CrossEncoder

from claire_rag.corpus.chunking import Chunk

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# Default cross-encoder model for re-ranking
# This is a small, fast model trained on MS MARCO dataset
DEFAULT_RANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Global ranker instance (lazy loaded)
_ranker: "Ranker | None" = None


class Ranker:
    """
    Cross-encoder ranker for re-ranking retrieved chunks.

    Uses a cross-encoder model to score query-chunk pairs, providing
    more accurate relevance scores than bi-encoder similarity search.
    """

    def __init__(self, model_name: str = DEFAULT_RANKER_MODEL, debug: bool = False):
        """
        Initialize the ranker with a cross-encoder model.

        Args:
            model_name: Name of the cross-encoder model to use
            debug: If True, enable verbose logging
        """
        self.model_name = model_name
        if debug:
            logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        # Set PyTorch model to evaluation/inference mode for deterministic results
        # This is PyTorch's model.eval() method - it disables dropout, batch norm updates, etc.
        # IMPORTANT: This is NOT the same as the --eval CLI flag, which switches the LLM
        # from GPT-4o-mini to GPT-4o. This only affects the ranker model's behavior.
        self.model.model.eval()
        # Explicitly disable dropout layers for deterministic inference
        for module in self.model.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        # Configure PyTorch for deterministic operations
        # This ensures the same inputs produce the same outputs across runs
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if debug:
            logger.info("Cross-encoder model loaded successfully")

    def rank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Rank chunks by relevance to the query.

        Args:
            query: Query string
            chunks: List of Chunk objects to rank
            top_k: Optional number of top results to return. If None, returns all.

        Returns:
            List of tuples (chunk, score), ordered by score (highest first)
        """
        if not chunks:
            return []

        if not query.strip():
            # If query is empty, return chunks with zero scores
            return [(chunk, 0.0) for chunk in chunks]

        # Prepare pairs for cross-encoder: (query, chunk_text)
        pairs = [[query, chunk.text] for chunk in chunks]

        # Get scores from cross-encoder
        # Scores are relevance scores (higher = more relevant)
        # Use convert_to_tensor=False and ensure deterministic behavior
        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Combine chunks with scores
        chunk_scores = list(zip(chunks, scores))

        # Sort by score (descending), then by chunk ID for deterministic ordering
        # Use a stable sort with secondary key to ensure consistent results
        # Round scores to 6 decimal places to handle floating point precision issues
        chunk_scores.sort(
            key=lambda x: (round(x[1], 6), x[0].id),
            reverse=True,
        )

        # Return top_k if specified
        if top_k is not None:
            return chunk_scores[:top_k]

        return chunk_scores

    def rank_to_chunks(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
    ) -> list[Chunk]:
        """
        Rank chunks and return only the Chunk objects (without scores).

        Args:
            query: Query string
            chunks: List of Chunk objects to rank
            top_k: Optional number of top results to return. If None, returns all.

        Returns:
            List of Chunk objects, ordered by relevance (most relevant first)
        """
        ranked = self.rank(query, chunks, top_k=top_k)
        return [chunk for chunk, _ in ranked]


def get_ranker(model_name: str | None = None, debug: bool = False) -> Ranker:
    """
    Get or create a ranker instance (singleton).

    Args:
        model_name: Optional model name. If None, uses default.
        debug: If True, enable verbose logging

    Returns:
        Ranker instance
    """
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name=model_name or DEFAULT_RANKER_MODEL, debug=debug)
    elif model_name is not None and _ranker.model_name != model_name:
        # If different model requested, create new instance
        _ranker = Ranker(model_name=model_name, debug=debug)
    return _ranker


def rank_chunks(
    query: str,
    chunks: list[Chunk],
    top_k: int | None = None,
    ranker: Ranker | None = None,
) -> list[Chunk]:
    """
    Convenience function to rank chunks.

    Args:
        query: Query string
        chunks: List of Chunk objects to rank
        top_k: Optional number of top results to return
        ranker: Optional Ranker instance. If None, uses default singleton.

    Returns:
        List of Chunk objects, ordered by relevance (most relevant first)
    """
    if ranker is None:
        ranker = get_ranker()

    return ranker.rank_to_chunks(query, chunks, top_k=top_k)
