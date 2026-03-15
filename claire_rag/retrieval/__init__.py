"""Retrieval service for RAG queries."""

from claire_rag.retrieval.service import embed_query, retrieve, retrieve_and_rank

__all__ = ["embed_query", "retrieve", "retrieve_and_rank"]
