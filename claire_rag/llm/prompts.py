"""Prompt building utilities for RAG."""

from claire_rag.corpus.chunking import Chunk


def build_context_prompt(query: str, chunks: list[Chunk]) -> str:
    """
    Build a prompt with query and context chunks.

    Args:
        query: User query
        chunks: List of retrieved Chunk objects

    Returns:
        Formatted prompt string

    Example:
        >>> chunks = [Chunk(id="1", document_id="doc1", dataset="CVE", source_id="CVE-2024-12345", text="Test", metadata={})]
        >>> prompt = build_context_prompt("What is a buffer overflow?", chunks)
        >>> "buffer overflow" in prompt.lower()
        True
    """
    prompt_parts = [
        "Question:",
        query,
        "",
    ]

    if chunks:
        prompt_parts.extend(
            [
                "Context:",
            ]
        )

        for idx, chunk in enumerate(chunks, start=1):
            dataset = chunk.dataset
            source_id = chunk.source_id
            text = chunk.text

            prompt_parts.append(f"[{idx}] ({dataset} {source_id})")
            prompt_parts.append(text)
            prompt_parts.append("")  # Empty line between chunks

        prompt_parts.append(
            "Answer the question based on the context above. If the context does not contain enough information to answer, use your general knowledge."
        )
    else:
        prompt_parts.append(
            "Answer questions based on general knowledge when no database results are available."
        )

    return "\n".join(prompt_parts)
