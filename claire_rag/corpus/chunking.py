"""Document chunking utilities."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import tiktoken

from claire_rag.corpus.models import Document

# Use cl100k_base encoding (GPT-3.5/GPT-4 tokenizer)
# This is a reasonable approximation for token counting
ENCODING = tiktoken.get_encoding("cl100k_base")

# Chunking parameters (per PRD)
CHUNK_SIZE_TOKENS = 768
OVERLAP_TOKENS = 100


@dataclass
class Chunk:
    """
    Represents a chunk of text from a document.

    Attributes:
        id: Unique chunk identifier (e.g., "CVE:CVE-2024-12345#0")
        document_id: Parent Document.id
        dataset: Dataset name
        source_id: Original source identifier
        text: Text segment to embed
        metadata: Additional metadata including chunk index
    """

    id: str
    document_id: str
    dataset: str
    source_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(ENCODING.encode(text))


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using simple regex.

    This is a basic sentence splitter that handles common cases.
    """
    # Pattern to match sentence endings (., !, ?) followed by space or end of string
    # Also handles cases with quotes, parentheses, etc.
    pattern = r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$"
    sentences = re.split(pattern, text)
    # Filter out empty strings
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(
    text: str,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    """
    Chunk text into segments of approximately chunk_size_tokens with overlap.

    Args:
        text: Text to chunk
        chunk_size_tokens: Target chunk size in tokens
        overlap_tokens: Overlap size in tokens

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    # Count total tokens
    total_tokens = _count_tokens(text)

    # If text fits in one chunk, return as-is
    if total_tokens <= chunk_size_tokens:
        return [text]

    chunks = []
    sentences = _split_into_sentences(text)

    if not sentences:
        # Fallback: if sentence splitting fails, split by characters
        # This should rarely happen, but provides a safety net
        char_chunk_size = chunk_size_tokens * 4  # Rough estimate: ~4 chars per token
        char_overlap = overlap_tokens * 4
        for i in range(0, len(text), char_chunk_size - char_overlap):
            chunk = text[i : i + char_chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    current_chunk_sentences = []
    current_tokens = 0
    overlap_sentences = []

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)

        # If a single sentence exceeds chunk size, split it by characters
        if sentence_tokens > chunk_size_tokens:
            # Save current chunk if we have one
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_tokens = 0

            # Split the long sentence
            char_chunk_size = chunk_size_tokens * 4
            char_overlap = overlap_tokens * 4
            for i in range(0, len(sentence), char_chunk_size - char_overlap):
                chunk = sentence[i : i + char_chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
            continue

        # Check if adding this sentence would exceed chunk size
        if (
            current_tokens + sentence_tokens > chunk_size_tokens
            and current_chunk_sentences
        ):
            # Save current chunk
            chunks.append(" ".join(current_chunk_sentences))

            # Prepare overlap: keep last sentences that fit in overlap size
            overlap_tokens_used = 0
            overlap_sentences = []
            for sent in reversed(current_chunk_sentences):
                sent_tokens = _count_tokens(sent)
                if overlap_tokens_used + sent_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, sent)
                    overlap_tokens_used += sent_tokens
                else:
                    break

            # Start new chunk with overlap
            current_chunk_sentences = overlap_sentences.copy()
            current_tokens = overlap_tokens_used

        # Add sentence to current chunk
        current_chunk_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Add final chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


def chunk_document(document: Document) -> Iterable[Chunk]:
    """
    Chunk a single document into Chunk objects.

    Args:
        document: Document to chunk

    Yields:
        Chunk objects
    """
    chunks = _chunk_text(document.body)

    for idx, chunk_text in enumerate(chunks):
        chunk_id = f"{document.id}#{idx}"

        yield Chunk(
            id=chunk_id,
            document_id=document.id,
            dataset=document.dataset,
            source_id=document.source_id,
            text=chunk_text,
            metadata={
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "document_title": document.title,
                **document.metadata,
            },
        )


def chunk_documents(documents: Iterable[Document]) -> Iterable[Chunk]:
    """
    Chunk multiple documents into Chunk objects.

    Args:
        documents: Iterable of Document objects

    Yields:
        Chunk objects from all documents
    """
    for document in documents:
        yield from chunk_document(document)


def save_chunks(chunks: Iterable[Chunk], output_path: Path) -> None:
    """
    Save chunks to JSONL file.

    Args:
        chunks: Iterable of Chunk objects
        output_path: Path to output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json_line = json.dumps(
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "dataset": chunk.dataset,
                    "source_id": chunk.source_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                },
                ensure_ascii=False,
            )
            f.write(json_line + "\n")


def chunk_and_save_documents(
    documents: Iterable[Document],
    output_path: Path | None = None,
) -> Path:
    """
    Chunk documents and save to JSONL file.

    Args:
        documents: Iterable of Document objects
        output_path: Optional output path. Defaults to ./artifacts/corpus/chunks.jsonl

    Returns:
        Path to the saved file
    """
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent.parent
            / "artifacts"
            / "corpus"
            / "chunks.jsonl"
        )

    chunks = chunk_documents(documents)
    save_chunks(chunks, output_path)

    return output_path
