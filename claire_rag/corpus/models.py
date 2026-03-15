"""Corpus data models."""

from dataclasses import dataclass, field


@dataclass
class Document:
    """
    Unified document representation for RAG corpus.

    Attributes:
        id: Internal unique identifier (e.g., "CVE:CVE-2024-12345")
        dataset: Dataset name (one of: "CVE", "CWE", "CAPEC", "ATTACK", "NICE", "DCWF")
        source_id: Original identifier from source dataset (e.g., "CVE-2024-12345")
        title: Short name or title
        body: Full text content used for RAG (concatenation of description and other fields)
        metadata: Additional metadata dictionary
    """

    id: str
    dataset: str
    source_id: str
    title: str
    body: str
    metadata: dict = field(default_factory=dict)
