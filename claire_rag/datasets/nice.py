"""NICE Framework dataset loader."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Default data path relative to project root
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "nice" / "nice_components.jsonl"


@dataclass
class NiceRecord:
    """Represents a NICE Framework component record."""

    nice_id: str  # Element identifier
    element_type: str  # e.g., "sort", "category", "specialty_area", "work_role", etc.
    title: str
    text: str
    doc_identifier: str = ""
    metadata: dict = field(default_factory=dict)


def iter_nice(path: Path | None = None) -> Iterable[NiceRecord]:
    """
    Iterate over NICE Framework component records.

    Args:
        path: Optional path to the JSONL file. Uses default if not provided.

    Yields:
        NiceRecord objects for each NICE component.
    """
    file_path = path or DATA_PATH

    if not file_path.exists():
        raise FileNotFoundError(f"NICE data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            element_id = data.get("element_identifier", "")
            if not element_id:
                continue

            yield NiceRecord(
                nice_id=element_id,
                element_type=data.get("element_type", ""),
                title=data.get("title", ""),
                text=data.get("text", ""),
                doc_identifier=data.get("doc_identifier", ""),
                metadata={
                    "raw": data,
                },
            )

