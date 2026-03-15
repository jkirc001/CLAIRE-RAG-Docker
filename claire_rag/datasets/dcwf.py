"""DCWF (DoD Cyber Workforce Framework) dataset loader."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Default data path relative to project root
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "dcwf" / "dcwf_roles.jsonl"


@dataclass
class DcwfRecord:
    """Represents a DCWF (DoD Cyber Workforce Framework) role record."""

    dcwf_id: str  # DCWF code
    ncwf_id: str  # NCWF identifier
    work_role: str  # Role name/title
    element: str  # Category/element (e.g., "Information Technology (IT)")
    definition: str  # Role definition/description
    metadata: dict = field(default_factory=dict)


def iter_dcwf(path: Path | None = None) -> Iterable[DcwfRecord]:
    """
    Iterate over DCWF role records.

    Only loads from dcwf_roles.jsonl (per PRD requirements).
    Excludes the detailed task/KSA files.

    Args:
        path: Optional path to the JSONL file. Uses default if not provided.

    Yields:
        DcwfRecord objects for each DCWF role.
    """
    file_path = path or DATA_PATH

    if not file_path.exists():
        raise FileNotFoundError(f"DCWF data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            dcwf_code = data.get("dcwf_code", "")
            if not dcwf_code:
                continue

            # Format DCWF ID consistently
            dcwf_id = f"DCWF-{dcwf_code}" if not str(dcwf_code).startswith("DCWF-") else str(dcwf_code)

            yield DcwfRecord(
                dcwf_id=dcwf_id,
                ncwf_id=data.get("ncwf_id", ""),
                work_role=data.get("work_role", ""),
                element=data.get("element", "").replace("\n", " ").strip(),
                definition=data.get("definition", ""),
                metadata={
                    "raw": data,
                },
            )

