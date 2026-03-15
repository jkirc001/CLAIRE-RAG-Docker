"""CWE dataset loader."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Default data path relative to project root
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "cwe" / "cwec_v4.18.jsonl"


@dataclass
class CweRecord:
    """Represents a CWE (Common Weakness Enumeration) record."""

    cwe_id: str
    name: str
    description: str
    extended_description: str = ""
    abstraction: str | None = None
    status: str | None = None
    platforms: list[str] = field(default_factory=list)
    consequences: list[str] = field(default_factory=list)
    mitigations: list[str] = field(default_factory=list)
    detection_methods: list[str] = field(default_factory=list)
    related_weaknesses: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def iter_cwe(path: Path | None = None) -> Iterable[CweRecord]:
    """
    Iterate over CWE records from the CWE dictionary.

    Args:
        path: Optional path to the JSONL file. Uses default if not provided.

    Yields:
        CweRecord objects for each CWE entry.
    """
    file_path = path or DATA_PATH

    if not file_path.exists():
        raise FileNotFoundError(f"CWE data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            cwe_id = data.get("ID", "")
            if not cwe_id:
                continue

            # Format CWE ID consistently
            cwe_id_formatted = f"CWE-{cwe_id}" if not cwe_id.startswith("CWE-") else cwe_id

            yield CweRecord(
                cwe_id=cwe_id_formatted,
                name=data.get("Name", ""),
                description=data.get("Description", ""),
                extended_description=data.get("Extended_Description", "").strip(),
                abstraction=data.get("Abstraction"),
                status=data.get("Status"),
                platforms=data.get("Applicable_Platforms", []),
                consequences=data.get("Common_Consequences", []),
                mitigations=data.get("Potential_Mitigations", []),
                detection_methods=data.get("Detection_Methods", []),
                related_weaknesses=data.get("Related_Weaknesses", []),
                metadata={
                    "modes_of_introduction": data.get("Modes_of_Introduction", []),
                    "alternate_terms": data.get("Alternate_Terms", []),
                    "observed_examples": data.get("Observed_Examples", []),
                    "notes": data.get("Notes", ""),
                    "background_details": data.get("Background_Details", ""),
                },
            )

