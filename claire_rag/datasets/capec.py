"""CAPEC dataset loader."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Default data path relative to project root
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "capec" / "capec-dictionary.jsonl"


@dataclass
class CapecRecord:
    """Represents a CAPEC (Common Attack Pattern Enumeration and Classification) record."""

    capec_id: str
    name: str
    description: str
    abstraction: str | None = None
    status: str | None = None
    likelihood: str | None = None
    severity: str | None = None
    prerequisites: list[str] = field(default_factory=list)
    mitigations: str = ""
    example_instances: str = ""
    execution_flow: str = ""
    related_weaknesses: list[str] = field(default_factory=list)
    related_attack_patterns: str = ""
    metadata: dict = field(default_factory=dict)


def _parse_prerequisites(prereq_str: str) -> list[str]:
    """Parse prerequisites string into list."""
    if not prereq_str:
        return []
    # Split by :: delimiter used in CAPEC data
    parts = prereq_str.split("::")
    return [p.strip() for p in parts if p.strip()]


def _parse_related_weaknesses(weaknesses_str: str) -> list[str]:
    """Parse related weaknesses string into list of CWE IDs."""
    if not weaknesses_str:
        return []
    # The format is like "::276::285::434::"
    parts = weaknesses_str.split("::")
    cwes = []
    for p in parts:
        p = p.strip()
        if p and p.isdigit():
            cwes.append(f"CWE-{p}")
    return cwes


def iter_capec(path: Path | None = None) -> Iterable[CapecRecord]:
    """
    Iterate over CAPEC records from the CAPEC dictionary.

    Args:
        path: Optional path to the JSONL file. Uses default if not provided.

    Yields:
        CapecRecord objects for each CAPEC entry.
    """
    file_path = path or DATA_PATH

    if not file_path.exists():
        raise FileNotFoundError(f"CAPEC data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            capec_id = data.get("ID", "")
            if not capec_id:
                continue

            # Format CAPEC ID consistently
            capec_id_formatted = f"CAPEC-{capec_id}" if not str(capec_id).startswith("CAPEC-") else str(capec_id)

            yield CapecRecord(
                capec_id=capec_id_formatted,
                name=data.get("Name", ""),
                description=data.get("Description", ""),
                abstraction=data.get("Abstraction"),
                status=data.get("Status"),
                likelihood=data.get("Likelihood_of_Attack"),
                severity=data.get("Typical_Severity"),
                prerequisites=_parse_prerequisites(data.get("Prerequisites", "")),
                mitigations=data.get("Mitigations", ""),
                example_instances=data.get("Example Instances", ""),
                execution_flow=data.get("Execution_Flow", ""),
                related_weaknesses=_parse_related_weaknesses(data.get("Related Weaknesses", "")),
                related_attack_patterns=data.get("Related Attack Patterns", ""),
                metadata={
                    "skills_required": data.get("Skills Required", ""),
                    "resources_required": data.get("Resources Required", ""),
                    "consequences": data.get("Consequences", ""),
                    "indicators": data.get("Indicators", ""),
                    "taxonomy_mappings": data.get("Taxonomy_Mappings", ""),
                    "notes": data.get("Notes", ""),
                },
            )

