"""ATT&CK dataset loader."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Default data path relative to project root
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "attack" / "enterprise-attack.jsonl"

# Object types to include (per PRD requirements)
INCLUDED_TYPES = {"attack-pattern", "malware", "tool", "intrusion-set"}


@dataclass
class AttackRecord:
    """Represents a MITRE ATT&CK record."""

    attack_id: str  # e.g., "T1055.011", "S0061", "G0119"
    stix_id: str  # e.g., "attack-pattern--0042a9f5-..."
    object_type: str  # e.g., "attack-pattern", "malware", "tool", "intrusion-set"
    name: str
    description: str
    aliases: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    tactics: list[str] = field(default_factory=list)
    is_subtechnique: bool = False
    deprecated: bool = False
    revoked: bool = False
    metadata: dict = field(default_factory=dict)


def _extract_attack_id(data: dict) -> str:
    """Extract the ATT&CK ID (T####, S####, G####) from external references."""
    external_refs = data.get("external_references", [])
    for ref in external_refs:
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "")
    return ""


def _extract_tactics(data: dict) -> list[str]:
    """Extract tactic names from kill chain phases."""
    kill_chain = data.get("kill_chain_phases", [])
    return [phase.get("phase_name", "") for phase in kill_chain if phase.get("phase_name")]


def _extract_aliases(data: dict) -> list[str]:
    """Extract aliases from various fields."""
    aliases = []

    # Standard aliases field
    if "aliases" in data:
        aliases.extend(data["aliases"])

    # x_mitre_aliases for malware/tools
    if "x_mitre_aliases" in data:
        aliases.extend(data["x_mitre_aliases"])

    # Remove the main name from aliases to avoid duplication
    name = data.get("name", "")
    return [a for a in aliases if a and a != name]


def iter_attack(path: Path | None = None) -> Iterable[AttackRecord]:
    """
    Iterate over ATT&CK records from the Enterprise ATT&CK dataset.

    Only includes content-bearing object types:
    - attack-pattern (techniques)
    - malware
    - tool
    - intrusion-set (threat groups)

    Args:
        path: Optional path to the JSONL file. Uses default if not provided.

    Yields:
        AttackRecord objects for each ATT&CK entry.
    """
    file_path = path or DATA_PATH

    if not file_path.exists():
        raise FileNotFoundError(f"ATT&CK data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            # Filter to included object types only
            obj_type = data.get("type", "")
            if obj_type not in INCLUDED_TYPES:
                continue

            # Skip deprecated or revoked entries
            if data.get("x_mitre_deprecated", False):
                continue
            if data.get("revoked", False):
                continue

            attack_id = _extract_attack_id(data)
            stix_id = data.get("id", "")

            # Skip if no identifiable ID
            if not attack_id and not stix_id:
                continue

            # Determine if this is a subtechnique (contains a dot in technique ID)
            is_subtechnique = "." in attack_id if attack_id.startswith("T") else False

            yield AttackRecord(
                attack_id=attack_id,
                stix_id=stix_id,
                object_type=obj_type,
                name=data.get("name", ""),
                description=data.get("description", ""),
                aliases=_extract_aliases(data),
                platforms=data.get("x_mitre_platforms", []),
                tactics=_extract_tactics(data),
                is_subtechnique=is_subtechnique,
                deprecated=data.get("x_mitre_deprecated", False),
                revoked=data.get("revoked", False),
                metadata={
                    "created": data.get("created"),
                    "modified": data.get("modified"),
                    "version": data.get("x_mitre_version"),
                    "domains": data.get("x_mitre_domains", []),
                    "contributors": data.get("x_mitre_contributors", []),
                    "data_sources": data.get("x_mitre_data_sources", []),
                    "detection": data.get("x_mitre_detection"),
                },
            )

