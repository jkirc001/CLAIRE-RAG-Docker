"""CVE dataset loader."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Default data path relative to project root
DATA_PATH = Path(__file__).parent.parent.parent / "data" / "cve" / "nvdcve-2.0-2024.jsonl"


@dataclass
class CveRecord:
    """Represents a CVE (Common Vulnerabilities and Exposures) record."""

    cve_id: str
    description: str
    cwes: list[str] = field(default_factory=list)
    cvss_score: float | None = None
    cvss_severity: str | None = None
    published: str | None = None
    last_modified: str | None = None
    status: str | None = None
    references: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _extract_description(data: dict) -> str:
    """Extract English description from CVE data."""
    descriptions = data.get("descriptions", [])
    for desc in descriptions:
        if desc.get("lang") == "en":
            return desc.get("value", "")
    # Fallback to first description if no English found
    if descriptions:
        return descriptions[0].get("value", "")
    return ""


def _extract_cwes(data: dict) -> list[str]:
    """Extract CWE IDs from CVE weakness data."""
    cwes = []
    weaknesses = data.get("weaknesses", [])
    for weakness in weaknesses:
        for desc in weakness.get("description", []):
            value = desc.get("value", "")
            if value.startswith("CWE-"):
                cwes.append(value)
    return cwes


def _extract_cvss(data: dict) -> tuple[float | None, str | None]:
    """Extract CVSS score and severity from metrics."""
    metrics = data.get("metrics", {})

    # Try CVSS v3.1 first
    cvss_v31 = metrics.get("cvssMetricV31", [])
    if cvss_v31:
        primary = cvss_v31[0].get("cvssData", {})
        return primary.get("baseScore"), primary.get("baseSeverity")

    # Try CVSS v3.0
    cvss_v30 = metrics.get("cvssMetricV30", [])
    if cvss_v30:
        primary = cvss_v30[0].get("cvssData", {})
        return primary.get("baseScore"), primary.get("baseSeverity")

    # Try CVSS v2
    cvss_v2 = metrics.get("cvssMetricV2", [])
    if cvss_v2:
        primary = cvss_v2[0].get("cvssData", {})
        return primary.get("baseScore"), primary.get("baseSeverity")

    return None, None


def _extract_references(data: dict) -> list[str]:
    """Extract reference URLs from CVE data."""
    refs = data.get("references", [])
    return [ref.get("url", "") for ref in refs if ref.get("url")]


def iter_cve_2024(path: Path | None = None) -> Iterable[CveRecord]:
    """
    Iterate over CVE records from the 2024 NVD dataset.

    Args:
        path: Optional path to the JSONL file. Uses default if not provided.

    Yields:
        CveRecord objects for each CVE entry.
    """
    file_path = path or DATA_PATH

    if not file_path.exists():
        raise FileNotFoundError(f"CVE data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            # Skip rejected CVEs with minimal content
            description = _extract_description(data)
            if description.startswith("Rejected reason:"):
                continue

            cve_id = data.get("id", "")
            if not cve_id:
                continue

            cvss_score, cvss_severity = _extract_cvss(data)

            yield CveRecord(
                cve_id=cve_id,
                description=description,
                cwes=_extract_cwes(data),
                cvss_score=cvss_score,
                cvss_severity=cvss_severity,
                published=data.get("published"),
                last_modified=data.get("lastModified"),
                status=data.get("vulnStatus"),
                references=_extract_references(data),
                metadata={
                    "sourceIdentifier": data.get("sourceIdentifier"),
                    "cveTags": data.get("cveTags", []),
                },
            )

