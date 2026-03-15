"""Document builder - converts dataset records to unified Document format."""

import json
from pathlib import Path
from typing import Iterable

from claire_rag.corpus.models import Document
from claire_rag.datasets import (
    iter_attack,
    iter_capec,
    iter_cve_2024,
    iter_cwe,
    iter_dcwf,
    iter_nice,
)


def _build_cve_document(record) -> Document:
    """Convert CVE record to Document."""
    doc_id = f"CVE:{record.cve_id}"
    title = record.cve_id

    # Build body from description and related information
    body_parts = [record.description]
    if record.cwes:
        body_parts.append(f"\nRelated CWEs: {', '.join(record.cwes)}")
    if record.cvss_score:
        body_parts.append(
            f"\nCVSS Score: {record.cvss_score} ({record.cvss_severity or 'Unknown'})"
        )

    body = "\n".join(body_parts)

    return Document(
        id=doc_id,
        dataset="CVE",
        source_id=record.cve_id,
        title=title,
        body=body,
        metadata={
            "cvss_score": record.cvss_score,
            "cvss_severity": record.cvss_severity,
            "cwes": record.cwes,
            "published": record.published,
            "last_modified": record.last_modified,
            "status": record.status,
            "references": record.references,
            **record.metadata,
        },
    )


def _build_cwe_document(record) -> Document:
    """Convert CWE record to Document."""
    doc_id = f"CWE:{record.cwe_id}"
    title = f"{record.cwe_id}: {record.name}"

    # Build body from description, extended description, and other fields
    body_parts = [record.description]
    if record.extended_description:
        body_parts.append(f"\n{record.extended_description}")
    if record.consequences:
        body_parts.append(f"\nCommon Consequences: {'; '.join(record.consequences)}")
    if record.mitigations:
        body_parts.append(f"\nPotential Mitigations: {'; '.join(record.mitigations)}")

    body = "\n".join(body_parts)

    return Document(
        id=doc_id,
        dataset="CWE",
        source_id=record.cwe_id,
        title=title,
        body=body,
        metadata={
            "abstraction": record.abstraction,
            "status": record.status,
            "platforms": record.platforms,
            "consequences": record.consequences,
            "mitigations": record.mitigations,
            "detection_methods": record.detection_methods,
            "related_weaknesses": record.related_weaknesses,
            **record.metadata,
        },
    )


def _build_capec_document(record) -> Document:
    """Convert CAPEC record to Document."""
    doc_id = f"CAPEC:{record.capec_id}"
    title = f"{record.capec_id}: {record.name}"

    # Build body from description, execution flow, prerequisites, mitigations
    body_parts = []
    if record.description:
        body_parts.append(record.description)
    if record.execution_flow:
        body_parts.append(f"\nExecution Flow: {record.execution_flow}")
    if record.prerequisites:
        body_parts.append(f"\nPrerequisites: {'; '.join(record.prerequisites)}")
    if record.mitigations:
        body_parts.append(f"\nMitigations: {record.mitigations}")

    # Fallback to name if no body content
    body = "\n".join(body_parts) if body_parts else record.name

    return Document(
        id=doc_id,
        dataset="CAPEC",
        source_id=record.capec_id,
        title=title,
        body=body,
        metadata={
            "abstraction": record.abstraction,
            "status": record.status,
            "likelihood": record.likelihood,
            "severity": record.severity,
            "prerequisites": record.prerequisites,
            "related_weaknesses": record.related_weaknesses,
            "related_attack_patterns": record.related_attack_patterns,
            **record.metadata,
        },
    )


def _build_attack_document(record) -> Document:
    """Convert ATT&CK record to Document."""
    # Use attack_id if available, otherwise stix_id
    source_id = record.attack_id if record.attack_id else record.stix_id
    doc_id = f"ATTACK:{source_id}"
    title = record.name

    # Build body from description, tactics, platforms, aliases
    body_parts = [record.description]
    if record.tactics:
        body_parts.append(f"\nTactics: {', '.join(record.tactics)}")
    if record.platforms:
        body_parts.append(f"\nPlatforms: {', '.join(record.platforms)}")
    if record.aliases:
        body_parts.append(f"\nAliases: {', '.join(record.aliases)}")

    body = "\n".join(body_parts)

    return Document(
        id=doc_id,
        dataset="ATTACK",
        source_id=source_id,
        title=title,
        body=body,
        metadata={
            "object_type": record.object_type,
            "stix_id": record.stix_id,
            "attack_id": record.attack_id,
            "tactics": record.tactics,
            "platforms": record.platforms,
            "aliases": record.aliases,
            "is_subtechnique": record.is_subtechnique,
            **record.metadata,
        },
    )


def _build_nice_document(record) -> Document:
    """Convert NICE record to Document."""
    doc_id = f"NICE:{record.nice_id}"
    title = record.title or record.nice_id

    # Build body from text and title
    body_parts = []
    if record.title:
        body_parts.append(record.title)
    if record.text:
        body_parts.append(record.text)

    # Fallback to nice_id if no body content
    body = "\n".join(body_parts) if body_parts else record.nice_id

    return Document(
        id=doc_id,
        dataset="NICE",
        source_id=record.nice_id,
        title=title,
        body=body,
        metadata={
            "element_type": record.element_type,
            "doc_identifier": record.doc_identifier,
            **record.metadata,
        },
    )


def _build_dcwf_document(record) -> Document:
    """Convert DCWF record to Document."""
    doc_id = f"DCWF:{record.dcwf_id}"
    title = record.work_role

    # Build body from definition, element, and work role
    body_parts = [f"Work Role: {record.work_role}"]
    if record.element:
        body_parts.append(f"Category: {record.element}")
    if record.definition:
        body_parts.append(f"\n{record.definition}")

    body = "\n".join(body_parts)

    return Document(
        id=doc_id,
        dataset="DCWF",
        source_id=record.dcwf_id,
        title=title,
        body=body,
        metadata={
            "ncwf_id": record.ncwf_id,
            "element": record.element,
            **record.metadata,
        },
    )


def build_documents() -> Iterable[Document]:
    """
    Build unified Document objects from all dataset loaders.

    Yields:
        Document objects from all datasets (CVE, CWE, CAPEC, ATT&CK, NICE, DCWF)
    """
    # CVE
    for record in iter_cve_2024():
        yield _build_cve_document(record)

    # CWE
    for record in iter_cwe():
        yield _build_cwe_document(record)

    # CAPEC
    for record in iter_capec():
        yield _build_capec_document(record)

    # ATT&CK
    for record in iter_attack():
        yield _build_attack_document(record)

    # NICE
    for record in iter_nice():
        yield _build_nice_document(record)

    # DCWF
    for record in iter_dcwf():
        yield _build_dcwf_document(record)


def save_documents(documents: Iterable[Document], output_path: Path) -> None:
    """
    Save documents to JSONL file.

    Args:
        documents: Iterable of Document objects
        output_path: Path to output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            json_line = json.dumps(
                {
                    "id": doc.id,
                    "dataset": doc.dataset,
                    "source_id": doc.source_id,
                    "title": doc.title,
                    "body": doc.body,
                    "metadata": doc.metadata,
                },
                ensure_ascii=False,
            )
            f.write(json_line + "\n")


def build_and_save(output_path: Path | None = None) -> Path:
    """
    Build all documents and save to JSONL file.

    Args:
        output_path: Optional output path. Defaults to ./artifacts/corpus/documents.jsonl

    Returns:
        Path to the saved file
    """
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent.parent
            / "artifacts"
            / "corpus"
            / "documents.jsonl"
        )

    documents = build_documents()
    save_documents(documents, output_path)

    return output_path
