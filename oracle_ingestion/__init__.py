"""
Utilities for ingesting Oracle Content Management manifests into the
Secure RAG retrieval pipeline.

This package is intentionally lightweight so it can be executed from
CLI scripts or background jobs without importing the full Flask/Rasa
stack.  The manifest format is expected to be JSON Lines (one JSON
object per line) containing, at minimum, the following fields:

    {
        "doc_id": "unique-id",
        "source_path": "/path/to/file.ext",
        "language": "en",
        "title": "Document title",
        "summary_hint": "Optional short summary",
        "security_groups": ["group-a", "group-b"],
        "tags": ["optional", "labels"],
        "last_modified": "2025-02-14T10:01:00Z",
        "checksum": "sha256:..."
    }

Additional keys are preserved in the metadata passed to the vector
store, making it easy to extend the schema for future needs.
"""

from .manifest_ingestor import OracleManifestIngestor, ManifestEntry, embed_preview

__all__ = ["OracleManifestIngestor", "ManifestEntry", "embed_preview"]
