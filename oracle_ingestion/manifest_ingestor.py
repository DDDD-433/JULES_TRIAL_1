from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import requests

from secure_rag.embeddings import embed_texts, EMBEDDING_DIM
from shared_utils import FileService, MessageFormatter

LOGGER = logging.getLogger("oracle_manifest_ingestor")


@dataclass
class ManifestEntry:
    """Representation of a single manifest row."""

    doc_id: str
    source_path: str
    language: str
    title: str
    summary_hint: Optional[str]
    security_groups: List[str]
    tags: List[str]
    last_modified: Optional[str]
    checksum: Optional[str]
    extra: Dict[str, object]

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "ManifestEntry":
        doc_id = str(payload.get("doc_id") or payload.get("id") or "").strip()
        if not doc_id:
            raise ValueError("manifest row missing 'doc_id'")
        source_path = str(payload.get("source_path") or payload.get("path") or "").strip()
        if not source_path:
            raise ValueError(f"manifest row {doc_id!r} missing 'source_path'")
        language = str(payload.get("language") or "en").strip().lower()
        title = str(payload.get("title") or doc_id).strip()
        summary_hint = payload.get("summary_hint")
        if summary_hint is not None:
            summary_hint = str(summary_hint).strip() or None
        security_groups = payload.get("security_groups") or payload.get("groups") or []
        if isinstance(security_groups, str):
            security_groups = [security_groups]
        security_groups = [str(group).strip().lower() for group in security_groups if str(group).strip()]
        tags = payload.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        tags = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
        last_modified = payload.get("last_modified") or payload.get("modified_at")
        if last_modified is not None:
            last_modified = str(last_modified).strip() or None
        checksum = payload.get("checksum")
        if checksum is not None:
            checksum = str(checksum).strip() or None

        known_keys = {
            "doc_id",
            "id",
            "source_path",
            "path",
            "language",
            "title",
            "summary_hint",
            "security_groups",
            "groups",
            "tags",
            "last_modified",
            "modified_at",
            "checksum",
            "text",
        }
        extra = {k: v for k, v in payload.items() if k not in known_keys}
        return cls(
            doc_id=doc_id,
            source_path=source_path,
            language=language,
            title=title or doc_id,
            summary_hint=summary_hint,
            security_groups=security_groups,
            tags=tags,
            last_modified=last_modified,
            checksum=checksum,
            extra=extra,
        )


class OracleManifestIngestor:
    """Stream documents from a JSONL manifest into the Secure RAG vector service."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        namespace: str = "oracle_docs",
        chunk_words: int = 220,
        chunk_overlap: int = 40,
        vector_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        backend: str = "auto",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.namespace = namespace
        self.chunk_words = max(40, chunk_words)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_words // 2))
        self.vector_base_url = (vector_base_url or os.getenv("VECTOR_BASE_URL", "http://localhost:8001")).rstrip("/")
        self.api_key = api_key or os.getenv("VECTOR_API_KEY", "").strip()
        self.backend = backend
        self.file_service = FileService()

    def ingest(self, limit: Optional[int] = None) -> int:
        """Ingest the manifest into the retrieval service.

        Args:
            limit: Optional cap on the number of manifest entries processed.

        Returns:
            Total number of vector items successfully uploaded.
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest file not found: {self.manifest_path}")

        processed = 0
        uploaded = 0

        for entry in self._iter_manifest(limit=limit):
            processed += 1
            try:
                text = self._load_entry_text(entry)
            except Exception as exc:
                LOGGER.warning("Skipping %s due to load error: %s", entry.doc_id, exc)
                continue

            chunks = self._chunk_text(text)
            if not chunks:
                LOGGER.debug("Skipping %s because no chunks produced", entry.doc_id)
                continue

            metadata_base = {
                "doc_id": entry.doc_id,
                "title": entry.title,
                "language": entry.language,
                "summary": entry.summary_hint or (chunks[0][:200] if chunks else ""),
                "source_path": entry.source_path,
                "security_groups": entry.security_groups,
                "tags": entry.tags,
                "last_modified": entry.last_modified,
                "checksum": entry.checksum,
            }
            metadata_base.update(entry.extra)

            items = []
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{entry.doc_id}::chunk-{idx:04d}"
                items.append(
                    {
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": {
                            **metadata_base,
                            "chunk_index": idx,
                            "chunk_word_count": len(chunk.split()),
                        },
                    }
                )

            uploaded += self._upsert_items(items)

        LOGGER.info(
            "Oracle manifest ingestion complete | processed=%s uploaded=%s namespace=%s",
            processed,
            uploaded,
            self.namespace,
        )
        return uploaded

    def _iter_manifest(self, *, limit: Optional[int] = None) -> Iterator[ManifestEntry]:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, 1):
                if limit is not None and line_no > limit:
                    break
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                    entry = ManifestEntry.from_json(payload)
                    yield entry
                except Exception as exc:
                    LOGGER.warning("Invalid manifest row on line %s: %s", line_no, exc)

    def _load_entry_text(self, entry: ManifestEntry) -> str:
        """Load text content for a manifest entry."""
        # Inline text support (helps for quick POCs)
        if "text" in entry.extra:
            inline = str(entry.extra["text"]).strip()
            if inline:
                return inline

        path = Path(entry.source_path)
        if not path.is_absolute():
            path = self.manifest_path.parent / path

        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".json"}:
            return path.read_text(encoding="utf-8", errors="ignore")

        # Reuse FileService extraction utilities for richer document types.
        if suffix in {".pdf", ".docx"}:
            return self.file_service._extract_document_text(path)  # type: ignore[attr-defined]

        if suffix in {".html", ".htm"}:
            return path.read_text(encoding="utf-8", errors="ignore")

        raise ValueError(f"Unsupported file type for ingestion: {path}")

    def _chunk_text(self, text: str) -> List[str]:
        """Break text into overlapping word windows."""
        cleaned = MessageFormatter.clean_markdown_text(text or "")
        words = cleaned.split()
        if not words:
            return []

        chunks: List[str] = []
        step = self.chunk_words - self.chunk_overlap
        if step <= 0:
            step = self.chunk_words

        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_words]
            if not window:
                break
            chunk = " ".join(window).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _upsert_items(self, items: Sequence[Dict[str, object]]) -> int:
        if not items:
            return 0
        endpoint = f"{self.vector_base_url}/upsert"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"namespace": self.namespace, "backend": self.backend, "items": items}
        response = requests.post(endpoint, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return int(data.get("count", 0))


def embed_preview(chunks: Iterable[str]) -> List[List[float]]:
    """Return preview embeddings for diagnostic purposes."""
    vectors = embed_texts(chunks, dim=EMBEDDING_DIM)
    return vectors.round(6).tolist()


__all__ = ["OracleManifestIngestor", "ManifestEntry", "embed_preview"]
