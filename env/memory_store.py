"""FAISS-backed memory store (plan §1.3).

Wraps a flat inner-product FAISS index over normalized
`sentence-transformers/all-MiniLM-L6-v2` embeddings (so inner product == cosine).
The benign corpus written by `data/build_benign_corpus.py` already stores its
embeddings inline, so `from_corpus` reuses them and only loads the embedder
lazily — which matters for episode runtime since episode #N injects fresh
payloads via `ingest`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384


@dataclass
class MemoryEntry:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


def _load_default_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


class MemoryStore:
    def __init__(self, embedder=None, dim: Optional[int] = None):
        import faiss

        self._faiss = faiss
        self._embedder = embedder
        self._dim = dim or DEFAULT_DIM
        self._index = faiss.IndexFlatIP(self._dim)
        self._entries: List[MemoryEntry] = []

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = _load_default_embedder()
        return self._embedder

    def __len__(self) -> int:
        return len(self._entries)

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedder.encode([text], normalize_embeddings=True)
        return np.asarray(vec, dtype="float32").reshape(1, -1)

    def _add_vector(self, entry: MemoryEntry, vector: np.ndarray) -> None:
        if vector.shape[1] != self._dim:
            raise ValueError(f"embedding dim {vector.shape[1]} != index dim {self._dim}")
        self._index.add(vector)
        self._entries.append(entry)

    @classmethod
    def from_corpus(
        cls,
        path: str | Path,
        embedder=None,
        limit: Optional[int] = None,
    ) -> "MemoryStore":
        """Build a store from a JSONL corpus.

        Each row must have `id`, `text`, and optionally `embedding` + `metadata`
        + `persona` + `method`. Persona/method get folded into metadata.
        """
        rows: list[dict] = []
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break

        if not rows:
            raise ValueError(f"corpus at {path} is empty")

        # Use the dim of the first embedded row to size the index. If no row
        # has an embedding, fall back to the default model dim.
        dim = next(
            (len(r["embedding"]) for r in rows if r.get("embedding")),
            DEFAULT_DIM,
        )
        store = cls(embedder=embedder, dim=dim)

        # Group rows by whether they already carry an embedding.
        pre_embedded_vecs: list[list[float]] = []
        pre_embedded_entries: list[MemoryEntry] = []
        to_embed_texts: list[str] = []
        to_embed_entries: list[MemoryEntry] = []

        for row in rows:
            md = dict(row.get("metadata", {}))
            if "persona" in row:
                md.setdefault("persona", row["persona"])
            if "method" in row:
                md.setdefault("method", row["method"])
            entry = MemoryEntry(id=row["id"], text=row["text"], metadata=md)
            if row.get("embedding"):
                pre_embedded_vecs.append(row["embedding"])
                pre_embedded_entries.append(entry)
            else:
                to_embed_texts.append(row["text"])
                to_embed_entries.append(entry)

        if pre_embedded_vecs:
            arr = np.asarray(pre_embedded_vecs, dtype="float32")
            store._index.add(arr)
            store._entries.extend(pre_embedded_entries)

        if to_embed_entries:
            vecs = store.embedder.encode(to_embed_texts, normalize_embeddings=True)
            arr = np.asarray(vecs, dtype="float32")
            store._index.add(arr)
            store._entries.extend(to_embed_entries)

        return store

    def ingest(self, text: str, metadata: Optional[dict] = None) -> str:
        """Embed and add a single entry. Returns the assigned id."""
        entry_id = f"ingested_{len(self._entries):06d}"
        entry = MemoryEntry(id=entry_id, text=text, metadata=dict(metadata or {}))
        vec = self._embed(text)
        self._add_vector(entry, vec)
        return entry_id

    def query(self, text: str, k: int = 5) -> List[MemoryEntry]:
        if len(self._entries) == 0:
            return []
        k = min(k, len(self._entries))
        vec = self._embed(text)
        scores, idxs = self._index.search(vec, k)
        out: List[MemoryEntry] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            base = self._entries[idx]
            out.append(
                MemoryEntry(
                    id=base.id,
                    text=base.text,
                    metadata=base.metadata,
                    score=float(score),
                )
            )
        return out
