"""Finance memory store with TF-IDF (default) or optional MiniLM retrieval."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from .schemas import DATA_DIR, MemoryEntry, RetrievalHit, load_clean_memories


Backend = Literal["tfidf", "minilm"]
DEFAULT_BACKEND: Backend = "tfidf"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _default_backend() -> Backend:
    env = os.environ.get("FINANCE_RETRIEVER", DEFAULT_BACKEND).lower()
    if env in ("minilm", "tfidf"):
        return env  # type: ignore[return-value]
    return DEFAULT_BACKEND


class FinanceMemoryStore:
    def __init__(
        self,
        entries: Optional[list[MemoryEntry]] = None,
        backend: Optional[Backend] = None,
        embedder=None,
    ):
        self.backend: Backend = backend or _default_backend()
        self._embedder = embedder
        self._clean_entries: list[MemoryEntry] = []
        self._entries: list[MemoryEntry] = []
        self._vectors: Optional[np.ndarray] = None
        self._tfidf = None
        if entries is not None:
            self._load_entries(entries)

    @classmethod
    def from_clean_corpus(
        cls,
        path: Path | str = DATA_DIR / "clean_memories.jsonl",
        backend: Optional[Backend] = None,
        embedder=None,
    ) -> FinanceMemoryStore:
        entries = load_clean_memories(path)
        store = cls(entries=[], backend=backend, embedder=embedder)
        store._load_entries(entries)
        store._clean_entries = copy.deepcopy(store._entries)
        return store

    def _load_entries(self, entries: list[MemoryEntry]) -> None:
        self._entries = copy.deepcopy(entries)
        self._fit_index()

    def _fit_index(self) -> None:
        if not self._entries:
            self._vectors = None
            self._tfidf = None
            return
        texts = [e.text for e in self._entries]
        if self.backend == "minilm":
            vecs = self.embedder.encode(texts, normalize_embeddings=True)
            self._vectors = np.asarray(vecs, dtype="float32")
            self._tfidf = None
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(stop_words="english")
            self._vectors = self._tfidf.fit_transform(texts).astype("float32")
            self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(MINILM_MODEL)
        return self._embedder

    def __len__(self) -> int:
        return len(self._entries)

    def add_memory(self, memory: MemoryEntry) -> None:
        entry = copy.deepcopy(memory)
        entry.is_poison = False
        self._entries.append(entry)
        self._fit_index()

    def add_poison_memory(self, memory: MemoryEntry) -> None:
        entry = copy.deepcopy(memory)
        entry.is_poison = True
        self._entries.append(entry)
        self._fit_index()

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalHit]:
        if not self._entries or self._vectors is None:
            return []
        k = min(k, len(self._entries))

        if self.backend == "minilm":
            q_vec = self.embedder.encode([query], normalize_embeddings=True)
            q_vec = np.asarray(q_vec, dtype="float32").reshape(1, -1)
            scores = (self._vectors @ q_vec.T).flatten()
        else:
            q_vec = self._tfidf.transform([query])
            from sklearn.metrics.pairwise import cosine_similarity

            scores = cosine_similarity(q_vec, self._vectors).flatten()

        ranked = np.argsort(-scores)[:k]
        hits: list[RetrievalHit] = []
        for rank, idx in enumerate(ranked, start=1):
            entry = self._entries[int(idx)]
            value = entry.poison_value if entry.is_poison else entry.true_value
            hits.append(
                RetrievalHit(
                    memory_id=entry.memory_id,
                    rank=rank,
                    score=float(scores[int(idx)]),
                    text=entry.text,
                    fact_id=entry.fact_id,
                    is_poison=entry.is_poison,
                    value=value,
                )
            )
        return hits

    def reset_to_clean(self) -> None:
        self._entries = copy.deepcopy(self._clean_entries)
        self._fit_index()

    def clone_with_poison(self, poison: MemoryEntry) -> FinanceMemoryStore:
        store = FinanceMemoryStore(
            entries=copy.deepcopy(self._clean_entries),
            backend=self.backend,
            embedder=self._embedder,
        )
        store._clean_entries = copy.deepcopy(self._clean_entries)
        store.add_poison_memory(poison)
        return store

    @property
    def clean_entry_count(self) -> int:
        return len(self._clean_entries)

    @property
    def poison_count(self) -> int:
        return sum(1 for e in self._entries if e.is_poison)
