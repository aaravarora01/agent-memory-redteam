"""Finance memory store adapter that delegates all retrieval to the
top-level FAISS-backed `env.memory_store.MemoryStore` so the repo uses a
single retrieval implementation across experiments.

This preserves the `finance_poisoning/env/schemas.py` `MemoryEntry` and
`RetrievalHit` shapes while reusing the robust FAISS/MiniLM code in
`env/memory_store.py`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np

from .schemas import DATA_DIR, MemoryEntry, RetrievalHit, load_clean_memories

# Prefer the top-level FAISS-backed store when FAISS is available; fall
# back to a TF-IDF based retriever so tests run in lightweight envs.
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

if FAISS_AVAILABLE:
    from env.memory_store import MemoryStore as FaissMemoryStore
else:
    FaissMemoryStore = None  # type: ignore


class FinanceMemoryStore:
    """Adapter presenting the finance memory store API while delegating to
    the top-level FAISS-backed `MemoryStore` for embedding & retrieval.
    """

    def __init__(self, entries: Optional[list[MemoryEntry]] = None, embedder=None):
        self._clean_entries: list[MemoryEntry] = []
        self._entries: list[MemoryEntry] = []
        self._use_faiss = FAISS_AVAILABLE and FaissMemoryStore is not None
        self._embedder = embedder
        self._faiss: Optional[FaissMemoryStore] = None
        self._tfidf = None
        self._vectors = None
        if self._use_faiss:
            self._faiss = FaissMemoryStore(embedder=embedder)
        if entries is not None:
            self._load_entries(entries)

    @classmethod
    def from_clean_corpus(
        cls,
        path: Path | str = DATA_DIR / "clean_memories.jsonl",
        embedder=None,
    ) -> "FinanceMemoryStore":
        entries = load_clean_memories(path)
        store = cls(entries=entries, embedder=embedder)
        store._clean_entries = copy.deepcopy(store._entries)
        return store

    def _load_entries(self, entries: list[MemoryEntry]) -> None:
        # keep finance entries and build the FAISS store in parallel
        self._entries = copy.deepcopy(entries)
        texts = [e.text for e in self._entries]
        if texts:
            if self._use_faiss and self._faiss is not None:
                vecs = self._faiss.embedder.encode(texts, normalize_embeddings=True)
                arr = np.asarray(vecs, dtype="float32")
                # Add vectors to the underlying FAISS index
                self._faiss._index.add(arr)
                # Mirror entries as simple env MemoryEntry with id=memory_id
                self._faiss._entries = [
                    type("_E", (), {"id": e.memory_id, "text": e.text, "metadata": {}})()
                    for e in self._entries
                ]
                self._vectors = arr
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer

                self._tfidf = TfidfVectorizer(stop_words="english")
                self._vectors = self._tfidf.fit_transform(texts).astype("float32")

    def __len__(self) -> int:
        return len(self._entries)

    def add_memory(self, memory: MemoryEntry) -> None:
        entry = copy.deepcopy(memory)
        entry.is_poison = False
        self._entries.append(entry)
        # add vector to the underlying index
        if self._use_faiss and self._faiss is not None:
            vec = self._faiss._embed(entry.text)
            self._faiss._add_vector(
                type("_E", (), {"id": entry.memory_id, "text": entry.text, "metadata": {}})(),
                vec,
            )
        else:
            # rebuild TF-IDF with the appended entry
            texts = [e.text for e in self._entries]
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(stop_words="english")
            self._vectors = self._tfidf.fit_transform(texts).astype("float32")

    def add_poison_memory(self, memory: MemoryEntry) -> None:
        entry = copy.deepcopy(memory)
        entry.is_poison = True
        self._entries.append(entry)
        if self._use_faiss and self._faiss is not None:
            vec = self._faiss._embed(entry.text)
            self._faiss._add_vector(
                type("_E", (), {"id": entry.memory_id, "text": entry.text, "metadata": {}})(),
                vec,
            )
        else:
            texts = [e.text for e in self._entries]
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(stop_words="english")
            self._vectors = self._tfidf.fit_transform(texts).astype("float32")

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalHit]:
        if not self._entries:
            return []
        k = min(k, len(self._entries))
        if self._use_faiss and self._faiss is not None:
            hits = self._faiss.query(query, k=k)
            # Map Faiss MemoryEntry (id=text) back to finance MemoryEntry by id
            id_to_entry = {e.memory_id: e for e in self._entries}
            out: list[RetrievalHit] = []
            for rank, h in enumerate(hits, start=1):
                fin = id_to_entry.get(h.id)
                if fin is None:
                    continue
                value = fin.poison_value if fin.is_poison else fin.true_value
                out.append(
                    RetrievalHit(
                        memory_id=fin.memory_id,
                        rank=rank,
                        score=h.score,
                        text=fin.text,
                        fact_id=fin.fact_id,
                        is_poison=fin.is_poison,
                        value=value,
                    )
                )
            return out
        else:
            # TF-IDF cosine similarity path
            if self._vectors is None:
                return []
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
        # rebuild underlying index according to availability
        self._faiss = None
        if self._use_faiss and FaissMemoryStore is not None:
            self._faiss = FaissMemoryStore(embedder=self._embedder)
        self._load_entries(self._entries)

    def clone_with_poison(self, poison: MemoryEntry) -> "FinanceMemoryStore":
        emb = None
        if self._use_faiss and self._faiss is not None:
            emb = getattr(self._faiss, "_embedder", None)
        if emb is None:
            emb = self._embedder
        store = FinanceMemoryStore(entries=copy.deepcopy(self._clean_entries), embedder=emb)
        store._clean_entries = copy.deepcopy(self._clean_entries)
        store.add_poison_memory(poison)
        return store

    @property
    def clean_entry_count(self) -> int:
        return len(self._clean_entries)

    @property
    def poison_count(self) -> int:
        return sum(1 for e in self._entries if e.is_poison)
