"""Pluggable vector storage for the RAG baseline.

``NumpyVectorStore`` is the default: a small, dependency-light, in-memory
cosine-similarity index good enough for local reproducibility (the same
role Chroma/FAISS/Qdrant-local would play). It partitions vectors by
``user_id`` so cross-user retrieval is structurally impossible, not just
filtered after the fact.

The :class:`VectorStore` interface is intentionally narrow so a real
Chroma/FAISS/Qdrant backend can be dropped in later without touching the RAG
adapter. See ``ChromaVectorStore`` below for the extension point (it is a
real integration, guarded behind the optional ``chroma`` extra).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class VectorRecord:
    record_id: str
    vector: np.ndarray
    text: str
    metadata: Dict


@dataclass
class VectorHit:
    record_id: str
    score: float
    text: str
    metadata: Dict


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, user_id: str, records: List[VectorRecord]) -> None:
        ...

    @abstractmethod
    def query(self, user_id: str, vector: np.ndarray, k: int) -> List[VectorHit]:
        ...

    @abstractmethod
    def reset(self, user_id: str) -> None:
        ...

    @abstractmethod
    def count(self, user_id: str) -> int:
        ...


@dataclass
class _UserPartition:
    ids: List[str] = field(default_factory=list)
    vectors: List[np.ndarray] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    metadatas: List[Dict] = field(default_factory=list)


class NumpyVectorStore(VectorStore):
    """In-memory, per-user cosine-similarity vector index."""

    def __init__(self) -> None:
        self._partitions: Dict[str, _UserPartition] = {}

    def _partition(self, user_id: str) -> _UserPartition:
        return self._partitions.setdefault(user_id, _UserPartition())

    def upsert(self, user_id: str, records: List[VectorRecord]) -> None:
        p = self._partition(user_id)
        for r in records:
            p.ids.append(r.record_id)
            p.vectors.append(r.vector)
            p.texts.append(r.text)
            p.metadatas.append(r.metadata)

    def query(self, user_id: str, vector: np.ndarray, k: int) -> List[VectorHit]:
        p = self._partitions.get(user_id)
        if not p or not p.vectors:
            return []
        matrix = np.stack(p.vectors)  # (n, dim)
        denom = (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)) + 1e-12
        # On some platforms (observed with the Accelerate BLAS backend numpy
        # uses on macOS), matmul trips FP exception flags (divide-by-zero /
        # overflow / invalid) even when every element of the result is
        # finite and correct: a known spurious-warning quirk, not a real
        # numerical error. We suppress the warning here and defensively
        # zero out any actual non-finite score (e.g. a genuine zero-vector
        # query) rather than letting it silently propagate as NaN.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = (matrix @ vector) / denom
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        top_idx = np.argsort(-scores)[:k]
        return [
            VectorHit(
                record_id=p.ids[i],
                score=float(scores[i]),
                text=p.texts[i],
                metadata=p.metadatas[i],
            )
            for i in top_idx
        ]

    def reset(self, user_id: str) -> None:
        self._partitions.pop(user_id, None)

    def count(self, user_id: str) -> int:
        p = self._partitions.get(user_id)
        return len(p.ids) if p else 0


class ChromaVectorStore(VectorStore):
    """Real Chroma-backed vector store (requires the `chroma` extra).

    Provided as a concrete example of swapping in a production vector
    database behind the same :class:`VectorStore` interface. Not exercised
    by the default unit test suite (it needs `chromadb` installed).
    """

    def __init__(self, persist_directory: str = ".chroma") -> None:
        try:
            import chromadb
        except ImportError as e:
            raise RuntimeError(
                "chromadb is not installed. Install the 'chroma' extra "
                "(`pip install mem0-bench[chroma]`) or use vector_store.provider=numpy."
            ) from e
        self._client = chromadb.PersistentClient(path=persist_directory)

    def _collection(self, user_id: str):
        return self._client.get_or_create_collection(name=f"user_{user_id}")

    def upsert(self, user_id: str, records: List[VectorRecord]) -> None:
        if not records:
            return
        col = self._collection(user_id)
        col.upsert(
            ids=[r.record_id for r in records],
            embeddings=[r.vector.tolist() for r in records],
            documents=[r.text for r in records],
            metadatas=[r.metadata for r in records],
        )

    def query(self, user_id: str, vector: np.ndarray, k: int) -> List[VectorHit]:
        col = self._collection(user_id)
        if col.count() == 0:
            return []
        res = col.query(query_embeddings=[vector.tolist()], n_results=min(k, col.count()))
        hits = []
        for rid, dist, doc, meta in zip(
            res["ids"][0], res["distances"][0], res["documents"][0], res["metadatas"][0]
        ):
            hits.append(VectorHit(record_id=rid, score=1.0 - dist, text=doc, metadata=meta or {}))
        return hits

    def reset(self, user_id: str) -> None:
        self._client.delete_collection(name=f"user_{user_id}")

    def count(self, user_id: str) -> int:
        return self._collection(user_id).count()


def get_vector_store(provider: str = "numpy", **kwargs) -> VectorStore:
    if provider == "numpy":
        return NumpyVectorStore(**kwargs)
    if provider == "chroma":
        return ChromaVectorStore(**kwargs)
    raise ValueError(f"unknown vector store provider: {provider!r}")
