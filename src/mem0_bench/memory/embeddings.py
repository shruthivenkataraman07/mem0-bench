"""Embedding providers for the RAG baseline.

``hashing`` is the default: a deterministic, offline feature-hashing
embedder (a real, if simple, bag-of-words embedding, not a mock). It
requires no network access and no API key, which is what lets the RAG
backend and its tests run fully offline.

``openai`` is a real integration against OpenAI's embeddings API, used only
when explicitly configured with an API key.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import httpx
import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class EmbeddingResult:
    vectors: np.ndarray  # shape (n, dim)
    latency_ms: float


class EmbeddingProvider(ABC):
    name: str
    dim: int

    @abstractmethod
    async def embed(self, texts: List[str]) -> EmbeddingResult:
        ...


class HashingEmbedding(EmbeddingProvider):
    """Deterministic feature-hashing bag-of-words embedding.

    Each token is hashed into one of `dim` buckets (sign determined by a
    second hash, following the standard feature-hashing trick), then the
    resulting vector is L2-normalized. Identical text always yields an
    identical vector; no external calls are made.
    """

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    async def embed(self, texts: List[str]) -> EmbeddingResult:
        start = time.monotonic()
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            for tok in tokens:
                h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                sign = 1.0 if h[4] % 2 == 0 else -1.0
                vectors[i, idx] += sign
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        latency_ms = (time.monotonic() - start) * 1000
        return EmbeddingResult(vectors=vectors, latency_ms=latency_ms)


class OpenAIEmbedding(EmbeddingProvider):
    """Real integration against the OpenAI embeddings API."""

    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        dim: int = 1536,
        timeout_s: float = 30.0,
    ) -> None:
        self.model = model
        self.dim = dim
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.timeout_s = timeout_s
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Configure it in your .env file, "
                "or select embeddings.provider=hashing for an offline run."
            )

    async def embed(self, texts: List[str]) -> EmbeddingResult:
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
        vectors = np.array([d["embedding"] for d in data["data"]], dtype=np.float32)
        latency_ms = (time.monotonic() - start) * 1000
        return EmbeddingResult(vectors=vectors, latency_ms=latency_ms)


def get_embedding_provider(provider: str = "hashing", **kwargs) -> EmbeddingProvider:
    if provider == "hashing":
        return HashingEmbedding(**kwargs)
    if provider == "openai":
        return OpenAIEmbedding(**kwargs)
    raise ValueError(f"unknown embedding provider: {provider!r}")
