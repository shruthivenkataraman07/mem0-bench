"""Builds a configured :class:`MemoryBackend` instance for a given backend
name + :class:`~mem0_bench.config.BenchmarkConfig`.

This is the one place backend-specific *construction* (which embedding
provider, which vector store, which Mem0 config) lives, kept deliberately
separate from :class:`~mem0_bench.benchmark.runner.BenchmarkRunner`, which
never branches on backend identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from mem0_bench.config import CONFIG_DIR, BenchmarkConfig, load_yaml_config
from mem0_bench.llm.client import LLMClient, get_llm_client
from mem0_bench.llm.tokenizer import get_tokenizer
from mem0_bench.memory.base import MemoryBackend
from mem0_bench.memory.embeddings import get_embedding_provider
from mem0_bench.memory.full_context import FullContextBackend
from mem0_bench.memory.mem0_adapter import Mem0Adapter
from mem0_bench.memory.rag_adapter import RAGBackend
from mem0_bench.memory.vector_store import get_vector_store

BACKEND_CONFIG_FILES = {
    "mem0": "mem0.yaml",
    "rag": "rag.yaml",
    "full-context": "full_context.yaml",
}


def _load_backend_config(backend: str, override_path: Optional[str] = None) -> Dict:
    path = Path(override_path) if override_path else CONFIG_DIR / BACKEND_CONFIG_FILES[backend]
    if not path.exists():
        return {}
    return load_yaml_config(path)


def build_llm(config: BenchmarkConfig) -> LLMClient:
    tokenizer = get_tokenizer(config.tokenizer.provider)
    if config.llm.provider == "local":
        return get_llm_client("local")
    return get_llm_client(
        "openai", model=config.llm.model, temperature=config.llm.temperature,
        timeout_s=config.llm.timeout_s, tokenizer=tokenizer,
    )


def build_backend(
    name: str,
    config: BenchmarkConfig,
    backend_config_path: Optional[str] = None,
) -> Tuple[MemoryBackend, Dict]:
    """Returns (backend_instance, resolved_backend_config_dict_for_tracing)."""
    tokenizer = get_tokenizer(config.tokenizer.provider)
    llm = build_llm(config)
    raw_config = _load_backend_config(name, backend_config_path)

    backend: MemoryBackend

    if name == "full-context":
        max_context_tokens = raw_config.get("max_context_tokens", 8000)
        backend = FullContextBackend(llm=llm, tokenizer=tokenizer, max_context_tokens=max_context_tokens)
        return backend, {"max_context_tokens": max_context_tokens}

    if name == "rag":
        top_k = raw_config.get("top_k", config.top_k)
        threshold = raw_config.get("similarity_threshold")
        embed_cfg = raw_config.get("embeddings", {"provider": "hashing", "dim": 512})
        vs_cfg = raw_config.get("vector_store", {"provider": "numpy"})
        embedding_provider = get_embedding_provider(
            embed_cfg.get("provider", "hashing"),
            **{k: v for k, v in embed_cfg.items() if k != "provider"},
        )
        vector_store = get_vector_store(
            vs_cfg.get("provider", "numpy"),
            **{k: v for k, v in vs_cfg.items() if k != "provider"},
        )
        backend = RAGBackend(
            embedding_provider=embedding_provider, vector_store=vector_store,
            llm=llm, tokenizer=tokenizer, default_top_k=top_k, similarity_threshold=threshold,
        )
        return backend, {
            "top_k": top_k, "similarity_threshold": threshold,
            "embedding_provider": embedding_provider.name,
            "vector_store_provider": vs_cfg.get("provider", "numpy"),
        }

    if name == "mem0":
        top_k = raw_config.get("top_k", config.top_k)
        mem0_config = raw_config.get("mem0_config") or None
        backend = Mem0Adapter(llm=llm, tokenizer=tokenizer, mem0_config=mem0_config, default_top_k=top_k)
        return backend, {"top_k": top_k, "mem0_config_provider": (mem0_config or {}).get("llm", {}).get("provider")}

    raise ValueError(f"unknown backend: {name!r}. Expected one of: mem0, rag, full-context")


def resolve_model_name(config: BenchmarkConfig) -> str:
    return config.llm.model


def resolve_embedding_model_name(
    name: str, config: BenchmarkConfig, backend_config_path: Optional[str] = None
) -> Optional[str]:
    if name != "rag":
        return None
    raw_config = _load_backend_config(name, backend_config_path)
    embed_cfg = raw_config.get("embeddings", {})
    return embed_cfg.get("provider", config.embeddings.provider)
