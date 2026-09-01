"""A small registry so new memory backends can be benchmarked without
modifying :class:`~mem0_bench.benchmark.runner.BenchmarkRunner` or the CLI.

Usage::

    from mem0_bench.memory.registry import register_backend
    from mem0_bench.memory.base import MemoryBackend

    @register_backend("my-memory")
    class MyMemorySystem(MemoryBackend):
        ...

Once registered, ``mem0-bench run --backend my-memory ...`` and
``mem0-bench compare --backends mem0,rag,my-memory`` work immediately.
"""

from __future__ import annotations

from typing import Callable, Dict, Type

from mem0_bench.memory.base import MemoryBackend

_REGISTRY: Dict[str, Type[MemoryBackend]] = {}


def register_backend(name: str) -> Callable[[Type[MemoryBackend]], Type[MemoryBackend]]:
    def _decorator(cls: Type[MemoryBackend]) -> Type[MemoryBackend]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"backend name already registered: {name!r}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_backend_class(name: str) -> Type[MemoryBackend]:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered yet)"
        raise KeyError(f"no memory backend registered as {name!r}. Available: {available}")
    return _REGISTRY[name]


def available_backends() -> Dict[str, Type[MemoryBackend]]:
    return dict(_REGISTRY)
