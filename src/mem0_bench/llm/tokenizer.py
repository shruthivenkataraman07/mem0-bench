"""Token counting.

Exact token counts depend on the target model's tokenizer. By default we use
a dependency-free, deterministic regex tokenizer that approximates
whitespace/punctuation-delimited tokens; this keeps the entire benchmark
(including CI) runnable with no network access and no vendor lock-in.

If the ``tiktoken`` package is installed *and* its encoding files are
reachable (they are fetched from OpenAI's CDN on first use), you can select
the exact OpenAI tokenizer via config. We never silently fall back: if you
ask for tiktoken and it isn't usable, you get a clear error.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", re.UNICODE)


class Tokenizer(ABC):
    name: str

    @abstractmethod
    def count(self, text: str) -> int:
        ...


class ApproximateTokenizer(Tokenizer):
    """Regex word/punctuation tokenizer. Deterministic, offline, approximate.

    This is documented as an approximation, not presented as exact-matching
    any particular model's BPE vocabulary.
    """

    name = "approximate"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(_TOKEN_RE.findall(text))


class TiktokenTokenizer(Tokenizer):
    """Exact OpenAI tokenizer, via the optional `tiktoken` dependency."""

    name = "tiktoken"

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as e:
            raise RuntimeError(
                "tiktoken is not installed. Install it with `pip install tiktoken` "
                "or set tokenizer.provider=approximate in your config."
            ) from e
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            raise RuntimeError(
                f"Could not load tiktoken encoding '{encoding_name}' "
                "(this usually means no network access to fetch it). "
                "Use tokenizer.provider=approximate for offline runs."
            ) from e

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))


def get_tokenizer(provider: str = "approximate", **kwargs) -> Tokenizer:
    if provider == "approximate":
        return ApproximateTokenizer()
    if provider == "tiktoken":
        return TiktokenTokenizer(**kwargs)
    raise ValueError(f"unknown tokenizer provider: {provider!r}")
