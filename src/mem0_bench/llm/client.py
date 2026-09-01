"""LLM client abstraction used for answer generation across all backends.

Two providers ship in this repository:

- ``local`` (:class:`LocalExtractiveLLM`): a deterministic, dependency-free,
  network-free heuristic answerer. It is **not** a language model: it
  scores candidate context lines against the query by lexical overlap and
  returns the best match (or a combination of the best two, for
  conjunctive/multi-hop queries). It exists so the entire benchmark
  pipeline (ingestion, retrieval, generation, metrics, CLI, dashboard)
  can be demonstrated end-to-end with zero API keys and zero cost. It is
  clearly labeled as such everywhere it appears in output.
- ``openai`` (:class:`OpenAILLM`): a real integration against the OpenAI
  Chat Completions API, used when ``OPENAI_API_KEY`` is configured.

Both implement the same :class:`LLMClient` interface, so the rest of the
codebase (all three memory backends) is provider-agnostic.
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mem0_bench.llm.tokenizer import Tokenizer, get_tokenizer

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "i",
    "my", "me", "you", "your", "what", "where", "who", "when", "how", "to",
    "of", "in", "on", "at", "and", "or", "it", "this", "that", "currently",
    "current", "tell", "about",
}


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    provider: str


class LLMError(RuntimeError):
    """Raised when an LLM provider fails after retries, or is misconfigured."""


class LLMClient(ABC):
    provider: str
    model: str

    def __init__(self, tokenizer: Optional[Tokenizer] = None) -> None:
        self.tokenizer = tokenizer or get_tokenizer("approximate")

    @abstractmethod
    async def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        ...


def _tokenize(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


class LocalExtractiveLLM(LLMClient):
    """Deterministic, offline stand-in for an LLM. See module docstring."""

    provider = "local"
    model = "local-extractive-v1"

    async def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        start = time.monotonic()
        context, question = _split_prompt(prompt)
        lines = [ln.strip("- ").strip() for ln in context.splitlines() if ln.strip()]
        q_tokens = set(_tokenize(question))

        scored = []
        for i, line in enumerate(lines):
            l_tokens = set(_tokenize(line))
            if not l_tokens or not q_tokens:
                overlap = 0.0
            else:
                overlap = len(q_tokens & l_tokens) / len(q_tokens)
            recency_bonus = i / max(1, len(lines) - 1) * 0.05
            scored.append((overlap + recency_bonus, i, line))

        scored.sort(key=lambda t: (-t[0], -t[1]))

        if not scored or scored[0][0] == 0.0:
            text = "I don't have enough information to answer that."
        elif " and " in question.lower() and len(scored) > 1 and scored[1][0] >= scored[0][0] * 0.5:
            text = f"{scored[0][2]} {scored[1][2]}"
        else:
            text = scored[0][2]

        latency_ms = (time.monotonic() - start) * 1000
        return LLMResponse(
            text=text,
            input_tokens=self.tokenizer.count(prompt) + self.tokenizer.count(system or ""),
            output_tokens=self.tokenizer.count(text),
            latency_ms=latency_ms,
            model=self.model,
            provider=self.provider,
        )


def _split_prompt(prompt: str) -> tuple:
    if "Question:" in prompt:
        context, _, question = prompt.rpartition("Question:")
    else:
        context, question = prompt, prompt
    return context, question


class OpenAILLM(LLMClient):
    """Real integration against the OpenAI Chat Completions API."""

    provider = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 30.0,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        super().__init__(tokenizer=tokenizer or get_tokenizer("tiktoken"))
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.timeout_s = timeout_s
        if not self.api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set. Configure it in your .env file, "
                "or select llm.provider=local for an offline run."
            )

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    async def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": self.temperature,
                    "messages": messages,
                },
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                resp.raise_for_status()
            if resp.status_code >= 400:
                raise LLMError(f"OpenAI API error {resp.status_code}: {resp.text}")
            data = resp.json()

        latency_ms = (time.monotonic() - start) * 1000
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", self.tokenizer.count(prompt)),
            output_tokens=usage.get("completion_tokens", self.tokenizer.count(text)),
            latency_ms=latency_ms,
            model=self.model,
            provider=self.provider,
        )


def get_llm_client(provider: str = "local", **kwargs) -> LLMClient:
    if provider == "local":
        return LocalExtractiveLLM()
    if provider == "openai":
        return OpenAILLM(**kwargs)
    raise ValueError(f"unknown llm provider: {provider!r}")
