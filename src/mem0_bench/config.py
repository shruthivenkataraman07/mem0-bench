"""Configuration loading: YAML files + environment variable overrides.

A YAML string value of the form ``${VAR_NAME}`` or ``${VAR_NAME:default}``
is substituted from the environment at load time. This is how
``configs/*.yaml`` reference API keys without ever containing a literal
secret.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _substitute_env(value: Any) -> Any:
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            var_name, default = match.group(1), match.group(2)
            return os.environ.get(var_name, default if default is not None else "")

        return _ENV_VAR_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


def load_env(dotenv_path: Optional[str] = None) -> None:
    load_dotenv(dotenv_path=dotenv_path, override=False)


def load_yaml_config(path: str | Path) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _substitute_env(raw)


class RetryPolicy(BaseModel):
    max_attempts: int = 4
    backoff_seconds: float = 0.5
    backoff_max_seconds: float = 8.0


class LLMConfig(BaseModel):
    provider: str = "local"
    model: str = "local-extractive-v1"
    temperature: float = 0.0
    timeout_s: float = 30.0


class EmbeddingConfig(BaseModel):
    provider: str = "hashing"
    model: str = "hashing-512"
    dim: int = 512


class TokenizerConfig(BaseModel):
    provider: str = "approximate"


class JudgeConfig(BaseModel):
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o-mini"


class EfficiencyScoreConfig(BaseModel):
    """Weights for the experimental, clearly-labeled Memory Efficiency Score.

    See README, "Engineering composite metric"; this is NOT a validated
    scientific metric, just a configurable weighted combination exposed for
    convenience. The raw underlying metrics are always reported alongside it.
    """

    enabled: bool = True
    weight_accuracy: float = 0.5
    weight_latency: float = 0.3
    weight_tokens: float = 0.2
    latency_reference_ms: float = 2000.0
    tokens_reference: float = 4000.0


class BenchmarkConfig(BaseModel):
    seed: int = 42
    concurrency: int = 4
    output_dir: str = "results"
    dataset_path: Optional[str] = None
    top_k: int = 5

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    efficiency_score: EfficiencyScoreConfig = Field(default_factory=EfficiencyScoreConfig)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    pricing: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        return cls.model_validate(load_yaml_config(path))

    @classmethod
    def default(cls) -> BenchmarkConfig:
        default_path = CONFIG_DIR / "benchmark.yaml"
        if default_path.exists():
            return cls.from_yaml(default_path)
        return cls()
