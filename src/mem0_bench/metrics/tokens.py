"""Token economics: usage accounting and configurable cost estimation.

Pricing lives in configuration (``configs/benchmark.yaml`` -> ``pricing``),
never hardcoded here, so the same evaluation logic works for any model's
published rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TokenUsage:
    input_tokens: int = 0
    retrieved_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.retrieved_tokens + self.output_tokens


@dataclass
class ModelPricing:
    """Price per 1,000 tokens, in USD."""

    input_per_1k: float
    output_per_1k: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1000) * self.input_per_1k + (output_tokens / 1000) * self.output_per_1k


def load_pricing_table(raw: Dict) -> Dict[str, ModelPricing]:
    table = {}
    for model, rates in raw.items():
        table[model] = ModelPricing(
            input_per_1k=float(rates["input_per_1k"]),
            output_per_1k=float(rates["output_per_1k"]),
        )
    return table


@dataclass
class TokenAggregator:
    usages: List[TokenUsage] = field(default_factory=list)

    def record(self, usage: TokenUsage) -> None:
        self.usages.append(usage)

    def total(self) -> TokenUsage:
        agg = TokenUsage()
        for u in self.usages:
            agg.input_tokens += u.input_tokens
            agg.retrieved_tokens += u.retrieved_tokens
            agg.output_tokens += u.output_tokens
        return agg

    def per_query_average(self) -> TokenUsage:
        n = len(self.usages) or 1
        t = self.total()
        return TokenUsage(
            input_tokens=t.input_tokens // n,
            retrieved_tokens=t.retrieved_tokens // n,
            output_tokens=t.output_tokens // n,
        )

    def estimated_cost(self, pricing: ModelPricing) -> float:
        t = self.total()
        return pricing.cost(t.input_tokens + t.retrieved_tokens, t.output_tokens)

    def estimated_cost_per_1k_queries(self, pricing: ModelPricing) -> float:
        n = len(self.usages)
        if n == 0:
            return 0.0
        return self.estimated_cost(pricing) * (1000 / n)
