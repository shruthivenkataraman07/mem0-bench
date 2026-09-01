"""Latency aggregation using high-resolution monotonic timing.

Raw samples are always kept alongside the summary so callers can inspect the
full distribution rather than trusting a single point estimate. Percentiles
computed from very small sample counts are annotated as low-confidence
rather than presented with false precision.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List

LOW_CONFIDENCE_SAMPLE_THRESHOLD = 20


def percentile(sorted_samples: List[float], p: float) -> float:
    """Linear-interpolation percentile (0 <= p <= 100) over already-sorted samples."""
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    k = (len(sorted_samples) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_samples) - 1)
    if f == c:
        return sorted_samples[f]
    return sorted_samples[f] + (sorted_samples[c] - sorted_samples[f]) * (k - f)


@dataclass
class LatencySummary:
    count: int
    mean: float
    median: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float
    stddev: float
    low_confidence: bool

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean_ms": round(self.mean, 3),
            "median_ms": round(self.median, 3),
            "p50_ms": round(self.p50, 3),
            "p95_ms": round(self.p95, 3),
            "p99_ms": round(self.p99, 3),
            "min_ms": round(self.min, 3),
            "max_ms": round(self.max, 3),
            "stddev_ms": round(self.stddev, 3),
            "low_confidence": self.low_confidence,
        }


@dataclass
class LatencyTracker:
    """Accumulates raw latency samples (in milliseconds) for one stage/backend."""

    samples: List[float] = field(default_factory=list)

    def record(self, ms: float) -> None:
        self.samples.append(ms)

    def summarize(self) -> LatencySummary:
        if not self.samples:
            return LatencySummary(0, 0, 0, 0, 0, 0, 0, 0, 0, low_confidence=True)
        s = sorted(self.samples)
        mean = statistics.fmean(s)
        stddev = statistics.pstdev(s) if len(s) > 1 else 0.0
        return LatencySummary(
            count=len(s),
            mean=mean,
            median=statistics.median(s),
            p50=percentile(s, 50),
            p95=percentile(s, 95),
            p99=percentile(s, 99),
            min=s[0],
            max=s[-1],
            stddev=stddev,
            low_confidence=len(s) < LOW_CONFIDENCE_SAMPLE_THRESHOLD,
        )


def summarize_latencies(samples: List[float]) -> LatencySummary:
    tracker = LatencyTracker(samples=list(samples))
    return tracker.summarize()
