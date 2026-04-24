"""
Benchmark harness for side-by-side SDK comparisons.

Usage:
    with bench("openai gpt-4o-mini") as rec:
        resp = client.chat.completions.create(...)
        rec.input_tokens = resp.usage.prompt_tokens
        rec.output_tokens = resp.usage.completion_tokens
        rec.ok = True  # flip False if parsing/validation fails
    # rec is also appended to the shared results list

The harness times each call, tracks tokens, and (given a pricing table) estimates cost.
Aggregates into a pandas DataFrame for side-by-side reporting.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import mean, quantiles
from typing import Any

import pandas as pd

# Posted per-token pricing (USD) — snapshot 2026-04-23.
# Verify against vendor pricing pages before reporting numbers; rates change.
PRICING = {
    # OpenAI cached_input tokens are billed at 0.1x input; Anthropic cache reads
    # are also 0.1x input (5-min TTL) and cache writes are 1.25x. For a single
    # cost_usd formula we keep only "input" / "output" here and apply cache
    # multipliers inline on the Record / TaskOutcome cost properties.
    "gpt-5.4-mini":       {"input": 0.75 / 1_000_000, "output": 4.50 / 1_000_000},
    "gpt-5.4":            {"input": 2.50 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-haiku-4-5":   {"input": 1.00 / 1_000_000, "output": 5.00 / 1_000_000},
    "claude-sonnet-4-6":  {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
}


@dataclass
class Record:
    label: str                      # e.g. "openai/gpt-5.4-mini — function_calling"
    model: str                      # e.g. "gpt-5.4-mini"
    provider: str                   # "openai" or "anthropic"
    latency_ms: float = 0.0
    ttft_ms: float = 0.0             # time to first streamed token (0 for non-streaming calls)
    input_tokens: int = 0            # uncached input tokens (full price)
    output_tokens: int = 0
    # Anthropic prompt caching (0 for non-cached / non-Anthropic calls):
    cache_creation_input_tokens: int = 0   # priced at 1.25x input (5-min ephemeral TTL)
    cache_read_input_tokens: int = 0       # priced at 0.1x input
    ok: bool = False                # True if the response parsed + validated
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        p = PRICING.get(self.model)
        if not p:
            return 0.0
        return (
            self.input_tokens * p["input"]
            + self.cache_creation_input_tokens * p["input"] * 1.25
            + self.cache_read_input_tokens * p["input"] * 0.1
            + self.output_tokens * p["output"]
        )


# Shared results list populated by the `bench` context manager.
# Callers can also maintain their own list and pass it in explicitly.
RESULTS: list[Record] = []


@contextmanager
def bench(label: str, model: str, provider: str, results: list[Record] | None = None):
    """Time an SDK call and record metrics. Yields a Record the caller fills in."""
    rec = Record(label=label, model=model, provider=provider)
    start = time.perf_counter()
    try:
        yield rec
    except Exception as e:
        rec.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        rec.latency_ms = (time.perf_counter() - start) * 1000
        (results if results is not None else RESULTS).append(rec)


def summarise(records: list[Record] | None = None) -> pd.DataFrame:
    """Return a DataFrame with per-group aggregates: p50/p95 latency, mean tokens, cost/1k, validity."""
    records = records if records is not None else RESULTS
    if not records:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    by_label: dict[str, list[Record]] = {}
    for r in records:
        by_label.setdefault(r.label, []).append(r)

    any_cache = any(
        r.cache_creation_input_tokens or r.cache_read_input_tokens for r in records
    )
    any_ttft = any(r.ttft_ms for r in records)

    for label, rs in by_label.items():
        latencies = [r.latency_ms for r in rs]
        ok_rate = sum(1 for r in rs if r.ok) / len(rs)
        in_tokens = [r.input_tokens for r in rs]
        out_tokens = [r.output_tokens for r in rs]
        cost_per_call = mean([r.cost_usd for r in rs])
        # quantiles needs n>=2; fall back to single-value duplicated when needed
        p50, p95 = (
            quantiles(latencies, n=100)[49] if len(latencies) >= 2 else latencies[0],
            quantiles(latencies, n=100)[94] if len(latencies) >= 2 else latencies[0],
        )
        row = {
            "label": label,
            "n": len(rs),
            "latency_p50_ms": round(p50, 1),
            "latency_p95_ms": round(p95, 1),
            "mean_input_tokens": round(mean(in_tokens), 1),
            "mean_output_tokens": round(mean(out_tokens), 1),
            "cost_per_1k_usd": round(cost_per_call * 1000, 4),
            "ok_rate": round(ok_rate, 3),
        }
        if any_ttft:
            ttfts = [r.ttft_ms for r in rs if r.ttft_ms]
            if ttfts:
                ttft_p50 = quantiles(ttfts, n=100)[49] if len(ttfts) >= 2 else ttfts[0]
                ttft_p95 = quantiles(ttfts, n=100)[94] if len(ttfts) >= 2 else ttfts[0]
                row["ttft_p50_ms"] = round(ttft_p50, 1)
                row["ttft_p95_ms"] = round(ttft_p95, 1)
        if any_cache:
            row["mean_cache_creation"] = round(mean([r.cache_creation_input_tokens for r in rs]), 1)
            row["mean_cache_read"] = round(mean([r.cache_read_input_tokens for r in rs]), 1)
        rows.append(row)

    return pd.DataFrame(rows)


def reset():
    """Clear the global results list between notebook runs."""
    RESULTS.clear()
