# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 04 — Streaming: TTFT vs total latency
#
# Notebooks 01–03 measure request-level latency: p50 and p95 of the full round-trip
# time. That's the right metric for batch workloads, but the wrong one for anything
# user-facing. What drives perceived responsiveness is **TTFT — time to first token**:
# how long before the user sees *something*.
#
# This notebook streams a short generative task (2-sentence review summary + sentiment
# label) through both providers and compares:
#
# - **TTFT p50 / p95** — latency of the first visible character.
# - **Total latency p50 / p95** — time to the last token, same as prior notebooks.
# - **Steady-state throughput** — output tokens per second of generation (excluding TTFT).
#
# A generative task is used instead of the earlier classification task because
# streaming tool-use / structured-output changes the definition of "first token"
# (first `{` byte? first tool_use block?). Plain prose sidesteps that.

# %%
import os
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

from src.bench import RESULTS, bench, reset, summarise

load_dotenv()
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

OPENAI_MODEL = "gpt-5.4-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"

import json  # noqa: E402

DATA_PATH = next(p for p in (Path("data/reviews.json"), Path("../data/reviews.json")) if p.exists())
reviews = json.loads(DATA_PATH.read_text())
print(f"Loaded {len(reviews)} reviews")

# %% [markdown]
# ## Prompts — generative summary task

# %%
SYSTEM = (
    "You are a product-review analyst. Given a product review, write a concise "
    "two-sentence summary highlighting what the reviewer cared about, then state "
    "the overall sentiment in one word (positive, negative, neutral, or mixed)."
)

USER_TEMPLATE = "Review:\n{review_text}"


# %% [markdown]
# ## Per-provider streaming functions
#
# For each call we record:
# - `ttft_ms` — wall-clock time from request start to the first streamed text chunk.
# - `latency_ms` — already captured by the `bench()` context manager.
# - `output_tokens`, `input_tokens` — from the post-stream usage object.

# %%
def run_openai_stream(review_text: str, rec):
    """Stream an OpenAI chat completion. Sets rec.ttft_ms inside the loop."""
    start = time.perf_counter()
    stream = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        max_completion_tokens=256,  # gpt-5 series requires max_completion_tokens, not max_tokens
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TEMPLATE.format(review_text=review_text)},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    text_parts = []
    usage = None
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            if rec.ttft_ms == 0.0:
                rec.ttft_ms = (time.perf_counter() - start) * 1000
            text_parts.append(chunk.choices[0].delta.content)
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage

    return "".join(text_parts), usage


def run_anthropic_stream(review_text: str, rec):
    """Stream an Anthropic message. Sets rec.ttft_ms inside the loop."""
    start = time.perf_counter()
    with anthropic_client.messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=256,
        system=SYSTEM,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(review_text=review_text)}],
    ) as stream:
        text_parts = []
        for text in stream.text_stream:
            if rec.ttft_ms == 0.0:
                rec.ttft_ms = (time.perf_counter() - start) * 1000
            text_parts.append(text)
        final = stream.get_final_message()

    return "".join(text_parts), final.usage


# %% [markdown]
# ## Warm-up
# One discarded call per provider. First-request latency often includes route
# resolution / connection setup / model warm-up that would otherwise skew p95.

# %%
warm_prompt = "Say hi."
_ = openai_client.chat.completions.create(
    model=OPENAI_MODEL, max_completion_tokens=8,
    messages=[{"role": "user", "content": warm_prompt}],
)
_ = anthropic_client.messages.create(
    model=ANTHROPIC_MODEL, max_tokens=8,
    messages=[{"role": "user", "content": warm_prompt}],
)
print("Warm-up complete")


# %% [markdown]
# ## Run the streaming benchmark

# %%
reset()

for row in reviews:
    # --- OpenAI streaming ---
    with bench(f"openai/{OPENAI_MODEL} — streaming", OPENAI_MODEL, "openai") as rec:
        try:
            text, usage = run_openai_stream(row["text"], rec)
            if usage is not None:
                rec.input_tokens = usage.prompt_tokens
                rec.output_tokens = usage.completion_tokens
            rec.ok = bool(text.strip()) and rec.ttft_ms > 0
            rec.extra["text"] = text
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"[:200]

    # --- Anthropic streaming ---
    with bench(f"anthropic/{ANTHROPIC_MODEL} — streaming", ANTHROPIC_MODEL, "anthropic") as rec:
        try:
            text, usage = run_anthropic_stream(row["text"], rec)
            rec.input_tokens = usage.input_tokens
            rec.output_tokens = usage.output_tokens
            rec.ok = bool(text.strip()) and rec.ttft_ms > 0
            rec.extra["text"] = text
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"[:200]

print(f"Collected {len(RESULTS)} records")


# %% [markdown]
# ## Aggregate summary

# %%
df = summarise()
df


# %% [markdown]
# ## Steady-state throughput
#
# `tokens/sec = output_tokens / (total_time - ttft)` — the rate of token generation
# once the model has started producing output, excluding the time-to-first-token
# latency. Useful for estimating streaming bandwidth to a UI.

# %%
rows = []
for r in RESULTS:
    gen_time_s = max((r.latency_ms - r.ttft_ms) / 1000.0, 1e-6)
    rows.append({
        "provider": r.provider,
        "model": r.model,
        "ttft_ms": round(r.ttft_ms, 1),
        "total_ms": round(r.latency_ms, 1),
        "gen_time_ms": round(r.latency_ms - r.ttft_ms, 1),
        "output_tokens": r.output_tokens,
        "tokens_per_sec": round(r.output_tokens / gen_time_s, 1),
    })
throughput = pd.DataFrame(rows)
throughput.groupby(["provider", "model"]).agg(
    mean_ttft_ms=("ttft_ms", "mean"),
    mean_total_ms=("total_ms", "mean"),
    mean_gen_time_ms=("gen_time_ms", "mean"),
    mean_output_tokens=("output_tokens", "mean"),
    mean_tokens_per_sec=("tokens_per_sec", "mean"),
).round(1)


# %% [markdown]
# ## Per-call detail
#
# Per-review view: which provider showed the first character sooner on each review.
# The per-call TTFT is noisy at N=20 but directional — pay attention to the
# **p95/worst-case** more than the p50.

# %%
by_review = []
oa_records = [r for r in RESULTS if r.provider == "openai"]
an_records = [r for r in RESULTS if r.provider == "anthropic"]
for i, review in enumerate(reviews):
    by_review.append({
        "id": review["id"],
        "text": review["text"][:60] + "…",
        "openai_ttft_ms": round(oa_records[i].ttft_ms, 1),
        "anthropic_ttft_ms": round(an_records[i].ttft_ms, 1),
        "openai_total_ms": round(oa_records[i].latency_ms, 1),
        "anthropic_total_ms": round(an_records[i].latency_ms, 1),
    })
pd.DataFrame(by_review)
