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
# # 01 — Function calling / tool use, OpenAI vs Anthropic
#
# Same task, same prompts, same data. Two SDKs. Measure latency, tokens, cost, validity.
#
# **Task:** for each product review, classify sentiment, extract key features, and estimate a 1-5 star rating — returned via a tool call / function call in both providers.

# %%
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from src.bench import RESULTS, bench, reset, summarise
from src.prompts import ANTHROPIC_TOOL, OPENAI_TOOL, SYSTEM_PROMPT, USER_TEMPLATE

load_dotenv()

# %%
# Models to test. Swap as you like — bench.py has pricing for the common picks.
OPENAI_MODEL = "gpt-5.4-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"

# %%
# Load reviews. Works whether the notebook is executed from the repo root or from notebooks/.
DATA_PATH = next(p for p in (Path("data/reviews.json"), Path("../data/reviews.json")) if p.exists())
reviews = json.loads(DATA_PATH.read_text())
print(f"Loaded {len(reviews)} reviews")


# %% [markdown]
# ## Pydantic schema for validation
# Same schema on both sides so the "did the model produce valid output" comparison is apples-to-apples.

# %%
class ReviewClassification(BaseModel):
    sentiment: str = Field(pattern=r"^(positive|negative|neutral|mixed)$")
    key_features: list[str]
    rating_estimate: int = Field(ge=1, le=5)


def validate(obj: dict) -> tuple[bool, str | None]:
    try:
        ReviewClassification.model_validate(obj)
        return True, None
    except ValidationError as e:
        return False, str(e)


# %% [markdown]
# ## OpenAI implementation

# %%
from openai import OpenAI  # noqa: E402

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def run_openai(review_text: str) -> tuple[dict | None, object]:
    """Returns (parsed_args, usage_obj)."""
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(review_text=review_text)},
        ],
        tools=[OPENAI_TOOL],
        tool_choice={"type": "function", "function": {"name": "classify_review"}},
        temperature=0,
    )

    tool_calls = resp.choices[0].message.tool_calls or []
    if not tool_calls:
        return None, resp.usage

    args = json.loads(tool_calls[0].function.arguments)
    return args, resp.usage


# %% [markdown]
# ## Anthropic implementation

# %%
from anthropic import Anthropic  # noqa: E402

anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def run_anthropic(review_text: str) -> tuple[dict | None, object]:
    """Returns (parsed_args, usage_obj)."""
    resp = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(review_text=review_text)}],
        tools=[ANTHROPIC_TOOL],
        tool_choice={"type": "tool", "name": "classify_review"},
        temperature=0,
    )

    tool_uses = [b for b in resp.content if b.type == "tool_use"]
    if not tool_uses:
        return None, resp.usage

    return tool_uses[0].input, resp.usage


# %% [markdown]
# ## Run the benchmark

# %%
reset()

for row in reviews:
    # --- OpenAI ---
    with bench(f"openai/{OPENAI_MODEL} — fn_call", OPENAI_MODEL, "openai") as rec:
        try:
            args, usage = run_openai(row["text"])
            rec.input_tokens = usage.prompt_tokens
            rec.output_tokens = usage.completion_tokens
            if args is not None:
                ok, err = validate(args)
                rec.ok = ok
                if err:
                    rec.error = err[:200]
                rec.extra["result"] = args
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"[:200]

    # --- Anthropic ---
    with bench(f"anthropic/{ANTHROPIC_MODEL} — fn_call", ANTHROPIC_MODEL, "anthropic") as rec:
        try:
            args, usage = run_anthropic(row["text"])
            rec.input_tokens = usage.input_tokens
            rec.output_tokens = usage.output_tokens
            if args is not None:
                ok, err = validate(args)
                rec.ok = ok
                if err:
                    rec.error = err[:200]
                rec.extra["result"] = args
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"[:200]

print(f"Collected {len(RESULTS)} records")


# %% [markdown]
# ## Results

# %%
df = summarise()
df

# %%
# Per-review spot check — does one provider disagree with the other on sentiment?
rows = []
for i, r in enumerate(reviews):
    oa = RESULTS[2 * i]
    an = RESULTS[2 * i + 1]
    rows.append({
        "id": r["id"],
        "text": r["text"][:80] + "…",
        "openai_sentiment": oa.extra.get("result", {}).get("sentiment"),
        "anthropic_sentiment": an.extra.get("result", {}).get("sentiment"),
        "agree": (oa.extra.get("result", {}).get("sentiment") == an.extra.get("result", {}).get("sentiment")),
    })

pd.DataFrame(rows)


# %% [markdown]
# ## Observations — fill in after running
#
# - Which provider was faster on p50? On p95?
# - Did either fail to produce valid JSON?
# - Cost per 1k calls at posted pricing?
# - Any category of review (e.g., mixed sentiment) where one model disagreed with the other?
# - Developer ergonomics: which SDK was easier to set up, error-handle, debug?
#
# (These observations become the README summary and the talking points in interviews.)
