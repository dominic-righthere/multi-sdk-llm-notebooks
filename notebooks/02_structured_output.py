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
# # 02 — Structured output, OpenAI vs Anthropic
#
# Same task and data as notebook 01, but using each provider's *structured output* feature instead of tool use.
#
# - **OpenAI:** `response_format={"type": "json_schema", "json_schema": ...}` with `strict: True`.
# - **Anthropic:** prompt engineering + response-prefill to force JSON. (Anthropic's tool-use path is their recommended structured-output mechanism, but this notebook uses the freeform path for contrast with tool calling in notebook 01.)

# %%
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from src.bench import RESULTS, bench, reset, summarise
from src.prompts import SCHEMA, SYSTEM_PROMPT, USER_TEMPLATE

load_dotenv()

# %%
OPENAI_MODEL = "gpt-5.4-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "reviews.json"
reviews = json.loads(DATA_PATH.read_text())

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
# ## OpenAI — `response_format=json_schema` (strict mode)

# %%
from openai import OpenAI  # noqa: E402

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def run_openai(review_text: str) -> tuple[dict | None, object]:
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(review_text=review_text)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "classify_review",
                "schema": SCHEMA,
                "strict": True,
            },
        },
        temperature=0,
    )

    content = resp.choices[0].message.content
    if not content:
        return None, resp.usage

    return json.loads(content), resp.usage


# %% [markdown]
# ## Anthropic — prompt engineering + response prefill
#
# Anthropic does not expose a direct JSON-schema-constrained mode for freeform text;
# the recommended path is tool use (covered in notebook 01). This notebook approximates
# structured output with a carefully worded system prompt and response prefill, to
# contrast with the native tool-use approach.

# %%
from anthropic import Anthropic  # noqa: E402

anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

ANTHROPIC_SYSTEM = SYSTEM_PROMPT + f"""

You must reply with a JSON object matching this schema exactly. No prose, no markdown fences.
Schema:
{json.dumps(SCHEMA, indent=2)}
"""


def run_anthropic(review_text: str) -> tuple[dict | None, object]:
    resp = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=512,
        system=ANTHROPIC_SYSTEM,
        messages=[
            {"role": "user", "content": USER_TEMPLATE.format(review_text=review_text)},
            {"role": "assistant", "content": "{"},  # prefill forces JSON start
        ],
        temperature=0,
    )

    # Reconstruct full JSON by prepending the prefilled "{".
    text_blocks = [b.text for b in resp.content if b.type == "text"]
    raw = "{" + "".join(text_blocks)
    try:
        return json.loads(raw), resp.usage
    except json.JSONDecodeError:
        # Attempt to trim trailing content if Claude continued past the JSON.
        end = raw.rfind("}")
        if end > 0:
            try:
                return json.loads(raw[: end + 1]), resp.usage
            except json.JSONDecodeError:
                pass
        return None, resp.usage


# %% [markdown]
# ## Run the benchmark

# %%
reset()

for row in reviews:
    with bench(f"openai/{OPENAI_MODEL} — json_schema", OPENAI_MODEL, "openai") as rec:
        try:
            obj, usage = run_openai(row["text"])
            rec.input_tokens = usage.prompt_tokens
            rec.output_tokens = usage.completion_tokens
            if obj is not None:
                ok, err = validate(obj)
                rec.ok = ok
                if err:
                    rec.error = err[:200]
                rec.extra["result"] = obj
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"[:200]

    with bench(f"anthropic/{ANTHROPIC_MODEL} — prefill_json", ANTHROPIC_MODEL, "anthropic") as rec:
        try:
            obj, usage = run_anthropic(row["text"])
            rec.input_tokens = usage.input_tokens
            rec.output_tokens = usage.output_tokens
            if obj is not None:
                ok, err = validate(obj)
                rec.ok = ok
                if err:
                    rec.error = err[:200]
                rec.extra["result"] = obj
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"[:200]

print(f"Collected {len(RESULTS)} records")


# %% [markdown]
# ## Results

# %%
df = summarise()
df


# %% [markdown]
# ## Compare with notebook 01 (tool use) — fill in after running both
#
# Key questions:
# - Was OpenAI's strict `json_schema` mode more or less reliable than tool calling?
# - For Anthropic, how often did prefill-based JSON fail vs native tool use?
# - Which approach is cheaper per successful call (tokens × validity rate)?
# - Which feels more production-ready when you factor in observability and error handling?
#
# The headline comparison goes into the repo README's findings table.
