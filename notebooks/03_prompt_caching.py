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
# # 03 — Prompt caching on Anthropic
#
# Notebook 01 found that Anthropic's tool schema adds ~577 extra input tokens per call
# vs OpenAI, inflating per-call cost ~3.4x on a small classification workload.
# The obvious follow-up: **Anthropic's prompt caching is designed for exactly this static content.
# Does it close the gap?**
#
# This notebook measures a realistic production shape: a detailed classification rubric
# plus the `classify_review` tool schema, cached via `cache_control`. We compare:
#
# - **Baseline** — same request, no cache_control. Every call pays full input price.
# - **Cached** — `cache_control: {type: "ephemeral"}` on the last system block.
#   First call writes cache (priced at **1.25x** input). Subsequent calls within the
#   5-minute TTL read from cache (priced at **0.1x** input).
#
# One gotcha to flag up front: **Haiku 4.5 requires a minimum 4096-token prefix to cache.**
# A bare tool schema is ~250 tokens — far below the threshold, so caching silently
# does nothing. You have to earn caching by having real static content (rubric,
# few-shots, guidelines) up front. This turns out to match production reality: the
# system prompts worth caching are the ones that are already large.

# %%
import json
import os
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from src.bench import RESULTS, bench, reset, summarise
from src.prompts import SCHEMA, USER_TEMPLATE

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-haiku-4-5"

DATA_PATH = next(p for p in (Path("data/reviews.json"), Path("../data/reviews.json")) if p.exists())
reviews = json.loads(DATA_PATH.read_text())
print(f"Loaded {len(reviews)} reviews")


# %% [markdown]
# ## The padded rubric
#
# A realistic production system prompt: classification rubric, rating anchors,
# feature-extraction guidelines, edge cases, and few-shot examples. Designed to
# exceed Haiku 4.5's 4096-token cache minimum.

# %%
RUBRIC = """You are a senior product review analyst. Read a product review and produce a
structured classification with three fields: sentiment, key_features, and rating_estimate.
This document defines the rubric.

# Sentiment classification

The sentiment field must be exactly one of: "positive", "negative", "neutral", "mixed".

## positive
Use "positive" when the review is overwhelmingly favorable. The reviewer recommends the
product, expresses satisfaction, and any complaints are incidental or do not meaningfully
affect their experience. A review that says "great, although the packaging was a bit much"
is positive — the complaint is not about the product itself. A review that says "love it,
would buy again" with no qualifiers is clearly positive.

Indicators of positive:
- Explicit recommendation ("would buy again", "highly recommend", "five stars")
- Expressions of satisfaction with no significant caveats
- A single minor complaint inside an otherwise favorable review
- Purchase repeat intent
- Favorable comparisons to alternatives

## negative
Use "negative" when the review is overwhelmingly unfavorable. The reviewer would not
recommend the product, experienced meaningful problems, or explicitly states regret.
A review that says "returned it, unusable" is negative. A review that says "mediocre"
but with pages of complaints is negative despite the tempered opening.

Indicators of negative:
- Explicit regret ("do not buy", "wish I hadn't", "returning it")
- Return or refund mentions
- Multiple structural complaints (not nit-picks)
- Product failed to perform its core function
- Unfavorable comparisons to alternatives or to earlier versions

## neutral
Use "neutral" when the review is genuinely unopinionated. The reviewer states facts,
gives a lukewarm verdict, or doesn't express clear approval or disapproval. This is
rare — most reviewers take a position, even implicitly.

Indicators of neutral:
- "It works", "does the job", "no opinion"
- Purely factual descriptions without affect
- "Fine", "okay", "meh" with no accompanying positives or negatives
- Very short reviews with no clear direction

## mixed
Use "mixed" when the review genuinely balances substantive positives and substantive
negatives. The positives and negatives are both about the product itself and neither
clearly wins. A review that says "great sound, terrible battery" is mixed if both
observations carry equal weight in the reviewer's verdict.

Indicators of mixed:
- Explicit framing ("mixed feelings", "love X but hate Y", "torn on this")
- At least one meaningful positive AND one meaningful negative, both about the product
- No clear overall verdict — the review ends ambivalently
- Reviewer would partially recommend (for some uses) but not others

# Distinguishing positive from mixed

The single most common calibration error is labeling a positive-with-one-complaint
review as "mixed". Use this rule:

- If the reviewer clearly recommends the product despite the complaint → "positive"
- If the reviewer's verdict is truly split → "mixed"

Example: "The battery life is incredible, build quality feels premium, only gripe is
the camera in low light." The complaint is one of several observations and the reviewer
has no overall reservation. This is "positive", not "mixed".

Example: "Mixed feelings. The camera is outstanding but the phone overheats during
gaming." The reviewer explicitly frames this as mixed and the negative is substantive.
This is "mixed".

# key_features extraction

The key_features field is an array of short strings describing product features or
aspects the reviewer mentioned. Each string should be a single feature, 1-5 words,
lowercase unless a proper noun.

Guidelines:
- Extract features, not opinions. "battery life" not "great battery life".
- Include both positive and negative features. "build quality", "camera low-light".
- Do not invent features the reviewer did not mention.
- Do not include product category ("laptop") — focus on attributes of the product.
- Deduplicate. If the reviewer mentions "battery" twice, include it once.
- Order by order-of-mention in the review.
- Keep features concrete. Prefer "screen brightness" over "display".
- If the review mentions price/value, include "price" or "value".
- Exclude seller/shipping attributes unless the reviewer makes them central.

Good feature examples:
- "battery life"
- "build quality"
- "noise cancellation"
- "app quality"
- "customer service"
- "keyboard"
- "trackpad"
- "screen brightness"
- "stitching"
- "fit"
- "gaming performance"
- "thermal performance"
- "setup"
- "compatibility"

Bad feature examples:
- "the battery is great" (includes opinion, not just a feature)
- "laptop" (product category)
- "me" (not a feature)
- "buy" (not a feature)
- "amazing" (an adjective, not a feature)
- "delivery" (seller attribute, not product)

# rating_estimate

The rating_estimate field is an integer 1-5 representing the reviewer's likely star
rating, inferred from sentiment and severity.

Anchors:
- 1: Would return, would not recommend to anyone. Product fundamentally failed.
- 2: Notable problems, would not buy again, but not a complete failure. Returning plausible.
- 3: Lukewarm. Middle-of-the-road. Genuine neutral or mildly negative.
- 4: Satisfied overall with minor complaints. Would recommend with caveats. Most "positive" reviews land here.
- 5: Overwhelmingly positive, explicit "five stars" or equivalent, unreserved recommendation.

Guidance:
- "It's fine, does what it says" → 3
- "Love it, only minor gripe is X" → 4 (not 5 — there is a reservation)
- "Five stars, no complaints" → 5
- "Returning it, buggy and unusable" → 1 or 2
- "Some good, some bad, I'm torn" → 3
- "Meh. Works. On sale so I can't complain" → 3
- Mixed reviews with roughly equal positives and negatives → 3
- Mixed reviews tipping positive → 4
- Mixed reviews tipping negative → 2

# Edge cases

## Sarcasm
If the review is clearly sarcastic ("great, another product that breaks in a week"),
classify on the underlying intent. Sarcasm usually maps to negative.

## Comparative reviews
If the reviewer compares the product favorably or unfavorably to another product,
treat the comparison as an observation about THIS product. "Better than my old pair"
is a positive observation about this product's relevant feature.

## Review mentions price
Price is a legitimate product feature. Include "price" or "value" in key_features if
the reviewer mentions it. "Great for the price" is positive with feature "value".

## Review mentions seller/delivery/return process
These are not product features by default. Do not include "shipping", "packaging",
"return process" in key_features unless the reviewer makes them central to the review
(e.g., "refund took four weeks" mentioned as the main complaint).

## Review is very short
A one-line review ("works fine") should be classified as neutral. A one-line explicit
positive ("great!") is positive with no features extracted. A one-line negative
("garbage") is negative with no features extracted. Always extract whatever features
the short review mentions — which may be zero.

## Mixed but tipping one way
Some reviews are mostly positive with one significant complaint, or mostly negative
with one redeeming feature. If the reviewer's overall verdict clearly tips one way,
classify that way and rate accordingly. Reserve "mixed" for genuinely balanced reviews
where no direction dominates.

## Explicit star rating in text
If the reviewer says "five stars" or "two stars" in their review, that is very strong
evidence for the rating_estimate. Align with it unless the written content contradicts.

## Deal-breakers mentioned
If the reviewer identifies a single deal-breaking issue despite other positives, weight
it heavily. "Love everything about it but the proprietary cable is a dealbreaker" tips
toward mixed or negative.

# Few-shot examples

Example 1:
Review: "The battery life is incredible — I went three days without charging. Build
quality feels premium, screen is sharp. Only gripe is the camera in low light."
Classification:
  sentiment: positive
  key_features: ["battery life", "build quality", "screen", "camera low-light performance"]
  rating_estimate: 4
Rationale: Three strong positives (battery, build, screen), one scoped negative (low-light
camera). The reviewer does not retract recommendation. 4 stars because there is a real
complaint, not unconditional praise.

Example 2:
Review: "Returned it within a week. Software is buggy, keyboard feels cheap, and the fan
noise under any load is unbearable. Not worth the price."
Classification:
  sentiment: negative
  key_features: ["software quality", "keyboard", "fan noise", "price"]
  rating_estimate: 1
Rationale: The reviewer returned the product and called out multiple structural issues.
This is a strong negative.

Example 3:
Review: "It's fine. Does what it says on the box. Nothing exciting. I wouldn't buy again
but I wouldn't warn a friend away either."
Classification:
  sentiment: neutral
  key_features: []
  rating_estimate: 3
Rationale: No features explicitly mentioned — the reviewer is lukewarm and non-specific.
Truly neutral.

Example 4:
Review: "Mixed feelings. The camera is outstanding but the phone overheats during gaming.
Battery drains fast with the always-on display enabled."
Classification:
  sentiment: mixed
  key_features: ["camera", "overheating during gaming", "battery life", "always-on display"]
  rating_estimate: 3
Rationale: Explicitly framed as mixed. Strong positive (camera) balanced by two
structural negatives (overheating, battery).

Example 5:
Review: "Five stars. Fast delivery, easy setup, exactly as described. Works flawlessly
with my existing home setup. Would buy again."
Classification:
  sentiment: positive
  key_features: ["setup", "compatibility", "reliability"]
  rating_estimate: 5
Rationale: Explicit five-star recommendation, no complaints, purchase repeat intent.
(Note: "fast delivery" is a seller attribute, not a product feature, and was not
extracted.)

Example 6:
Review: "Perfect for the price. Runs every game I throw at it at medium settings. Case
design is boring but thermals are surprisingly good."
Classification:
  sentiment: positive
  key_features: ["price", "gaming performance", "case design", "thermal performance"]
  rating_estimate: 4
Rationale: Multiple positives (price, gaming, thermals) with one aesthetic complaint
(case design). Positive overall; the design nitpick prevents a perfect 5.

Example 7:
Review: "Do not buy. Arrived damaged, replacement also had a scratched screen, and
returns took four weeks to refund. Never again with this brand."
Classification:
  sentiment: negative
  key_features: ["physical damage", "screen", "customer service"]
  rating_estimate: 1
Rationale: Repeated failures on arrival, multi-week refund delay, explicit "never again".
Extremely negative. "Customer service" is included because the reviewer makes the refund
process central to the complaint.

Example 8:
Review: "The stitching and leather feel premium, battery hits my expectations, but the
proprietary charging cable is a dealbreaker. Lost mine on a trip."
Classification:
  sentiment: mixed
  key_features: ["stitching", "leather quality", "battery life", "proprietary charging cable"]
  rating_estimate: 3
Rationale: Real positives (stitching, leather, battery) offset by a genuine dealbreaker.
The reviewer uses "dealbreaker" explicitly, tipping mixed toward the lower end of 3.

Example 9:
Review: "Meh. Works. Not as loud as advertised, bass is muddy, but it was on sale so I
can't complain too much."
Classification:
  sentiment: neutral
  key_features: ["volume", "bass quality", "price"]
  rating_estimate: 3
Rationale: Reviewer is unenthusiastic but not upset. Mild complaints softened by price
acknowledgment. Genuinely neutral, not mixed — there are no real positives, just
tempered negatives.

Example 10:
Review: "Very happy. Installed in 10 minutes, app works on both iOS and Android, and
the motion detection is accurate. Night vision is grainy but usable."
Classification:
  sentiment: positive
  key_features: ["setup", "app quality", "motion detection accuracy", "night vision"]
  rating_estimate: 4
Rationale: Clear positive framing ("very happy"), multiple positives (setup, app,
motion detection), one scoped complaint (night vision). 4 because of the complaint,
even though overall very positive.

# Output format

You must call the classify_review tool with a valid JSON object matching the schema.
Do not include prose, markdown, or any text outside the tool call. All three fields
are required. sentiment must be an enum value. rating_estimate must be an integer 1-5.
key_features must be an array of strings (empty array is allowed if the reviewer did
not mention specific features).

Do not invent features. Do not include features the reviewer did not mention even if
they would typically be relevant for this product category. The classification reflects
what the reviewer actually wrote, not what they could have written.

# Additional worked examples

Example 11:
Review: "Disappointed. The stitching came undone on the strap after two weeks. Customer
service was slow. Material feels like plastic despite 'genuine leather' claims."
Classification:
  sentiment: negative
  key_features: ["stitching", "strap durability", "customer service", "material quality"]
  rating_estimate: 1
Rationale: Product failure within two weeks, slow support, and deceptive marketing. Three
structural negatives. Strong 1-star territory.

Example 12:
Review: "Solid build, loud speakers, simple interface. Remote is clunky but I use a
universal one anyway. Picture quality is better than my previous TV at twice the price."
Classification:
  sentiment: positive
  key_features: ["build quality", "speakers", "interface", "remote", "picture quality", "price"]
  rating_estimate: 5
Rationale: Reviewer has one complaint (remote) but explicitly works around it. Everything
else is enthusiastic, including a favorable price/performance comparison. Close to 5
because the reviewer's overall verdict is "better than what I paid twice for".

Example 13:
Review: "Gorgeous screen, terrible typing experience. Key travel is too shallow, and
the trackpad registers accidental palm contact constantly. Won't keep."
Classification:
  sentiment: negative
  key_features: ["screen", "keyboard", "key travel", "trackpad palm rejection"]
  rating_estimate: 2
Rationale: One positive (screen) but two structural input-device failures plus "won't
keep" — reviewer is returning. Tips clearly negative, 2 stars because the screen
observation prevents the worst rating.

Example 14:
Review: "Ergonomic and quiet. The scroll wheel is smooth, side buttons are mapped well.
Software is the weak point — frequent crashes on Mac."
Classification:
  sentiment: mixed
  key_features: ["ergonomics", "noise level", "scroll wheel", "side buttons", "software quality", "Mac compatibility"]
  rating_estimate: 3
Rationale: Three clear hardware positives (ergonomics, quiet, scroll wheel) against one
software negative that affects daily use (frequent crashes on the reviewer's primary
platform). Software-weak-point framing is explicit; reviewer's verdict is genuinely
split. Mixed at 3.

# Common pitfalls to avoid

Do not conflate "mixed" with "balanced". A review can list many attributes — some
positive, some negative — without being mixed in verdict. The question is whether the
reviewer's overall evaluation is genuinely undecided. If they would recommend the
product on balance, it's positive with features; it's not mixed.

Do not upgrade "mixed tipping positive" to 5. A 5-star rating implies no meaningful
reservation. If the reviewer has one substantive complaint they still talk about, they
are in 4-star territory at most.

Do not downgrade "positive with nitpicks" to 3. A 3 is genuinely ambivalent or
unenthusiastic. A positive review with one or two scoped complaints is usually 4.

Do not extract features from sentiment words. "Amazing", "great", "terrible" are
adjectives, not features. The feature is what the adjective modifies.

Do not extract product category. If the review is of a laptop, "laptop" is not a
feature. Focus on laptop attributes.

Preserve reviewer framing when possible. If the reviewer uses "battery life", use
"battery life" as the feature, not "battery". If the reviewer says "noise cancellation",
use that, not "noise".

Be conservative about inferring features. Only include features the reviewer explicitly
mentioned or unmistakably implied. Do not extrapolate from product category norms.
"""

TOOL = {
    "name": "classify_review",
    "description": "Classify a product review and extract features.",
    "input_schema": SCHEMA,
}

# Sanity check: confirm the cached prefix exceeds the 4096-token minimum.
tc = client.messages.count_tokens(
    model=MODEL,
    system=RUBRIC,
    messages=[{"role": "user", "content": "x"}],
    tools=[TOOL],
)
print(f"Cacheable prefix: {tc.input_tokens} tokens (must be >=4096 for Haiku 4.5 to cache)")


# %% [markdown]
# ## Baseline: no cache_control

# %%
def run_baseline(text: str):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        temperature=0,
        system=RUBRIC,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(review_text=text)}],
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "classify_review"},
    )
    tool_uses = [b for b in resp.content if b.type == "tool_use"]
    return (tool_uses[0].input if tool_uses else None), resp.usage


reset()
for row in reviews:
    with bench(f"anthropic/{MODEL} — baseline (no cache)", MODEL, "anthropic") as rec:
        try:
            obj, usage = run_baseline(row["text"])
            rec.input_tokens = usage.input_tokens
            rec.output_tokens = usage.output_tokens
            rec.cache_creation_input_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
            rec.cache_read_input_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
            rec.ok = obj is not None
            rec.extra["result"] = obj
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"[:200]


# %% [markdown]
# ## Cached: `cache_control` on the last system block
#
# Tools render before system, so a single breakpoint on the system block caches
# both tool schema and rubric as one prefix.

# %%
def run_cached(text: str):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        temperature=0,
        system=[{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": USER_TEMPLATE.format(review_text=text)}],
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "classify_review"},
    )
    tool_uses = [b for b in resp.content if b.type == "tool_use"]
    return (tool_uses[0].input if tool_uses else None), resp.usage


for row in reviews:
    with bench(f"anthropic/{MODEL} — cached", MODEL, "anthropic") as rec:
        try:
            obj, usage = run_cached(row["text"])
            rec.input_tokens = usage.input_tokens
            rec.output_tokens = usage.output_tokens
            rec.cache_creation_input_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
            rec.cache_read_input_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
            rec.ok = obj is not None
            rec.extra["result"] = obj
        except Exception as e:
            rec.error = f"{type(e).__name__}: {e}"[:200]

print(f"Collected {len(RESULTS)} records")


# %% [markdown]
# ## Aggregate summary

# %%
df = summarise()
df


# %% [markdown]
# ## Per-call breakdown of the cached run
#
# Expected pattern: call 1 shows a large `cache_creation`. Calls 2-20 show a large
# `cache_read` with near-zero uncached `input`. If `cache_read` is zero on calls 2+,
# the cache is silently not firing — most commonly because the prefix is under 4096
# tokens or a byte changed somewhere in the rendered prompt.

# %%
cached_records = [r for r in RESULTS if r.label.endswith("cached")]
per_call = pd.DataFrame([{
    "call": i + 1,
    "input_uncached": r.input_tokens,
    "cache_creation": r.cache_creation_input_tokens,
    "cache_read": r.cache_read_input_tokens,
    "output": r.output_tokens,
    "latency_ms": round(r.latency_ms, 1),
    "cost_usd": round(r.cost_usd, 6),
} for i, r in enumerate(cached_records)])
per_call


# %% [markdown]
# ## Break-even vs baseline
#
# Baseline pays full input price every call. Cached pays 1.25x on call 1, then 0.1x
# thereafter for the cached portion. Plot cumulative cost over the sequence to see
# where caching overtakes.

# %%
baseline_records = [r for r in RESULTS if r.label.endswith("baseline (no cache)")]
cum = pd.DataFrame({
    "call": range(1, len(baseline_records) + 1),
    "baseline_cumulative_usd": pd.Series([r.cost_usd for r in baseline_records]).cumsum(),
    "cached_cumulative_usd": pd.Series([r.cost_usd for r in cached_records]).cumsum(),
})
cum["cached_vs_baseline_pct"] = (cum["cached_cumulative_usd"] / cum["baseline_cumulative_usd"] * 100).round(1)
cum
