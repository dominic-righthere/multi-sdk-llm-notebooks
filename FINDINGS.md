# What I learned running the same task through the OpenAI and Anthropic SDKs

*A writeup of [`multi-sdk-llm-notebooks`](https://github.com/dominic-righthere/multi-sdk-llm-notebooks) — last run 2026-04-24.*

Most "OpenAI vs Anthropic" benchmarks you see online are either flame-war shaped or cherry-picked for a blog post. I wanted to answer a narrower question: **for the same task, at the same temperature, with the same prompt, what do the two SDKs actually cost and cost-in-latency to use?** Not "which model is smarter" — that changes every six weeks. Something more structural.

So I built a small harness, ran 20 product reviews through both providers twice (once via tool use, once via structured output), and wrote down what the SDKs told me about themselves along the way.

Headline numbers are in the [README](./README.md). This post is about what they *mean*.

## Setup

- **Task.** Classify a product review: `sentiment` (positive / negative / neutral / mixed), `key_features` (list of strings), `rating_estimate` (1–5). Three fields, one schema, `temperature=0`.
- **Data.** 20 synthetic product reviews — laptops, headphones, home goods, clothing — hand-written to cover the obvious cases and a few edge ones.
- **Models.** `gpt-5.4-mini` and `claude-haiku-4-5`. Cheap-tier on both sides, comparable market positioning.
- **Two runs each.**
  - *Notebook 01* uses tool calling / function calling — the providers' recommended structured-data path.
  - *Notebook 02* uses each provider's structured output mechanism: OpenAI's strict `response_format={"type": "json_schema", ...}` and Anthropic's prompt-engineering + response-prefill approach.
- **Harness.** A `bench()` context manager that wraps each SDK call, times it, pulls tokens from the provider's `usage` object, and records to a dataclass. `summarise()` then aggregates into a pandas DataFrame.

The full code is in the repo. The comparison-driving insight is the harness: because both providers return a `usage` object with input/output token counts, I can compare like-for-like — without that, any cost number is a guess.

## Finding 1: Anthropic's "tool schema is input" tax

The single biggest thing I didn't expect: **Anthropic counts the tool schema as input tokens, OpenAI does not.**

| | gpt-5.4-mini | claude-haiku-4-5 |
|---|---|---|
| Mean input tokens (tool use) | 239 | 816 |
| Mean input tokens (structured output, no tool) | 144 | 288 |

Same review text. Same system prompt. The delta in the tool-use row (≈577 tokens of extra input, almost exactly the `classify_review` tool definition) shows up cleanly when you drop the tool wrapper in notebook 02.

This matters because **unit price doesn't tell the full cost story.** Haiku 4.5 at $1/$5 per MTok vs gpt-5.4-mini at $0.75/$4.50 looks like a modest premium on paper. Once you pay for the tool schema on every call, the real delta is ~3× on this workload. At 1M requests/month, that's a business decision.

Two mitigations to keep in mind if you're shipping Anthropic:

1. **Prompt caching.** The tool schema is static; it's exactly the kind of thing Anthropic's cache is designed for. I didn't measure it in this run — that's a follow-up notebook.
2. **Structured output instead of tools.** If you just want JSON out, skip the tool wrapper. Notebook 02 shows Anthropic's `prefill_json` at $0.58 / 1k vs its `fn_call` at $1.26 — 54% cheaper on the same task.

The trade is that OpenAI's `json_schema` strict mode is a real constraint-decoding contract; Anthropic's prefill is prompt engineering. For a simple schema they both hit 100% validity here. For a nested one they probably wouldn't, and that's where OpenAI's strict mode earns its keep. Measuring that is also a follow-up.

## Finding 2: "Anthropic is faster" is only half the story

On medians, Anthropic beat OpenAI on both tasks:

- Function calling: 1056 ms vs 1209 ms (p50)
- Structured output: 970 ms vs 1113 ms (p50)

But **tails diverge.** On function calling, Anthropic's p95 was 2952 ms vs OpenAI's 2629 ms — Anthropic's slowest calls were slower than OpenAI's slowest calls, even though its median was quicker.

If you're driving a user-facing UI where p95 is the budget you actually design against, the picture flips. If you're batching, it doesn't matter.

One honest caveat: **N=20, one trial.** Variance at the tail is unstable at this scale. A serious measurement would run N≥100 with at least 3 trials and report confidence intervals. This run is directionally interesting, not statistically rigorous.

## Finding 3: Structured output is ~25–55% cheaper than tool use

For the same task, on the same provider, swapping tools for the native structured-output path saved money on both sides:

| | Tool use | Structured output | Delta |
|---|---|---|---|
| OpenAI gpt-5.4-mini | $0.40 / 1k | $0.30 / 1k | −25% |
| Anthropic claude-haiku-4-5 | $1.26 / 1k | $0.58 / 1k | −54% |

Two things are happening:

- **OpenAI's savings** are modest and come from a shorter schema serialization when the tool wrapper is gone.
- **Anthropic's savings are large** because the tool schema was a large fraction of the input (finding 1).

The rule I'd take away for architecture decisions: **if your use case is "give me structured JSON," prefer structured output over tool use.** Reserve tool use for when you actually need a name-bound function in a multi-step loop. Too many teams default to tools because the tutorials lead with tools; it's often not the cheapest path.

## Finding 4: The two providers agreed 19/20 on sentiment

The one disagreement was review #1:

> *"The battery life is incredible — I went three days without charging. Build quality feels premium, screen is sharp. Only gripe is the camera in low light."*

OpenAI: `mixed`. Anthropic: `positive`. Both defensible — the review is overwhelmingly positive with one minor negative. What this really shows is that the `mixed` label in my schema is ambiguous; a better schema would either enforce a rubric for `mixed` ("at least one positive and one meaningfully negative point") or drop it and use a confidence score.

The lesson generalizes: **when your two frontier models disagree on a label, the label is usually the problem, not the models.**

## Finding 5: Prompt caching closes the gap (and halves p95 latency) — but has a silent-failure mode

This was the follow-up I promised at the end of the first run: if Finding 1 was "Anthropic's tool schema inflates input tokens ~3.4×", then Anthropic's prompt caching is the feature designed for exactly that shape. Does it close the gap?

**Yes — by more than I expected, and with a bonus latency effect I didn't expect at all.**

The setup: padded the system prompt with a realistic production-shaped rubric (classification criteria, rating anchors, edge cases, 14 few-shot examples — 4741 tokens total), added `cache_control: {type: "ephemeral"}` to the last system block, and ran the same 20 reviews through Anthropic `claude-haiku-4-5` with caching off and on.

| | Baseline (no cache) | Cached |
|---|---|---|
| Mean uncached input | 4872 tokens | 454 tokens |
| Mean cache-read | 0 | **4197 tokens** |
| Cost / 1k calls | $5.30 | **$1.58** (−70%) |
| Latency p50 | 1089 ms | 1021 ms |
| Latency p95 | 5525 ms | **1930 ms** (−65%) |
| Break-even call | — | **2** |

Three things stood out.

### The cost cliff is real but not quite "90% cheaper"

You see "prompt caching saves up to 90%" everywhere. The real number on this workload was **70%**, because:

- Only input tokens cache. Output tokens pay full price.
- The user's per-request review text doesn't cache — it's the volatile suffix.
- Cache reads are 0.1× input price, not free. 4197 tokens × 0.1× is the equivalent of ~420 uncached tokens per call.

The 90% number refers to **the cached portion of input**, not total cost. For this workload the cached portion was most of the input, so the total savings landed at 70%. For workloads where the cacheable prefix is a smaller share of each call (short system prompt, long user inputs), the savings shrink proportionally. Headline claims from vendor marketing always assume the friendliest shape; measure your own.

### Latency savings were a surprise

I went in expecting caching to be a pure cost play. It turned out to be a latency play too — p95 dropped from 5525 ms to 1930 ms. That's not token price, that's actual wall-clock time the model spends processing the prefix. When you cache, the server-side prefix processing happens once; subsequent requests skip it. The tail effect is biggest precisely because the tail is where prefix-processing variance dominates.

This matters for UX. If you're designing a user-facing flow where p95 latency is your budget, prompt caching is a lever — not just a cost optimization.

### The silent-failure mode

My first run of this notebook didn't cache. No error. No warning. Just `cache_creation_input_tokens: 0` and `cache_read_input_tokens: 0` on every call.

The prefix was 3874 tokens. Haiku 4.5's minimum cacheable size is **4096 tokens** — below that, the API silently skips caching. I added ~900 more tokens of rubric (more examples, more edge cases) and re-ran; the second run showed `cache_read: 4418` per call on calls 2-20.

This is the single most dangerous prompt-caching gotcha: when it works, it works great. When it doesn't, you silently pay full price and never notice unless you're inspecting `usage`. Three implications:

1. **A bare tool schema (~250 tokens) will never cache.** You have to pad to production reality — rubric, examples, guidelines — which is usually what real system prompts look like anyway.
2. **The minimum differs by model.** Opus 4.6, Opus 4.7, Haiku 4.5: 4096 tokens. Sonnet 4.6, Haiku 3.5: 2048. A prefix that caches on one model silently doesn't on another.
3. **Always verify.** `print(response.usage.cache_read_input_tokens)` on every cached request until you've confirmed the math. If it stays zero across multiple requests with an identical prefix, a silent invalidator is at work — minimum not hit, a `datetime.now()` in the prompt, tools reordered, something.

### What this means for Finding 1

The original cost gap in notebook 01 ($0.40/1k OpenAI vs $1.26/1k Anthropic) was on a tiny workload — 816 input tokens, all well below any cache minimum. Caching would never have helped that specific workload.

**But the finding generalizes the moment the prompt grows.** Any Anthropic workload with a static prefix ≥4096 tokens can use caching to drop per-call cost by 70%. The "tool schema inflates input" tax only survives at small prefix sizes where caching can't engage. For production-shaped prompts — where you have a real system prompt, real few-shots, real rubric — the cost picture flips toward Anthropic rather than against it.

That's the real takeaway from running this thing end-to-end: **the right question isn't "which provider is cheaper?". It's "which provider is cheaper at my prompt shape?"** The answer changes as you scale up the static portion.

---

## Finding 6: Streaming changes what latency means

Up to now every latency number in this repo has been **total request time** — the clock from `create()` call to the last byte of the response. That's the right metric for batch workloads, but the wrong one for any user-facing flow, where what actually matters is **TTFT — time to first token**. The user doesn't perceive the request as complete; they perceive it as responsive, and responsiveness is "when did something visible happen?"

So notebook 04 streams a small generative task (2-sentence review summary + sentiment label) through both providers, warm-up call included, and captures TTFT in the same `perf_counter()` clock as the total latency.

| | OpenAI gpt-5.4-mini | Anthropic claude-haiku-4-5 |
|---|---|---|
| TTFT p50 | **620 ms** | 725 ms |
| TTFT p95 | 1531 ms | **1023 ms** |
| Total p50 | **1038 ms** | 1247 ms |
| Total p95 | 2100 ms | **1524 ms** |
| Throughput (tokens/sec, steady-state) | **127** | 101 |

Three findings worth naming.

### The shape of the two providers is different

On every metric at the median, OpenAI is faster:

- First token arrives ~15% sooner.
- Once the stream starts, tokens flow ~26% faster.

But on the **p95 tail**, the picture flips completely:

- OpenAI TTFT p95 (1531 ms) is **50% worse** than Anthropic's (1023 ms).
- OpenAI total p95 (2100 ms) is **38% worse** than Anthropic's (1524 ms).

This is the kind of shape you don't see unless you measure both ends of the distribution. If a writeup reports only "OpenAI is 15% faster," it's telling you the happy-path story and hiding the tail. The per-call data (visible in the notebook) shows OpenAI has 2–3 clearly-outlier requests where TTFT took ~1500 ms; Anthropic's worst cases cluster much tighter.

### This inverts "which provider should I pick" for real UX

Depending on what you're designing for, the answer flips:

- **Chat UI with a progress spinner, user watching** — p50 TTFT is the budget. **Pick OpenAI.** User sees a response ~100 ms sooner at the median.
- **User-facing streaming under a hard SLA** — p95 TTFT is the budget. **Pick Anthropic.** Tail is 33% tighter; fewer users hit a visible pause.
- **Long generative UI (code, documents) where throughput dominates** — tokens/sec matters more than TTFT. **Pick OpenAI.** +26% steady-state throughput means the stream finishes meaningfully sooner for long outputs.

"Which is faster" is not a well-posed question. "Faster at which latency quantile, for which UX constraint?" is.

### The silent breaking change in OpenAI's SDK

Worth calling out because it cost me one failed notebook run: **`max_tokens` no longer works on gpt-5.4-mini**. The request returns a 400 with "Use 'max_completion_tokens' instead." Existing gpt-4o code migrated to gpt-5 silently stops working.

This is an API-level migration that didn't change the SDK version — the same `openai` SDK accepts both, but the gpt-5 series server-side requires the new parameter name. The kind of thing you only find out when you actually try to use the new model, not when you read the changelog.

---

## What I'd do next (and what I won't claim)

Things this run does NOT tell you, that an honest writeup should name:

- How either provider handles concurrency, rate limits, retries.
- How either provider handles long context (≥32k).
- How either does on nested or genuinely adversarial schemas.
- How either behaves in multi-turn tool-use loops, where output becomes input on every turn.
- Whether OpenAI's automatic prompt caching (`prompt_tokens_details.cached_tokens`) closes the gap symmetrically — I only measured the Anthropic side of caching.
- How TTFT and throughput change under concurrent load — single-request numbers are a floor, not a ceiling.

## What I got out of building this

Three things, honestly:

1. **Reading both SDKs in parallel is the fastest way to internalize them.** You notice what's different — the `content` block types in Anthropic's response vs the message structure in OpenAI's. What's *conspicuously* different is usually a design decision worth understanding.
2. **The `usage` object is the highest-information return value.** Every benchmark story worth telling is written in tokens. If you're designing an LLM feature and not logging `input_tokens` / `output_tokens` per call into observability, you're flying blind on unit economics.
3. **Small, opinionated benchmarks beat generic ones.** Artificial Analysis already publishes the big leaderboard. Nobody needs my version of that. What I can contribute is *this specific comparison, on this specific SDK shape, with the reasoning written down.* That's the deliverable.

---

*Repo: [`github.com/dominic-righthere/multi-sdk-llm-notebooks`](https://github.com/dominic-righthere/multi-sdk-llm-notebooks) · Dominic Lee · [domlee.dev](https://domlee.dev)*
