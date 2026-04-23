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

## What I'd do next (and what I won't claim)

Things this run does NOT tell you, that an honest writeup should name:

- How either provider handles concurrency, rate limits, retries.
- How either provider handles long context (≥32k).
- Streaming latency (TTFT vs TTLT) — different shape of question, often the one UX actually cares about.
- How prompt caching changes the Anthropic cost picture.
- How either does on nested or genuinely adversarial schemas.
- How either behaves in multi-turn tool-use loops, where output becomes input on every turn.

The next notebook I'd add is **prompt caching on Anthropic against a fixed tool schema** — because finding 1 strongly implies caching would close most of the cost gap, and measuring that is a sharper story than "Anthropic cost more."

## What I got out of building this

Three things, honestly:

1. **Reading both SDKs in parallel is the fastest way to internalize them.** You notice what's different — the `content` block types in Anthropic's response vs the message structure in OpenAI's. What's *conspicuously* different is usually a design decision worth understanding.
2. **The `usage` object is the highest-information return value.** Every benchmark story worth telling is written in tokens. If you're designing an LLM feature and not logging `input_tokens` / `output_tokens` per call into observability, you're flying blind on unit economics.
3. **Small, opinionated benchmarks beat generic ones.** Artificial Analysis already publishes the big leaderboard. Nobody needs my version of that. What I can contribute is *this specific comparison, on this specific SDK shape, with the reasoning written down.* That's the deliverable.

---

*Repo: [`github.com/dominic-righthere/multi-sdk-llm-notebooks`](https://github.com/dominic-righthere/multi-sdk-llm-notebooks) · Dominic Lee · [domlee.dev](https://domlee.dev)*
