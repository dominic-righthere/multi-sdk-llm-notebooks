# Practical implications

What the [findings](./FINDINGS.md) actually mean for someone building a production LLM system. Organized by the decision you're trying to make. Each takeaway links back to the finding that supports it.

If you're skimming, the most important calls are these three:

1. **For agent deployments, the metric that matters is cost-per-successful-task, not cost-per-call.** Per-call comparisons mislead because agents take 3–7 turns. (Finding 7)
2. **Anthropic prompt caching is a fix for naturally-large prompts, not a lever to pull whenever.** Padding a small prompt to clear the cache minimum is net-negative. (Finding 8)
3. **Default to structured output over tool use.** Tools cost more on both providers — significantly so on Anthropic — without being more reliable. (Findings 1 + 3)

Everything below expands on those plus context on when each applies.

---

## If you're building a classification or extraction pipeline

- **Default to structured output, not tool use.** Tool schemas inflate Anthropic input tokens ~3.4× and OpenAI ~2×; structured output has neither problem and is at least as reliable on simple schemas. Reserve tools for when you genuinely need a name-bound function in an agent loop. ([Findings 1 + 3](./FINDINGS.md))
- **Trust strict modes for flat schemas, measure for nested ones.** 100% validity at N=20 on a 3-field schema doesn't generalize to depth-3 nested objects, optional fields with constraints, or mutually-exclusive unions. Build retry logic but don't expect to use it unless your schema is genuinely nasty. ([Finding 4](./FINDINGS.md))
- **Don't add Anthropic prompt caching to a small classification prompt.** If your system prompt is 200 tokens, caching can't fire (4096-token Haiku 4.5 minimum, 2048-token Sonnet 4.6 minimum) — adding `cache_control` is a no-op. Padding the prompt just to enable it makes things *worse*. ([Findings 5 + 8](./FINDINGS.md))

## If you're building a chat or streaming UI

- **Pick your latency budget axis first.** Watching-with-spinner UX = p50 wins. Hard SLA budgets = p95 wins. Long-output streaming = throughput wins. The right provider changes per axis. The shape is durable; the specific numbers age fast. ([Findings 2 + 6](./FINDINGS.md))
- **Stream from day one.** Even for thin chat, p95 differences are a meaningful fraction of perceived latency. Streaming is also where TTFT becomes a designable metric. ([Finding 6](./FINDINGS.md))
- **Don't pick a provider based on someone else's latency screenshot — including the ones in this repo.** Run your own measurement at your own region/load profile, and design for failover. Both providers will have bad days. ([Findings 2 + 6](./FINDINGS.md))

## If you're building an agent

- **Cost-per-successful-task, not cost-per-call.** A model that takes 6 turns at $0.002/turn is more expensive than one that takes 3 turns at $0.003/turn even though its per-call rate is lower. Designing tools that minimize round-trips (`compare(many)` rather than `get_reviews(one)` × N) saves more than provider switching. ([Finding 7](./FINDINGS.md))
- **Anthropic agent loops are ~3× more expensive than OpenAI at small/medium prompt size.** This is a structural token-counting difference (Finding 1) compounded across turns. If you're cost-constrained and the workload doesn't *specifically* need Anthropic-only capabilities, OpenAI is the cheaper agent host at this size. ([Finding 7](./FINDINGS.md))
- **Anthropic prompt caching helps if your prompt is naturally large; leave it off otherwise.** A real production agent with a multi-thousand-token rubric and few-shots will cache reliably and pay off. A minimal-prompt agent doesn't benefit and pads-to-cache as a net loss. ([Finding 8](./FINDINGS.md))
- **The 5-minute TTL is a real operational constraint.** Bursty or batch agent workflows with gaps >5 min between turns lose the cache. Anthropic's 1-hour TTL (`cache_control: {ttl: "1h"}`, 2× write cost) is worth using if that's your shape. ([Finding 8](./FINDINGS.md))
- **OpenAI's automatic prefix caching is the no-op cost win.** Don't put `datetime.now()` in your system prompt and you'll get auto-caching for free. Many teams don't realize this fires automatically. ([Finding 8](./FINDINGS.md))
- **For prototyping and dev-velocity-heavy agent work, use the native runner.** `openai-agents` and Anthropic's `client.beta.messages.tool_runner` cut implementation code 56–61% with identical task-correctness. The runtime cost premium is 14–29% — usually invisible at prototype scale. ([Finding 9](./FINDINGS.md))
- **For high-volume production agents, hand-roll the loop.** The same runners cost 14–29% more per successful task because they add a wrap-up turn after `finalize` that the cost-per-call view doesn't capture. At 1M tasks/month that delta is real money. ([Finding 9](./FINDINGS.md))
- **Test your runner code in the deployment environment.** `Runner.run_sync` doesn't work inside Jupyter without `loop.run_until_complete` shimming. Anthropic's `@beta_tool` rejects dict returns from tool functions (400 error) — must JSON-serialize. Both gotchas are silent until you ship. ([Finding 9](./FINDINGS.md))

## When migrating between providers

- **Watch for silent API parameter breaks.** `max_tokens` is rejected by gpt-5.4-mini in favor of `max_completion_tokens`; the same SDK version accepts both, but the model server rejects the old name. Same kind of break to expect on any frontier-model swap. (Finding 6 sidebar)
- **Re-measure cost when changing prompt shape.** A prompt that's cheap on OpenAI may be expensive on Anthropic and vice versa. Migration cost-modeling that assumes "tokens are tokens" is wrong. ([Findings 1 + 8](./FINDINGS.md))
- **Cache invalidation is a deployment concern.** Changing models invalidates cached prefixes; changing the system prompt invalidates everything after it. Plan for the cost spike on rollout. ([Finding 5](./FINDINGS.md))

---

## Findings to weight low for production decisions

These are real measurements but should not drive durable architecture choices.

- **The specific TTFT and throughput numbers** ([Findings 2 + 6](./FINDINGS.md)) are server weather. The *shape* of the comparison (p50 vs p95 differ; throughput differs from latency) is what's useful. Don't pick a provider on these numbers — measure your own.
- **The 19/20 sentiment agreement** ([Finding 4 sub-finding](./FINDINGS.md)) measures inter-provider agreement, not accuracy. Without ground truth labels, this tells you the providers behave similarly on simple sentiment — not that either is correct.
- **The 100% success rate on the agent benchmark** ([Finding 7](./FINDINGS.md)) is a ceiling effect from task simplicity. It does not mean current-gen small models won't fail on harder agent tasks; it means the benchmark didn't separate them on this dimension.

## Anti-patterns to avoid

These are negative findings — patterns that look reasonable but the measurements don't support.

- **"Cache everything to save money."** Below the per-model cache minimum, the marker is silently inert. Above it, savings depend on prefix size as a fraction of input. Multi-turn agent loops dilute the saving. ([Findings 5 + 8](./FINDINGS.md))
- **"Use tool calling because it's more reliable than JSON output."** Tool calling costs more without being more reliable on simple schemas. ([Finding 3](./FINDINGS.md))
- **"OpenAI is cheaper for agents."** True at small prompt size; the gap shrinks at large prompt size and may invert if Anthropic-specific capabilities matter for the workload. The right answer is "depends on your prompt shape." ([Findings 7 + 8](./FINDINGS.md))
- **"p50 latency is good enough as a UX metric."** p95 is where the visible pauses happen, and it diverges substantially from p50 on both providers. ([Finding 6](./FINDINGS.md))

---

## How to use this document in practice

When making a build vs. buy or provider decision:

1. **Identify what you're optimizing for** — cost-per-task, p95 latency, schema reliability, or capability access.
2. **Find the relevant section above.** Most decisions cluster into one of the three product types (pipeline / chat / agent).
3. **Run the equivalent measurement on your own workload** — the harness in `src/agent_harness.py` and `src/bench.py` is reusable. Your numbers will differ; the *shape* of the trade-offs is more transferable than the magnitudes.
4. **Don't generalize from a single benchmark run.** A finding worth shipping on is one you've reproduced or seen reproduced. The findings in this repo are deliberately measured at small N (18–20); production decisions should use larger samples and confidence intervals.

The point of measuring isn't to get a definitive answer. It's to know which questions to ask next.
