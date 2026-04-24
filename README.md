# multi-sdk-llm-notebooks

Side-by-side benchmarks of the same LLM task implemented in the **OpenAI SDK** and the **Anthropic SDK**. Measures latency, tokens, cost, and failure modes so you can make informed provider choices.

Written as notebooks because the value is in the comparison walkthrough, not just the numbers.

## Why this exists

Most provider-comparison content is vibes or cherry-picked benchmarks. This repo runs the **same task** through both SDKs with the **same input data** and reports the deltas that actually matter when you're architecting a production LLM feature.

Current comparisons:

| Notebook | Task |
|---|---|
| `01_function_calling.ipynb` | Tool use / function calling — structured classification and extraction |
| `02_structured_output.ipynb` | JSON-schema-constrained output — same task, different mechanism |
| `03_prompt_caching.ipynb` | Anthropic prompt caching — does it close the tool-schema cost gap from notebook 01? |
| `04_streaming.ipynb` | Streaming TTFT vs total latency — the metric user-facing flows actually budget against |
| `05_agent_loop.ipynb` | Agent-loop benchmark — mock tools, deterministic ground truth, 20% error injection. Success rate, turns, cost per successful task, recovery rate. |
| `06_agent_loop_cached.ipynb` | Agent loop with padded prompt + prompt caching. Does caching close the multi-turn cost gap? (Spoiler: not really — and padding to enable caching is net-negative on Anthropic.) |

Metrics tracked per SDK per call:
- Request latency (p50, p95, mean)
- Input tokens, output tokens
- Estimated cost per 1,000 calls (at current posted pricing)
- JSON validity rate
- Schema conformance rate
- Retry count

## Quick start

Requires `uv` (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
# Clone and enter
git clone https://github.com/dominic-righthere/multi-sdk-llm-notebooks.git
cd multi-sdk-llm-notebooks

# Install dependencies
uv sync

# Set API keys
cp .env.example .env
# edit .env with your ANTHROPIC_API_KEY and OPENAI_API_KEY

# Convert jupytext .py files to notebooks (one-time)
uv run jupytext --to ipynb notebooks/*.py

# Launch Jupyter
uv run jupyter lab
```

## Repo structure

```
.
├── README.md
├── pyproject.toml              # uv-managed dependencies
├── .env.example                # API key template
├── .gitignore
├── data/
│   └── reviews.json            # 20 synthetic product reviews (public, generated)
├── src/
│   ├── __init__.py
│   ├── bench.py                # Benchmark harness (timing, cost, token tracking)
│   └── prompts.py              # Shared prompt definitions across providers
└── notebooks/
    ├── 01_function_calling.py  # Jupytext format; convert to .ipynb via `uv run jupytext`
    └── 02_structured_output.py
```

## Methodology

Each notebook runs the **same 20 product reviews** through both providers with:

- Identical system prompt and user template (`src/prompts.py`).
- Identical output schema — the only thing that differs is the provider-specific wrapping (OpenAI `tools`/`response_format` vs Anthropic `tools`/prefill).
- `temperature=0` on both sides for determinism.
- A Pydantic validator applied to the parsed response to score schema conformance independently of provider-reported "success".

Per-call metrics are captured by a `bench()` context manager in `src/bench.py`:

- **Latency** — wall-clock request time (p50, p95, mean).
- **Tokens** — input + output from each provider's usage object.
- **Cost** — tokens × posted per-token pricing snapshot (see `PRICING` in `src/bench.py`; verify before citing numbers).
- **Validity rate** — fraction of responses that both parse as JSON and pass the Pydantic schema.

Results aggregate into a pandas DataFrame via `summarise()`. Pricing, model behaviour, and SDK ergonomics all move, so this repo is structured to be re-run rather than to advertise a snapshot.

Default models (swap in the notebook's `OPENAI_MODEL` / `ANTHROPIC_MODEL` constants):

- OpenAI `gpt-5.4-mini`
- Anthropic `claude-haiku-4-5`

## Findings (run on 2026-04-24, N=20, single trial)

### Notebook 01 — Function calling / tool use

| | OpenAI `gpt-5.4-mini` | Anthropic `claude-haiku-4-5` |
|---|---|---|
| Latency p50 | 1209 ms | **1056 ms** |
| Latency p95 | **2629 ms** | 2952 ms |
| Mean input tokens | **239** | 816 |
| Mean output tokens | **48** | 88 |
| Cost / 1k calls | **$0.40** | $1.26 |
| Schema validity | 100% | 100% |

### Notebook 02 — Structured output (OpenAI strict `json_schema` vs Anthropic prefill)

| | OpenAI `gpt-5.4-mini` | Anthropic `claude-haiku-4-5` |
|---|---|---|
| Latency p50 | 1113 ms | **970 ms** |
| Latency p95 | 1731 ms | **1510 ms** |
| Mean input tokens | **144** | 288 |
| Mean output tokens | **43** | 59 |
| Cost / 1k calls | **$0.30** | $0.58 |
| Schema validity | 100% | 100% |

### Notebook 06 — Does caching close the multi-turn gap? (Deterministic)

Same harness as notebook 05, but with a padded 4387-token system prompt (rubric, tool contracts, worked examples — production-shaped). Three runs, apples-to-apples:

| Run | Cost / successful task | Ratio to OpenAI |
|---|---|---|
| OpenAI `gpt-5.4-mini` (padded, auto-cache fires 100%) | **$0.00205** | 1.0× |
| Anthropic `claude-haiku-4-5` (padded, no `cache_control`) | $0.01647 | 8.0× |
| Anthropic `claude-haiku-4-5` (padded, `cache_control` on) | $0.01315 | 6.4× |

Explicit caching narrowed the gap only 20%. OpenAI's auto-caching made the padded prompt **cheaper** than the unpadded baseline ($0.00205 vs $0.0027 in notebook 05). Opposite effect of the same intervention across providers.

### Notebook 05 — Agent-loop benchmark (deterministic, ages well)

18 parameterized tasks over a fixed 12-product mock catalog. 4 production-shaped tools, ground truth computed from the catalog, 20% error injection on non-finalize tool calls to test recovery. Common orchestration loop; native tool-use primitives on each provider.

| | OpenAI `gpt-5.4-mini` | Anthropic `claude-haiku-4-5` |
|---|---|---|
| Success rate | 18/18 (100%) | 18/18 (100%) |
| Mean turns to completion | 3.78 | **3.44** |
| Mean cost per successful task | **$0.0027** | $0.0078 |
| Errors encountered | 16 | 11 |
| Recovery rate | 0.75 | 0.73 |

### Notebook 04 — Streaming TTFT vs total latency *(snapshot, April 2026)*

Same 20 reviews, new task: 2-sentence summary + sentiment label (generative, not tool-use). Streamed through both providers. Warm-up call per provider before the timed loop.

| | OpenAI `gpt-5.4-mini` | Anthropic `claude-haiku-4-5` |
|---|---|---|
| TTFT p50 | **620 ms** | 725 ms |
| TTFT p95 | 1531 ms | **1023 ms** |
| Total latency p50 | **1038 ms** | 1247 ms |
| Total latency p95 | 2100 ms | **1524 ms** |
| Mean tokens/sec (steady-state) | **127.0** | 100.9 |
| Cost / 1k calls | **$0.26** | $0.34 |

### Notebook 03 — Anthropic prompt caching (follow-up to the tool-schema cost gap)

Padded the system prompt (rubric + 14 few-shot examples, 4741 tokens) to clear Haiku 4.5's 4096-token cache minimum, then compared `cache_control` on vs off.

| | Baseline (no cache) | Cached |
|---|---|---|
| Mean input tokens (uncached) | 4872 | 454 |
| Mean cache-read tokens | 0 | **4197** |
| Cost per 1k calls | $5.30 | **$1.58** (−70%) |
| Latency p50 | 1089 ms | 1021 ms |
| Latency p95 | **5525 ms** | **1930 ms** (−65%) |
| Break-even vs baseline | — | call 2 |

### Observations

- **Anthropic counts tool-schema tokens as input;** OpenAI reports a leaner input in tool-use mode. Across notebook 01 runs Anthropic billed ~3.4× the input tokens in function calling and ~2× in structured output. That explains most of the cost delta — it is not raw unit price.
- **Anthropic wins median latency on notebook 01 & 02**, but tails are uneven: on function calling Anthropic's p95 was ~12% *worse* than OpenAI's despite a faster p50. For user-facing flows where p95 is the budget that matters, this inverts the "Anthropic is faster" headline.
- **Structured output is materially cheaper than tool-use for both providers.** OpenAI `json_schema` strict → 25% cheaper than tools. Anthropic prefill → 54% cheaper than tool-use, because the tool schema stops inflating the input.
- **Prompt caching closes the cost gap and cuts p95 latency.** On a 4741-token static prefix, caching drops per-call cost to 30% of baseline and p95 latency to 35% of baseline. Break-even is call 2 — the 1.25× write premium pays back in one read.
- **Agent-loop performance is where this repo's durable findings live.** At this task complexity, both current-gen small models close simple agent loops with 100% reliability and roughly similar recovery rates (~73–75%). Anthropic completes tasks in ~9% fewer turns but costs ~2.9× more per successful task — the tool-schema token tax from Finding 1 compounds across multi-turn loops, which is why **cost-per-successful-task** is the unit-economics metric that actually matters for agent deployments, not cost-per-call. Numbers age well because the tools and catalog are deterministic.
- **Padding a prompt to enable caching is NET-NEGATIVE on Anthropic in this workload.** Adding rubric + few-shots to clear Haiku 4.5's 4096-token cache minimum increased cost per task from $0.0078 (notebook 05) to $0.01315 (notebook 06 cached). OpenAI's automatic prefix caching makes the same padded prompt **cheaper** than unpadded ($0.00205 vs $0.0027). Same intervention, opposite effect per provider. "Cache everything" is not a universal heuristic — it pays only when the prompt is naturally large and the cache fires reliably.
- **Streaming shape: OpenAI wins the median, Anthropic wins the tail.** *(snapshot — these numbers reflect deployment conditions on the run date.)* OpenAI delivers first-token faster at p50 (−15%) and generates tokens ~26% faster once started. But OpenAI's TTFT p95 is 50% worse than Anthropic's (1531 ms vs 1023 ms). If you design for p50 UX, pick OpenAI; if you design against a p95 SLA, Anthropic is more consistent.
- **OpenAI `max_tokens` is deprecated on gpt-5.4-mini;** the new parameter is `max_completion_tokens`. Silent breakage if migrating existing code from gpt-4o.
- **But caching has a silent-failure mode:** Haiku 4.5 requires ≥4096 tokens in the cached prefix. First run of notebook 03 landed at 3874 tokens; the API returned `cache_creation=0` and `cache_read=0` on every call with no error. Only production-shaped prompts (rubrics, few-shots, guidelines) reliably clear the threshold — a bare tool schema (~250 tokens) never will.
- **100% validity at N=20** across notebooks 01 and 02 is a small-sample ceiling, not proof. Both providers' strict/constrained paths parsed cleanly here; a real eval would need N≥100 with trickier schemas (nested, optional, enum edge cases) to surface failure modes.
- **19/20 sentiment agreement between providers.** The one disagreement was a positive review with one negative feature mention — OpenAI classified it as `mixed`, Anthropic as `positive`. Both defensible; it exposes that "mixed" is an ambiguous label more than a model disagreement.

A longer writeup with the narrative is in [`FINDINGS.md`](./FINDINGS.md).

## License

MIT — see [`LICENSE`](./LICENSE).

## Author

Dominic Lee — [domlee.dev](https://domlee.dev)
