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

### Observations

- **Anthropic counts tool-schema tokens as input;** OpenAI reports a leaner input in tool-use mode. Across these runs Anthropic billed ~3.4× the input tokens in function calling and ~2× in structured output. That explains most of the cost delta — it is not raw unit price.
- **Anthropic wins median latency on both tasks**, but tails are uneven: on function calling Anthropic's p95 was ~12% *worse* than OpenAI's despite a faster p50, so the cost of a slow call is higher. For user-facing flows where p95 is the budget that matters, this inverts the "Anthropic is faster" headline.
- **Structured output is materially cheaper than tool-use for both providers.** OpenAI `json_schema` strict → 25% cheaper than tools. Anthropic prefill → 54% cheaper than tool-use, because the tool schema stops inflating the input. If JSON is all you need, structured output is the right default.
- **100% validity at N=20** is a small-sample ceiling, not proof. Both providers' strict/constrained paths parsed cleanly here; a real eval would need N≥100 with trickier schemas (nested, optional, enum edge cases) to surface failure modes.
- **19/20 sentiment agreement between providers.** The one disagreement was a positive review with one negative feature mention — OpenAI classified it as `mixed`, Anthropic as `positive`. Both defensible; it exposes that "mixed" is an ambiguous label more than a model disagreement.

A longer writeup with the narrative is in [`FINDINGS.md`](./FINDINGS.md).

## License

MIT — see [`LICENSE`](./LICENSE).

## Author

Dominic Lee — [domlee.dev](https://domlee.dev)
