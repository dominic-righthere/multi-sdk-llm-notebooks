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

Results aggregate into a pandas DataFrame via `summarise()`. Run the notebooks to populate your own numbers; pricing, model behaviour, and SDK ergonomics all move, so this repo is structured to be re-run rather than to advertise a snapshot.

Default models (swap in the notebook's `OPENAI_MODEL` / `ANTHROPIC_MODEL` constants):

- OpenAI `gpt-5.4-mini`
- Anthropic `claude-haiku-4-5`

## License

MIT — see [`LICENSE`](./LICENSE).

## Author

Dominic Lee — [domlee.dev](https://domlee.dev)
