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
# # 05 — Agent-loop benchmark
#
# **The prior notebooks measure latency, cost, and token counts. Those numbers drift
# with server load, region, time of day. This notebook measures something that
# ages better: can the model close a multi-step agent loop?**
#
# Setup:
#
# - A fixed 12-product catalog with deterministic per-dimension scores.
# - 4 mock tools with production-realistic schemas (`search_products`,
#   `get_reviews`, `compare`, `finalize`). Tools are pure functions of the
#   catalog — no external variance.
# - 18 parameterized tasks ("Recommend a {category} under ${budget} that
#   maximizes {priority}"). Ground truth is a deterministic function of the
#   catalog, so "correct answer" is unambiguous.
# - **20% error injection** on non-finalize tool calls (seeded RNG, independent
#   per provider). Measures recovery behavior — when a tool returns `{error:
#   "..."}`, does the agent retry, reformulate, or give up?
# - Common orchestration harness. Model call + tool-result threading are native
#   to each provider (Anthropic `tools` with content-block results, OpenAI
#   `tools` with `tool_call_id` results), but the loop shape is identical.
#
# What this produces:
#
# - **Success rate** — how often did the agent reach the ground-truth answer?
# - **Mean turns to completion** — model→tool→model round-trips per successful task.
# - **Cost per successful task** — unit economics after factoring in retries
#   and failed runs. This is the number that actually matters for agent pricing.
# - **Recovery rate** — on tasks where errors were injected, did the agent
#   retry the failing tool?
# - **Failure-mode taxonomy** — `wrong_answer`, `loop`, `hallucinated_tool`,
#   `premature_finalize`, `never_called_tool`, `api_error`.
#
# These numbers age well because the catalog is fixed, the tools are
# deterministic, and every metric is a model-behavior property — not a
# deployment-weather snapshot.

# %%
import os

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

from src.agent_harness import (
    CATALOG,
    generate_tasks,
    ground_truth,
    run_anthropic_task,
    run_openai_task,
)

load_dotenv()
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

OPENAI_MODEL = "gpt-5.4-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"
MAX_TURNS = 10

tasks = generate_tasks()
print(f"{len(tasks)} tasks, {len(CATALOG)} products in catalog")
print("First 3 tasks:")
for t in tasks[:3]:
    print(f"  [{t['index']}] {t['prompt']}  (truth: {ground_truth(t)})")


# %% [markdown]
# ## Run the benchmark — both providers, same tasks, same error seeds

# %%
outcomes: list = []

for task in tasks:
    oa = run_openai_task(openai_client, OPENAI_MODEL, task, max_turns=MAX_TURNS)
    an = run_anthropic_task(anthropic_client, ANTHROPIC_MODEL, task, max_turns=MAX_TURNS)
    outcomes.extend([oa, an])
    print(
        f"task {task['index']:>2}  "
        f"openai={oa.classification:<20} ({oa.turns}t, ${oa.cost_usd:.4f})   "
        f"anthropic={an.classification:<20} ({an.turns}t, ${an.cost_usd:.4f})"
    )

print(f"\nCollected {len(outcomes)} outcomes")


# %% [markdown]
# ## Aggregate summary

# %%
def _summary_row(records):
    n = len(records)
    successes = [r for r in records if r.classification == "success"]
    errs = sum(r.errors_encountered for r in records)
    recovs = sum(r.errors_recovered for r in records)
    cost_success = (
        sum(r.cost_usd for r in successes) / len(successes) if successes else float("nan")
    )
    cost_total = sum(r.cost_usd for r in records) / n
    return {
        "n": n,
        "success_rate": round(len(successes) / n, 3),
        "mean_turns": round(sum(r.turns for r in records) / n, 2),
        "mean_turns_on_success": round(
            sum(r.turns for r in successes) / len(successes) if successes else float("nan"), 2
        ),
        "mean_cost_per_call_usd": round(cost_total, 5),
        "mean_cost_per_success_usd": round(cost_success, 5),
        "errors_encountered": errs,
        "errors_recovered": recovs,
        "recovery_rate": round(recovs / errs, 3) if errs else float("nan"),
    }


summary = pd.DataFrame(
    [
        {"provider": "openai", "model": OPENAI_MODEL,
         **_summary_row([r for r in outcomes if r.provider == "openai"])},
        {"provider": "anthropic", "model": ANTHROPIC_MODEL,
         **_summary_row([r for r in outcomes if r.provider == "anthropic"])},
    ]
).set_index("provider")
summary


# %% [markdown]
# ## Failure-mode breakdown

# %%
breakdown = pd.crosstab(
    index=[r.classification for r in outcomes],
    columns=[r.provider for r in outcomes],
    margins=True,
    margins_name="total",
).rename_axis(index="classification", columns="provider")
breakdown


# %% [markdown]
# ## Per-task detail

# %%
by_task = pd.DataFrame([
    {
        "task": t["index"],
        "category": t["category"],
        "budget": t["budget"],
        "priority": t["priority"],
        "truth": ground_truth(t),
        "openai_result": next(r for r in outcomes if r.task_index == t["index"] and r.provider == "openai").finalized_id,
        "openai_class": next(r for r in outcomes if r.task_index == t["index"] and r.provider == "openai").classification,
        "openai_turns": next(r for r in outcomes if r.task_index == t["index"] and r.provider == "openai").turns,
        "anthropic_result": next(r for r in outcomes if r.task_index == t["index"] and r.provider == "anthropic").finalized_id,
        "anthropic_class": next(r for r in outcomes if r.task_index == t["index"] and r.provider == "anthropic").classification,
        "anthropic_turns": next(r for r in outcomes if r.task_index == t["index"] and r.provider == "anthropic").turns,
    }
    for t in tasks
])
by_task
