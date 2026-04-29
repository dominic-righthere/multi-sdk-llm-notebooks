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
# # 07 — Native runners: OpenAI Agents SDK vs Anthropic tool_runner
#
# Notebooks 01–06 use a **hand-rolled** common harness — same loop on both
# sides, only the model API call and tool-result threading differ per provider.
# Good for fairness, but not the harness either company actually wants you to
# use. Both ship first-party runners that abstract the loop, retry semantics,
# and turn-state management:
#
# - **Anthropic** `client.beta.messages.tool_runner()` with the `@beta_tool`
#   decorator (Python beta).
# - **OpenAI** `openai-agents` package (separate `pip install`), `Agent` and
#   `Runner` classes with `@function_tool` decorator. v0.14.6 here.
#
# This notebook runs the same 18 agent tasks from notebook 05 through both
# native runners, then compares against the hand-rolled baseline. The
# question is not "which is faster" but **"can the canonical runner reproduce
# the hand-rolled result, and what does it cost / hide / expose?"**

# %%
import os
import random

import nest_asyncio
import pandas as pd
from anthropic import Anthropic, beta_tool
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
from openai import OpenAI

# Jupyter already has a running asyncio event loop. The OpenAI Agents SDK's
# Runner.run_sync() internally calls asyncio.run(), which raises inside a
# running loop. nest_asyncio patches asyncio to permit nested loops, which is
# the documented workaround for using Runner.run_sync in notebooks.
nest_asyncio.apply()

from src.agent_harness import (
    CATALOG,
    SYSTEM_PROMPT,
    TaskOutcome,
    TASK_RNG,
    TASK_STATE,
    classify,
    fresh_task_state,
    generate_tasks,
    ground_truth,
    run_anthropic_task,
    run_openai_task,
    tool_compare,
    tool_finalize,
    tool_get_reviews,
    tool_search_products,
)

load_dotenv()
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

OPENAI_MODEL = "gpt-5.4-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"
MAX_TURNS = 10

tasks = generate_tasks()
print(f"{len(tasks)} tasks, {len(CATALOG)} products in catalog")


# %% [markdown]
# ## OpenAI Agents SDK runner
#
# `@function_tool` infers the tool schema from the function signature + docstring.
# We pass plain Python functions in; the runner builds the loop, calls the
# tools, and handles the message threading. No manual context passing of
# message lists, tool_call_ids, or response.choices[0] inspection.

# %%
oa_search   = function_tool(tool_search_products)
oa_reviews  = function_tool(tool_get_reviews)
oa_compare  = function_tool(tool_compare)
oa_finalize = function_tool(tool_finalize)

oa_agent = Agent(
    name="ProductRecommender",
    instructions=SYSTEM_PROMPT,
    tools=[oa_search, oa_reviews, oa_compare, oa_finalize],
    model=OPENAI_MODEL,
)


def run_openai_native(task: dict) -> TaskOutcome:
    """Run one task via the OpenAI Agents SDK.

    Bridges Runner.run (async) to our sync harness by running on the existing
    event loop with nest_asyncio's loop.run_until_complete patch. Plain
    Runner.run_sync() raises inside Jupyter because it wraps asyncio.run(),
    which the kernel's running loop rejects even with nest_asyncio applied.
    """
    import asyncio
    outcome = TaskOutcome(task_index=task["index"], provider="openai", model=OPENAI_MODEL)
    rng_token = TASK_RNG.set(random.Random(f"task-{task['index']}-openai"))
    state = fresh_task_state()
    state_token = TASK_STATE.set(state)
    try:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            Runner.run(oa_agent, task["prompt"], max_turns=MAX_TURNS)
        )
        usage = result.context_wrapper.usage
        outcome.input_tokens = usage.input_tokens
        outcome.output_tokens = usage.output_tokens
        details = getattr(usage, "input_tokens_details", None)
        outcome.cache_read_input_tokens = getattr(details, "cached_tokens", 0) or 0
        # Turn count: each model API call = one turn (matches our hand-rolled loop's definition).
        outcome.turns = getattr(usage, "requests", 0) or len(getattr(usage, "request_usage_entries", []))
        outcome.finalized_id = state["finalized_id"]
        outcome.reason = state["reason"]
        outcome.called_tools = list(state["called_tools"])
        outcome.errors_encountered = state["errors_encountered"]
        outcome.errors_recovered = state["errors_recovered"]
    except Exception as e:
        outcome.error = f"{type(e).__name__}: {e}"[:300]
    finally:
        TASK_RNG.reset(rng_token)
        TASK_STATE.reset(state_token)
    outcome.classification = classify(outcome, task)
    return outcome


# %% [markdown]
# ## Anthropic tool_runner runner
#
# `@beta_tool` similarly wraps plain Python functions. `tool_runner()` returns
# an iterator of `BetaMessage` objects — each iteration is one model API call.
# Per-message `usage` includes `input_tokens`, `output_tokens`, and cache
# fields when caching is in play.

# %%
an_search   = beta_tool(tool_search_products)
an_reviews  = beta_tool(tool_get_reviews)
an_compare  = beta_tool(tool_compare)
an_finalize = beta_tool(tool_finalize)


def run_anthropic_native(task: dict) -> TaskOutcome:
    outcome = TaskOutcome(task_index=task["index"], provider="anthropic", model=ANTHROPIC_MODEL)
    rng_token = TASK_RNG.set(random.Random(f"task-{task['index']}-anthropic"))
    state = fresh_task_state()
    state_token = TASK_STATE.set(state)
    try:
        runner = anthropic_client.beta.messages.tool_runner(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": task["prompt"]}],
            tools=[an_search, an_reviews, an_compare, an_finalize],
            max_iterations=MAX_TURNS,
        )
        for message in runner:
            outcome.turns += 1
            usage = message.usage
            outcome.input_tokens += usage.input_tokens
            outcome.output_tokens += usage.output_tokens
            outcome.cache_creation_input_tokens += (
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            )
            outcome.cache_read_input_tokens += (
                getattr(usage, "cache_read_input_tokens", 0) or 0
            )
        outcome.finalized_id = state["finalized_id"]
        outcome.reason = state["reason"]
        outcome.called_tools = list(state["called_tools"])
        outcome.errors_encountered = state["errors_encountered"]
        outcome.errors_recovered = state["errors_recovered"]
    except Exception as e:
        outcome.error = f"{type(e).__name__}: {e}"[:300]
    finally:
        TASK_RNG.reset(rng_token)
        TASK_STATE.reset(state_token)
    outcome.classification = classify(outcome, task)
    return outcome


# %% [markdown]
# ## Run the three implementations on the same 18 tasks

# %%
hand_rolled_oa, hand_rolled_an, native_oa, native_an = [], [], [], []

for task in tasks:
    h_oa = run_openai_task(openai_client, OPENAI_MODEL, task, max_turns=MAX_TURNS)
    h_an = run_anthropic_task(anthropic_client, ANTHROPIC_MODEL, task, max_turns=MAX_TURNS)
    n_oa = run_openai_native(task)
    n_an = run_anthropic_native(task)
    hand_rolled_oa.append(h_oa)
    hand_rolled_an.append(h_an)
    native_oa.append(n_oa)
    native_an.append(n_an)
    print(
        f"task {task['index']:>2} | "
        f"hand_oa={h_oa.classification:<10} ({h_oa.turns}t,${h_oa.cost_usd:.4f}) | "
        f"hand_an={h_an.classification:<10} ({h_an.turns}t,${h_an.cost_usd:.4f}) | "
        f"native_oa={n_oa.classification:<10} ({n_oa.turns}t,${n_oa.cost_usd:.4f}) | "
        f"native_an={n_an.classification:<10} ({n_an.turns}t,${n_an.cost_usd:.4f})"
    )
    for label, rec in (("native_oa", n_oa), ("native_an", n_an)):
        if rec.error:
            print(f"  -> {label} error: {rec.error}")


# %% [markdown]
# ## Aggregate summary — does the native runner reproduce the hand-rolled result?

# %%
def _row(label, model, records):
    successes = [r for r in records if r.classification == "success"]
    return {
        "implementation": label,
        "model": model,
        "n": len(records),
        "success": len(successes),
        "mean_turns": round(sum(r.turns for r in records) / len(records), 2),
        "mean_cost_per_task_usd": round(sum(r.cost_usd for r in records) / len(records), 5),
        "mean_cost_per_success_usd": round(
            sum(r.cost_usd for r in successes) / len(successes) if successes else float("nan"), 5
        ),
    }

summary = pd.DataFrame([
    _row("hand_rolled", OPENAI_MODEL,    hand_rolled_oa),
    _row("native_sdk",  OPENAI_MODEL,    native_oa),
    _row("hand_rolled", ANTHROPIC_MODEL, hand_rolled_an),
    _row("native_sdk",  ANTHROPIC_MODEL, native_an),
]).set_index(["implementation", "model"])
summary


# %% [markdown]
# ## Per-task agreement — did the native runner pick the same product?

# %%
agreement = pd.DataFrame([
    {
        "task": t["index"],
        "truth": ground_truth(t),
        "openai_hand": hand_rolled_oa[i].finalized_id,
        "openai_native": native_oa[i].finalized_id,
        "openai_match": hand_rolled_oa[i].finalized_id == native_oa[i].finalized_id,
        "anthropic_hand": hand_rolled_an[i].finalized_id,
        "anthropic_native": native_an[i].finalized_id,
        "anthropic_match": hand_rolled_an[i].finalized_id == native_an[i].finalized_id,
    }
    for i, t in enumerate(tasks)
])
agreement


# %% [markdown]
# ## Code-density delta
#
# Lines of code to ship a working agent on each path. The hand-rolled count
# excludes the shared mock-tool layer (which both versions reuse) — only the
# loop-management code differs.

# %%
import inspect
from src.agent_harness import run_anthropic_task as harness_an, run_openai_task as harness_oa

def _loc(fn):
    src = inspect.getsource(fn)
    # Count non-blank, non-comment-only, non-docstring lines as a fairness heuristic.
    lines = [
        l for l in src.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    return len(lines)

loc = pd.DataFrame([
    {"implementation": "hand_rolled — anthropic loop", "lines": _loc(harness_an)},
    {"implementation": "hand_rolled — openai loop",    "lines": _loc(harness_oa)},
    {"implementation": "native — anthropic_runner",    "lines": _loc(run_anthropic_native)},
    {"implementation": "native — openai_runner",       "lines": _loc(run_openai_native)},
]).set_index("implementation")
loc


# %% [markdown]
# ## What's exposed vs what's hidden
#
# Quick inspection of which fields each runner surfaces. Useful to know when
# planning observability.

# %%
def _what_we_got(records, label):
    sample = records[0]
    return {
        "label": label,
        "input_tokens": "yes" if sample.input_tokens > 0 else "no",
        "output_tokens": "yes" if sample.output_tokens > 0 else "no",
        "cache_creation": "yes" if sample.cache_creation_input_tokens > 0 else "no/zero",
        "cache_read": "yes" if sample.cache_read_input_tokens > 0 else "no/zero",
        "turns": "yes" if sample.turns > 0 else "no",
        "called_tools": "yes" if sample.called_tools else "no",
        "errors_encountered": sample.errors_encountered,
        "errors_recovered": sample.errors_recovered,
    }

exposure = pd.DataFrame([
    _what_we_got(hand_rolled_oa, "hand_rolled openai"),
    _what_we_got(native_oa,      "native openai"),
    _what_we_got(hand_rolled_an, "hand_rolled anthropic"),
    _what_we_got(native_an,      "native anthropic"),
]).set_index("label")
exposure
