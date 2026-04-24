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
# # 06 — Agent loops with prompt caching
#
# Notebook 05 found that Anthropic completes the agent task in 9% fewer turns
# than OpenAI but costs **2.9× more per successful task** — the tool-schema
# token tax from Finding 1 compounds across turns when every turn re-sends
# tools + system + accumulating messages.
#
# Notebook 03 found that Anthropic prompt caching drops per-call cost by 70%
# on static prefixes ≥4096 tokens.
#
# This notebook closes the obvious follow-up: **does caching close the 2.9×
# agent-loop cost gap?**
#
# Setup:
#
# - Same 18 tasks, same catalog, same 20% error injection, same harness as
#   notebook 05.
# - **Padded system prompt** — tool-use guidelines, strategy guidance, error
#   handling, 3 worked examples of good agent behavior. Sized to clear Haiku
#   4.5's 4096-token cache minimum (otherwise caching silently fails — the
#   gotcha from Finding 5).
# - **Three runs** so the comparison is apples-to-apples:
#   1. OpenAI gpt-5.4-mini with padded prompt — baseline. Observes OpenAI's
#      automatic prefix caching, which fires without any user action.
#   2. Anthropic claude-haiku-4-5 with padded prompt, **no** `cache_control` —
#      baseline for the padded shape (isolates the caching effect, not the
#      prompt-size effect).
#   3. Anthropic claude-haiku-4-5 with padded prompt, `cache_control` on the
#      last system block (tools + system cache as one prefix).

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
    neutral_tools,
    as_anthropic_tools,
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


# %% [markdown]
# ## Padded system prompt
#
# Realistic production agent-system-prompt shape: role, tool contracts, strategy
# guidance, error handling, worked trajectories. Sized to exceed Haiku 4.5's
# 4096-token cache minimum (verified below via `count_tokens()`).

# %%
SYSTEM_PADDED = """You are a product recommendation assistant operating inside a small
research harness. Your job is to recommend one product that best matches the user's
stated budget and priority dimension, using the available tools to discover candidates
and evaluate them. Always end by calling the `finalize` tool exactly once with your
chosen product_id and a one-sentence reason.

# Tool contracts

You have four tools available. Their contracts are strict — any input that violates
the declared schema will be rejected by the runtime.

## search_products(category, max_price)

Use this tool FIRST on every task. It returns the set of products in the named
category that cost at or below `max_price` USD. The result is a list of
`{id, name, price}` records. Do not recommend a product_id that was not returned
by this tool. The category argument must be one of the enumerated values
(`headphones`, `laptop`, `vacuum`) — anything else will error.

## get_reviews(product_id)

Use this tool to fetch per-dimension review scores for a specific product, after
you have the candidate list from search_products. The result is a mapping of
dimension names (e.g., `battery_life`, `noise_cancellation`, `comfort`) to scores
on a 1-10 scale where higher is better. Scores are aggregated review sentiment;
treat them as the definitive measure for comparison on that dimension.

## compare(product_ids, dimension)

Use this tool when you want a ranked view of two or more products on a single
dimension. It returns the products sorted by score on that dimension, highest
first. This is faster than calling get_reviews individually when you already
know which dimension matters for the task.

## finalize(product_id, reason)

Call this EXACTLY ONCE, at the very end, to submit your recommendation. Once you
call finalize the task ends. Do not call finalize before you have seen the
candidate set from search_products. The reason field should be a single sentence
justifying why the chosen product best matches the user's stated priority within
their budget.

# Recommended workflow

For almost every task the right shape is:

1. Call `search_products(category, max_price)` with the user's stated category
   and budget to discover candidates.
2. If the candidate list is empty, report this and finalize with the closest
   available option or state that no product meets the criteria.
3. If there are multiple candidates, compare them on the user's stated priority
   dimension — either by calling `get_reviews` on each, or by calling `compare`
   once with the full list of candidate_ids and the priority as the dimension.
4. Select the product that scores highest on the priority dimension within the
   budget.
5. Call `finalize(product_id, reason)` with the choice and a one-sentence
   justification that references the dimension and score.

Do not skip steps. Do not guess. Do not recommend products outside the category
or above the budget. The evaluation harness checks your finalize call against a
deterministic ground-truth function of the catalog.

# Error handling

Tools occasionally return `{"error": "Service temporarily unavailable. Please retry."}`
due to simulated transient failures. When you see this error:

- Retry the same tool call with the same or slightly adjusted arguments. Errors
  are transient — the same call on the next turn often succeeds.
- Do not assume the error means the product does not exist or is not available.
  Assume it means the network call failed.
- Do not finalize based on an error response. You need successful tool output
  to justify the finalize call.
- After three consecutive errors on the same tool, adjust your approach — try a
  different tool or different arguments, but do not give up on the task.

Schema errors (e.g., passing a category not in the enum, or a non-numeric
max_price) are NOT transient. If you see a schema-related error message, adjust
the arguments to match the declared schema rather than retrying with the same
input.

# Worked examples

The following trajectories illustrate correct agent behavior. Follow these
patterns — they represent the expected shape of a successful task.

## Example 1 — simple happy path

User: "Recommend one vacuum under $200 that maximizes suction."

Turn 1 — call search_products(category="vacuum", max_price=200)
Response: {results: [{id: "vc_3", name: "BudgetVac", price: 89}]}

Turn 2 — only one candidate. Fetch its scores to confirm.
Call get_reviews(product_id="vc_3")
Response: {scores: {suction: 4, pet_hair: 3, noise_level: 6}}

Turn 3 — this is the only in-budget option. Finalize.
Call finalize(product_id="vc_3", reason="Only in-budget option under $200; suction score 4.")

## Example 2 — multiple candidates, compare on priority

User: "Recommend one laptop under $1500 that maximizes performance."

Turn 1 — call search_products(category="laptop", max_price=1500)
Response: {results: [
  {id: "lp_1", name: "ThinBook Air", price: 1200},
  {id: "lp_3", name: "BasicBook", price: 650},
  {id: "lp_4", name: "ProWorker", price: 1500},
]}

Turn 2 — three candidates, priority is performance. Use compare.
Call compare(product_ids=["lp_1", "lp_3", "lp_4"], dimension="performance")
Response: {dimension: "performance", ranking: [
  {product_id: "lp_4", score: 8},
  {product_id: "lp_1", score: 6},
  {product_id: "lp_3", score: 4},
]}

Turn 3 — lp_4 wins on performance and is in budget. Finalize.
Call finalize(product_id="lp_4", reason="ProWorker scores 8/10 on performance, the highest among the three in-budget laptop options.")

## Example 3 — error recovery

User: "Recommend one headphones under $250 that maximizes noise cancellation."

Turn 1 — call search_products(category="headphones", max_price=250)
Response: {"error": "Service temporarily unavailable. Please retry."}

Turn 2 — transient error. Retry the same call.
Call search_products(category="headphones", max_price=250)
Response: {results: [
  {id: "hp_1", name: "SoundPeaks Pro", price: 149},
  {id: "hp_2", name: "BudgetBuds", price: 49},
  {id: "hp_4", name: "MidrangeCans", price: 199},
]}

Turn 3 — three candidates, priority is noise cancellation.
Call compare(product_ids=["hp_1", "hp_2", "hp_4"], dimension="noise_cancellation")
Response: {dimension: "noise_cancellation", ranking: [
  {product_id: "hp_1", score: 9},
  {product_id: "hp_4", score: 7},
  {product_id: "hp_2", score: 3},
]}

Turn 4 — hp_1 wins decisively. Finalize.
Call finalize(product_id="hp_1", reason="SoundPeaks Pro scores 9/10 on noise cancellation and is well within the $250 budget.")

# Common pitfalls to avoid

Do not finalize before calling search_products. You cannot recommend a product
you have not seen returned by search_products. The evaluation harness treats
finalize-without-search as a distinct failure mode called `premature_finalize`.

Do not invent product_ids. The catalog has a fixed set of ids (format: hp_N,
lp_N, vc_N). Ids you did not receive from a tool call do not exist.

Do not call tools with arguments that do not appear in the schema. The schema
is strictly validated. Stick to the declared field names and types.

Do not call finalize more than once. The first call ends the task; any
subsequent calls will be ignored or will error.

Do not exceed the max_price budget. The user's budget is a hard constraint.
If all candidates are out of budget, state this in the finalize reason and
choose the closest in-budget option, or report no match.

Do not recommend based on dimensions the user did not ask about. The user
names a single priority dimension. Optimize for that dimension specifically,
not for overall quality, not for price, not for your own preferred heuristic.

Do not skip the compare step when there are three or more candidates. With
two candidates a direct get_reviews pair is fine, but with three or more
the compare tool is cleaner and cheaper than three separate get_reviews
calls.

Do not assume symmetry between dimensions. A product that scores high on
one dimension may score low on another. Always check the specific dimension
the user cares about.

Do not treat partial information as complete. If search_products returns
an error, you do not know the candidate set; do not proceed to get_reviews
or compare until you have a successful search result.

Do not skip the reason field in finalize. An empty or generic reason
("this is the best") is worse than no reason at all. Reference the dimension
and the specific score you used to decide.

## Example 4 — cascading errors and recovery

User: "Recommend one vacuum under $400 that maximizes pet hair pickup."

Turn 1 — call search_products(category="vacuum", max_price=400)
Response: {"error": "Service temporarily unavailable. Please retry."}

Turn 2 — transient error. Retry the same call.
Call search_products(category="vacuum", max_price=400)
Response: {"error": "Service temporarily unavailable. Please retry."}

Turn 3 — second consecutive error. Still retry — errors are transient and
do not mean the category is unavailable.
Call search_products(category="vacuum", max_price=400)
Response: {results: [
  {id: "vc_1", name: "SuckMaster 3000", price: 350},
  {id: "vc_2", name: "QuietClean", price: 250},
  {id: "vc_3", name: "BudgetVac", price: 89},
]}

Turn 4 — three candidates. Priority is pet_hair. Use compare.
Call compare(product_ids=["vc_1", "vc_2", "vc_3"], dimension="pet_hair")
Response: {"error": "Service temporarily unavailable. Please retry."}

Turn 5 — error on compare too. Retry.
Call compare(product_ids=["vc_1", "vc_2", "vc_3"], dimension="pet_hair")
Response: {dimension: "pet_hair", ranking: [
  {product_id: "vc_1", score: 10},
  {product_id: "vc_2", score: 6},
  {product_id: "vc_3", score: 3},
]}

Turn 6 — vc_1 wins decisively on pet_hair. Finalize.
Call finalize(product_id="vc_1", reason="SuckMaster 3000 scores 10/10 on pet hair pickup, the highest among the three in-budget vacuum options.")

Note the pattern: errors do not stop the agent. Each retry is worth trying at
least twice before concluding a tool is genuinely broken. The agent stayed on
the task and completed it within the turn budget despite three injected errors.

## Example 5 — tight budget with single candidate

User: "Recommend one laptop under $700 that maximizes portability."

Turn 1 — call search_products(category="laptop", max_price=700)
Response: {results: [{id: "lp_3", name: "BasicBook", price: 650}]}

Turn 2 — only one in-budget candidate. Confirm the priority score.
Call get_reviews(product_id="lp_3")
Response: {scores: {battery_life: 6, performance: 4, portability: 7}}

Turn 3 — lp_3 is the only option and it scores 7/10 on portability. Finalize.
Call finalize(product_id="lp_3", reason="BasicBook is the only laptop under $700; scores 7/10 on portability.")

Note: when there is only one candidate, compare is unnecessary. A single
get_reviews call is sufficient to confirm the priority score for the reason
field. Do not call compare with one product_id — it requires minimum two.

# Additional pitfalls

Do not call compare with a single product id. The schema requires at least two
ids. If only one candidate came back from search_products, use get_reviews
instead to fetch its scores.

Do not confuse the priority dimension with a score value. "Maximize battery
life" means rank candidates by their `battery_life` score, not find products
with battery_life equal to some specific number. Always pass the priority as
the `dimension` argument, never as a filter.

Do not batch multiple finalize calls "just in case". Finalize is terminal.
The system ignores subsequent calls, and emitting them wastes tokens and
suggests confusion about the workflow.

Do not mix categories. If the user asked for headphones, do not search
laptops even if you think it would be a helpful comparison. Stay strictly
within the requested category.

Do not add explanatory prose between tool calls beyond what is strictly
necessary. The harness scores based on tool calls and the finalize result,
not on narration. Concise reasoning is preferred over verbose commentary.

Do not assume the user's budget is negotiable. If all candidates exceed the
budget, the correct finalize call is to report "no product matches" rather
than recommending an over-budget option. The budget is a hard constraint.

Do not treat the priority dimension as a tiebreaker. It is the PRIMARY
ranking criterion. Budget is the hard filter; priority is the sort order;
everything else is secondary.
"""

# Count tokens on the static prefix (tools + system) to verify we clear
# Haiku 4.5's 4096-token cache minimum.
anthropic_tools = as_anthropic_tools(neutral_tools())
tc = anthropic_client.messages.count_tokens(
    model=ANTHROPIC_MODEL,
    system=SYSTEM_PADDED,
    messages=[{"role": "user", "content": "x"}],
    tools=anthropic_tools,
)
print(f"Static prefix (tools + system): {tc.input_tokens} tokens (need >=4096 to cache on Haiku 4.5)")


# %% [markdown]
# ## Run 1 — OpenAI baseline (padded prompt, automatic prefix caching)
#
# OpenAI's gpt-5 series has automatic prefix caching always on. We capture
# `prompt_tokens_details.cached_tokens` per turn to quantify it.

# %%
openai_outcomes = []
for task in tasks:
    oa = run_openai_task(
        openai_client, OPENAI_MODEL, task,
        max_turns=MAX_TURNS, system=SYSTEM_PADDED,
    )
    openai_outcomes.append(oa)
    print(
        f"task {task['index']:>2}  openai={oa.classification:<20}"
        f" ({oa.turns}t, ${oa.cost_usd:.4f}, cached={oa.cache_read_input_tokens})"
    )


# %% [markdown]
# ## Run 2 — Anthropic padded, NO cache_control (baseline)

# %%
anthropic_nocache = []
for task in tasks:
    an = run_anthropic_task(
        anthropic_client, ANTHROPIC_MODEL, task,
        max_turns=MAX_TURNS, system=SYSTEM_PADDED, use_cache=False,
    )
    anthropic_nocache.append(an)
    print(
        f"task {task['index']:>2}  anthropic_nocache={an.classification:<20}"
        f" ({an.turns}t, ${an.cost_usd:.4f})"
    )


# %% [markdown]
# ## Run 3 — Anthropic padded, WITH cache_control

# %%
anthropic_cached = []
for task in tasks:
    an = run_anthropic_task(
        anthropic_client, ANTHROPIC_MODEL, task,
        max_turns=MAX_TURNS, system=SYSTEM_PADDED, use_cache=True,
    )
    anthropic_cached.append(an)
    print(
        f"task {task['index']:>2}  anthropic_cached={an.classification:<20}"
        f" ({an.turns}t, ${an.cost_usd:.4f},"
        f" creation={an.cache_creation_input_tokens}, read={an.cache_read_input_tokens})"
    )


# %% [markdown]
# ## Aggregate summary

# %%
def _row(records):
    successes = [r for r in records if r.classification == "success"]
    return {
        "n": len(records),
        "success": len(successes),
        "mean_turns": round(sum(r.turns for r in records) / len(records), 2),
        "mean_input_tokens": round(sum(r.input_tokens for r in records) / len(records), 1),
        "mean_cache_creation": round(sum(r.cache_creation_input_tokens for r in records) / len(records), 1),
        "mean_cache_read": round(sum(r.cache_read_input_tokens for r in records) / len(records), 1),
        "mean_cost_per_task_usd": round(sum(r.cost_usd for r in records) / len(records), 5),
        "mean_cost_per_success_usd": round(
            sum(r.cost_usd for r in successes) / len(successes) if successes else float("nan"), 5
        ),
    }

summary = pd.DataFrame([
    {"run": "openai_padded",       "model": OPENAI_MODEL,    **_row(openai_outcomes)},
    {"run": "anthropic_padded",    "model": ANTHROPIC_MODEL, **_row(anthropic_nocache)},
    {"run": "anthropic_cached",    "model": ANTHROPIC_MODEL, **_row(anthropic_cached)},
]).set_index("run")
summary


# %% [markdown]
# ## Per-turn cache breakdown for the cached Anthropic run
#
# Expected pattern: first call of each task has cache_creation (1.25x write),
# subsequent calls within the task have cache_read (0.1x). If cache_read is
# zero across all calls, caching is silently not firing.

# %%
rows = []
for r in anthropic_cached:
    rows.append({
        "task": r.task_index,
        "turns": r.turns,
        "total_input_uncached": r.input_tokens,
        "total_cache_creation": r.cache_creation_input_tokens,
        "total_cache_read": r.cache_read_input_tokens,
        "total_output": r.output_tokens,
        "cost_usd": round(r.cost_usd, 5),
    })
per_task_cached = pd.DataFrame(rows)
per_task_cached


# %% [markdown]
# ## Cost ratio — does caching close the Finding 7 gap?

# %%
oa = sum(r.cost_usd for r in openai_outcomes) / len(openai_outcomes)
an_nocache = sum(r.cost_usd for r in anthropic_nocache) / len(anthropic_nocache)
an_cached = sum(r.cost_usd for r in anthropic_cached) / len(anthropic_cached)

comparison = pd.DataFrame([
    {"ratio": "Anthropic_padded / OpenAI_padded", "value": round(an_nocache / oa, 2)},
    {"ratio": "Anthropic_cached / OpenAI_padded", "value": round(an_cached / oa, 2)},
    {"ratio": "Anthropic_cached / Anthropic_padded", "value": round(an_cached / an_nocache, 2)},
]).set_index("ratio")
comparison
