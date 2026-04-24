"""
Agent-loop benchmark harness.

A deterministic micro-environment for measuring multi-turn agent behavior:

- A small fixed product catalog with per-dimension scores (no external data, no
  network variance — the "ground truth" is a pure function of the catalog).
- Four mock tools (`search_products`, `get_reviews`, `compare`, `finalize`) with
  production-realistic schemas and strict input validation on both providers.
- Deterministic error injection: a seeded RNG flips ~20% of non-finalize tool
  calls into `{"error": "..."}` responses, letting us measure recovery behavior.
- Per-provider agent-loop runners that share the same orchestration shape — only
  the model call and the tool_result threading differ, which isolates model
  behavior from harness bias.
- An outcome classifier that tags each run as one of: `success`, `wrong_answer`,
  `loop`, `hallucinated_tool`, `malformed_input`, `premature_finalize`,
  `never_called_tool`, `api_error`.

The numbers this harness produces age well: the tools are deterministic, the
catalog is fixed, and every metric (success rate, turns, cost per successful
task) is a capability property of the model rather than a snapshot of server
weather.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from src.bench import PRICING

# ---------------- Fixed catalog ----------------
#
# Each product has per-dimension scores on a 1-10 scale (higher = better). Ground
# truth for a task is the in-budget product with the highest score on the
# requested priority dimension — a deterministic function of this table.

CATALOG: list[dict[str, Any]] = [
    # Headphones
    {"id": "hp_1", "name": "SoundPeaks Pro",  "category": "headphones", "price": 149,
     "scores": {"noise_cancellation": 9, "battery_life": 7, "comfort": 8}},
    {"id": "hp_2", "name": "BudgetBuds",      "category": "headphones", "price": 49,
     "scores": {"noise_cancellation": 3, "battery_life": 6, "comfort": 5}},
    {"id": "hp_3", "name": "ElitePhones",     "category": "headphones", "price": 399,
     "scores": {"noise_cancellation": 10, "battery_life": 9, "comfort": 9}},
    {"id": "hp_4", "name": "MidrangeCans",    "category": "headphones", "price": 199,
     "scores": {"noise_cancellation": 7, "battery_life": 8, "comfort": 7}},
    # Laptops
    {"id": "lp_1", "name": "ThinBook Air",    "category": "laptop",     "price": 1200,
     "scores": {"battery_life": 9, "performance": 6, "portability": 10}},
    {"id": "lp_2", "name": "GamerBeast",      "category": "laptop",     "price": 1800,
     "scores": {"battery_life": 4, "performance": 10, "portability": 3}},
    {"id": "lp_3", "name": "BasicBook",       "category": "laptop",     "price": 650,
     "scores": {"battery_life": 6, "performance": 4, "portability": 7}},
    {"id": "lp_4", "name": "ProWorker",       "category": "laptop",     "price": 1500,
     "scores": {"battery_life": 7, "performance": 8, "portability": 6}},
    # Vacuums
    {"id": "vc_1", "name": "SuckMaster 3000", "category": "vacuum",     "price": 350,
     "scores": {"suction": 9, "pet_hair": 10, "noise_level": 5}},
    {"id": "vc_2", "name": "QuietClean",      "category": "vacuum",     "price": 250,
     "scores": {"suction": 7, "pet_hair": 6, "noise_level": 9}},
    {"id": "vc_3", "name": "BudgetVac",       "category": "vacuum",     "price": 89,
     "scores": {"suction": 4, "pet_hair": 3, "noise_level": 6}},
    {"id": "vc_4", "name": "DeluxeDuster",    "category": "vacuum",     "price": 599,
     "scores": {"suction": 10, "pet_hair": 9, "noise_level": 7}},
]

CATEGORIES = ["headphones", "laptop", "vacuum"]
PRIORITIES = {
    "headphones": ["noise_cancellation", "battery_life", "comfort"],
    "laptop":     ["battery_life", "performance", "portability"],
    "vacuum":     ["suction", "pet_hair", "noise_level"],
}
BUDGETS = {
    "headphones": [150, 250, 450],
    "laptop":     [800, 1500, 2000],
    "vacuum":     [150, 400, 700],
}


# ---------------- Task generation + ground truth ----------------

def generate_tasks() -> list[dict[str, Any]]:
    """18 parameterized tasks: 3 categories × 3 budgets × 2 priorities per category."""
    tasks: list[dict[str, Any]] = []
    for category in CATEGORIES:
        priorities = PRIORITIES[category][:2]  # 2 per category → 18 total
        for budget in BUDGETS[category]:
            for priority in priorities:
                tasks.append({
                    "index": len(tasks),
                    "category": category,
                    "budget": budget,
                    "priority": priority,
                    "prompt": (
                        f"Recommend one {category} under ${budget} that maximizes "
                        f"{priority.replace('_', ' ')}. Use the tools to research, "
                        f"then call finalize() with your choice."
                    ),
                })
    return tasks


def ground_truth(task: dict[str, Any]) -> str | None:
    """The in-budget product with the highest score on the task's priority."""
    candidates = [
        p for p in CATALOG
        if p["category"] == task["category"] and p["price"] <= task["budget"]
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p["scores"].get(task["priority"], 0))["id"]


# ---------------- Tool schemas ----------------

SYSTEM_PROMPT = (
    "You are a product recommendation assistant. The user will describe a product "
    "they're looking for with a budget and a priority dimension. Use the available "
    "tools to research candidates, then call `finalize` exactly once with your "
    "chosen product_id and a one-sentence reason.\n\n"
    "Guidelines:\n"
    "- Always start with `search_products` to discover in-budget candidates.\n"
    "- If a tool returns an error, retry with the same or adjusted arguments — "
    "errors are transient.\n"
    "- Do not call `finalize` before you have enough information to justify the "
    "choice (at minimum, you must have seen the candidate set).\n"
    "- Do not invent product ids. Only recommend products returned by "
    "`search_products`."
)


def neutral_tools() -> list[dict[str, Any]]:
    """Provider-neutral tool definitions. Converted below per provider."""
    return [
        {
            "name": "search_products",
            "description": (
                "Search the product catalog by category and max price. Returns a "
                "list of matching products with id, name, and price. Use this FIRST "
                "to discover candidates before calling get_reviews or compare."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": CATEGORIES,
                        "description": "The product category to search.",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price in USD. Inclusive.",
                    },
                },
                "required": ["category", "max_price"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_reviews",
            "description": (
                "Fetch review scores for a specific product, broken down by dimension "
                "(e.g., battery_life, noise_cancellation). Higher is better. "
                "Use after search_products to evaluate candidates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product id returned by search_products.",
                    },
                },
                "required": ["product_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "compare",
            "description": (
                "Compare two or more products on a specific dimension. "
                "Returns the products ranked by score on that dimension."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Two or more product ids to compare.",
                    },
                    "dimension": {
                        "type": "string",
                        "description": "The scoring dimension (e.g., battery_life).",
                    },
                },
                "required": ["product_ids", "dimension"],
                "additionalProperties": False,
            },
        },
        {
            "name": "finalize",
            "description": (
                "Submit your final recommendation. Call this exactly ONCE at the "
                "end after you have enough information to choose. Do NOT call this "
                "before calling search_products first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The id of the recommended product.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "A one-sentence justification.",
                    },
                },
                "required": ["product_id", "reason"],
                "additionalProperties": False,
            },
        },
    ]


def as_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in tools
    ]


def as_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
                "strict": True,
            },
        }
        for t in tools
    ]


# ---------------- Mock tool execution ----------------

VALID_TOOL_NAMES = {"search_products", "get_reviews", "compare", "finalize"}


def mock_tool_call(
    name: str,
    tool_input: dict[str, Any],
    rng: random.Random,
    error_rate: float = 0.20,
) -> dict[str, Any]:
    """Execute a mock tool call. Returns either a success payload or {error: ...}.

    20% of non-finalize calls are deliberately flipped into errors to test recovery.
    """
    if name not in VALID_TOOL_NAMES:
        return {"error": f"Unknown tool: {name}"}

    # Error injection (not on finalize — we want finalize to be the definitive signal)
    if name != "finalize" and rng.random() < error_rate:
        return {"error": "Service temporarily unavailable. Please retry."}

    if name == "search_products":
        cat = tool_input["category"]
        max_price = tool_input["max_price"]
        matches = [
            {"id": p["id"], "name": p["name"], "price": p["price"]}
            for p in CATALOG
            if p["category"] == cat and p["price"] <= max_price
        ]
        return {"category": cat, "max_price": max_price, "results": matches}

    if name == "get_reviews":
        pid = tool_input["product_id"]
        product = next((p for p in CATALOG if p["id"] == pid), None)
        if product is None:
            return {"error": f"Product {pid} not found."}
        return {"product_id": pid, "scores": product["scores"]}

    if name == "compare":
        pids = tool_input["product_ids"]
        dim = tool_input["dimension"]
        rows = []
        for pid in pids:
            product = next((p for p in CATALOG if p["id"] == pid), None)
            if product is None:
                continue
            rows.append({"product_id": pid, "score": product["scores"].get(dim, 0)})
        rows.sort(key=lambda r: -r["score"])
        return {"dimension": dim, "ranking": rows}

    # finalize
    return {"submitted": tool_input.get("product_id"), "status": "accepted"}


# ---------------- Per-task outcome record ----------------

@dataclass
class TaskOutcome:
    task_index: int
    provider: str
    model: str
    classification: str = "unknown"
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors_encountered: int = 0
    errors_recovered: int = 0  # retries of the same tool after an error
    finalized_id: str | None = None
    reason: str | None = None
    called_tools: list[str] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def cost_usd(self) -> float:
        p = PRICING.get(self.model)
        if not p:
            return 0.0
        return self.input_tokens * p["input"] + self.output_tokens * p["output"]


# ---------------- Anthropic agent-loop runner ----------------

def run_anthropic_task(
    client,
    model: str,
    task: dict[str, Any],
    max_turns: int = 10,
) -> TaskOutcome:
    """Run one task via Anthropic. Returns the outcome + trajectory."""
    outcome = TaskOutcome(task_index=task["index"], provider="anthropic", model=model)
    rng = random.Random(f"task-{task['index']}-anthropic")
    tools = as_anthropic_tools(neutral_tools())
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": task["prompt"]},
    ]
    last_tool_name: str | None = None
    last_tool_was_error: bool = False

    try:
        for _ in range(max_turns):
            outcome.turns += 1
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
            )
            outcome.input_tokens += resp.usage.input_tokens
            outcome.output_tokens += resp.usage.output_tokens

            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                break  # model returned text only

            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for tu in tool_uses:
                outcome.called_tools.append(tu.name)
                outcome.trajectory.append({
                    "turn": outcome.turns,
                    "tool": tu.name,
                    "input": tu.input,
                })
                result = mock_tool_call(tu.name, tu.input, rng)
                is_error = "error" in result
                if is_error:
                    outcome.errors_encountered += 1
                elif last_tool_was_error and tu.name == last_tool_name:
                    outcome.errors_recovered += 1
                last_tool_name, last_tool_was_error = tu.name, is_error
                if tu.name == "finalize":
                    outcome.finalized_id = tu.input.get("product_id")
                    outcome.reason = tu.input.get("reason")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result),
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": tool_results})

            if outcome.finalized_id is not None:
                break
    except Exception as e:
        outcome.error = f"{type(e).__name__}: {e}"[:300]
        outcome.classification = "api_error"

    if not outcome.classification or outcome.classification == "unknown":
        outcome.classification = classify(outcome, task)
    return outcome


# ---------------- OpenAI agent-loop runner ----------------

def run_openai_task(
    client,
    model: str,
    task: dict[str, Any],
    max_turns: int = 10,
) -> TaskOutcome:
    """Run one task via OpenAI. Returns the outcome + trajectory."""
    outcome = TaskOutcome(task_index=task["index"], provider="openai", model=model)
    rng = random.Random(f"task-{task['index']}-openai")
    tools = as_openai_tools(neutral_tools())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]
    last_tool_name: str | None = None
    last_tool_was_error: bool = False

    try:
        for _ in range(max_turns):
            outcome.turns += 1
            resp = client.chat.completions.create(
                model=model,
                max_completion_tokens=1024,
                messages=messages,
                tools=tools,
            )
            outcome.input_tokens += resp.usage.prompt_tokens
            outcome.output_tokens += resp.usage.completion_tokens
            msg = resp.choices[0].message

            if not msg.tool_calls:
                break  # model returned text only

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                outcome.called_tools.append(name)
                outcome.trajectory.append({
                    "turn": outcome.turns,
                    "tool": name,
                    "input": args,
                })
                result = mock_tool_call(name, args, rng)
                is_error = "error" in result
                if is_error:
                    outcome.errors_encountered += 1
                elif last_tool_was_error and name == last_tool_name:
                    outcome.errors_recovered += 1
                last_tool_name, last_tool_was_error = name, is_error
                if name == "finalize":
                    outcome.finalized_id = args.get("product_id")
                    outcome.reason = args.get("reason")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

            if outcome.finalized_id is not None:
                break
    except Exception as e:
        outcome.error = f"{type(e).__name__}: {e}"[:300]
        outcome.classification = "api_error"

    if not outcome.classification or outcome.classification == "unknown":
        outcome.classification = classify(outcome, task)
    return outcome


# ---------------- Outcome classifier ----------------

def classify(outcome: TaskOutcome, task: dict[str, Any]) -> str:
    """Map a completed outcome to one of the failure-mode categories."""
    if outcome.error:
        return "api_error"

    # Hallucinated tool name
    if any(t not in VALID_TOOL_NAMES for t in outcome.called_tools):
        return "hallucinated_tool"

    # Never called any tool
    if not outcome.called_tools:
        return "never_called_tool"

    # Finalized without ever searching
    if outcome.finalized_id is not None and "search_products" not in outcome.called_tools:
        return "premature_finalize"

    # Didn't finalize within the turn budget
    if outcome.finalized_id is None:
        return "loop"

    truth = ground_truth(task)
    if outcome.finalized_id == truth:
        return "success"
    return "wrong_answer"
