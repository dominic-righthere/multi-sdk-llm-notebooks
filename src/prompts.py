"""
Shared prompt content across both SDKs.

Keeping prompts here (not inline in notebooks) enforces the 'same task, same prompt'
constraint for a fair comparison.
"""

SYSTEM_PROMPT = """You are a product-review analyst. Given a product review, classify its sentiment, extract the key product features mentioned, and estimate the star rating (1-5)."""

# Review text goes in the user turn.
USER_TEMPLATE = """Review:
{review_text}"""

# The tool / JSON schema shared between both notebooks.
SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral", "mixed"],
        },
        "key_features": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Product features or aspects the reviewer mentioned.",
        },
        "rating_estimate": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
    },
    "required": ["sentiment", "key_features", "rating_estimate"],
    "additionalProperties": False,
}


# Tool/function definition in the providers' native shapes.
# Keep the "business logic" schema identical; only the wrapping differs.

OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_review",
        "description": "Classify a product review and extract features.",
        "parameters": SCHEMA,
    },
}

ANTHROPIC_TOOL = {
    "name": "classify_review",
    "description": "Classify a product review and extract features.",
    "input_schema": SCHEMA,
}
