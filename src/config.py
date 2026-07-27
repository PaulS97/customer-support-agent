"""Named pipeline configurations.

The first entry in CONFIGS is used as the default when --config is omitted.
"""

CONFIGS = {
    "claude-sonnet": {
        "classification_model": "claude-sonnet-5",
        "draft_model": "claude-sonnet-5",
    },
    "gpt-4.1-mini": {
        "classification_model": "gpt-4.1-mini",
        "draft_model": "gpt-4.1-mini",
    },
    "gpt-5-mini": {
        "classification_model": "gpt-5-mini",
        "draft_model": "gpt-5-mini",
    },
}
