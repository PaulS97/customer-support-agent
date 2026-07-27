"""Named pipeline configurations.

The first entry in CONFIGS is used as the default when --config is omitted.
"""

CONFIGS = {
    "gpt": {
        "classification_model": "gpt-4.1-mini",
        "draft_model": "gpt-4.1-mini",
    },
    "claude": {
        "classification_model": "claude-sonnet-5",
        "draft_model": "claude-sonnet-5",
    },
}
