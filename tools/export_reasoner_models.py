"""Export the curated reasoner models for openral.com's Account → Settings (ADR-0129).

The website offers ``REASONER_MODELS`` as its model drop-down and asks for a key each
one runs on (``keys``, any one enough): ``ANTHROPIC_API_KEY`` for an Anthropic-dialect
model, ``OPENAI_API_KEY`` or ``OPENROUTER_API_KEY`` for an ``openai/`` model,
``OPENROUTER_API_KEY`` for any other model on OpenRouter, ``OLLAMA_API_KEY`` for a
model on Ollama Cloud, none for one that runs on the robot.
``.github/workflows/reasoner-models.yml`` writes this to the website's
``server/reasoner-models.json`` whenever the list changes on master.

Usage: python tools/export_reasoner_models.py > reasoner-models.json
"""

from __future__ import annotations

import json
import sys

from openral_core import reasoner_model_options


def export() -> list[dict[str, object]]:
    """``[{id, label, needs, keys, served}]`` in the registry's order.

    Example:
        >>> {m["id"]: m["keys"] for m in export()}["claude-opus-4-8"]
        ['ANTHROPIC_API_KEY']
        >>> next(m for m in export() if m["id"] == "gpt-5.5")["label"]
        'GPT-5.5'
        >>> {m["id"]: m["served"] for m in export()}["kimi-k3-cloud"]
        'kimi-k3'
    """
    return [option.model_dump() for option in reasoner_model_options()]


if __name__ == "__main__":
    json.dump(export(), sys.stdout, indent=2)
    sys.stdout.write("\n")
