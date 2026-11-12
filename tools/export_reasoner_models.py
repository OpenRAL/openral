"""Export the curated reasoner models for openral.com's Account → Settings (ADR-0129).

The website offers ``REASONER_MODELS`` as its model drop-down and asks for the key each
one needs: ``ANTHROPIC_API_KEY`` for an Anthropic-dialect model, ``OPENAI_API_KEY`` for
an OpenAI-dialect cloud model (sent to OpenAI's own API), none for one that runs on the
robot. ``.github/workflows/reasoner-models.yml`` writes this to the website's
``server/reasoner-models.json`` whenever the list changes on master.

Usage: python tools/export_reasoner_models.py > reasoner-models.json
"""

from __future__ import annotations

import json
import sys
from typing import Any

from openral_core import reasoner_model_options


def export() -> list[dict[str, Any]]:
    """``[{id, label, needs}]`` in the registry's order.

    Example:
        >>> {m["id"]: m["needs"] for m in export()}["claude-opus-4-8"]
        'ANTHROPIC_API_KEY'
        >>> next(m for m in export() if m["id"] == "gpt-5.5")["label"]
        'GPT-5.5'
    """
    return reasoner_model_options()


if __name__ == "__main__":
    json.dump(export(), sys.stdout, indent=2)
    sys.stdout.write("\n")
