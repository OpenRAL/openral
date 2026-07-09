"""Process-boundary fakes for the openral integration test tier.

Per CLAUDE.md §1.11 the only acceptable test doubles are at process /
network boundaries and must live under ``tests/<tier>/fakes/``. The
fakes here satisfy that contract:

- :class:`fake_llm.FakeToolUseClient` — deterministic stand-in for the
  Anthropic / OpenAI-compatible LLM endpoints used by F4. Production
  code never imports it.
"""

from __future__ import annotations
