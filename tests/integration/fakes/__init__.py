"""Process-boundary fakes for the openral integration test tier.

Per CLAUDE.md §1.11 the only acceptable test doubles are at process /
network boundaries and must live under ``tests/<tier>/fakes/``. The
fakes here satisfy that contract:

- ``fake_llm.FakeToolUseClient`` — deterministic stand-in for the
  Anthropic / OpenAI-compatible LLM endpoints used by F4. Production
  code never imports it.
- ``fake_ur_dashboard.FakeURDashboard`` — serves ``/dashboard_client/stop``
  (``std_srvs/Trigger``) the way ``ur_robot_driver``'s dashboard client
  does, so the UR e-stop's vendor stop is driven through the production
  transport against a real service. Production code never imports it.
"""

from __future__ import annotations
