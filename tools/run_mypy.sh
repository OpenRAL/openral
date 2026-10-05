#!/usr/bin/env bash
# The one `mypy --strict` surface. `just lint`, the pre-commit hook and the
# quality / release-pypi workflows all call this, so the package list cannot
# drift between them.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run mypy --strict \
  -p openral_core \
  -p openral_cli \
  -p openral_sim \
  -p openral_observability \
  -p openral_runner \
  -p openral_reasoner \
  -p openral_hal
# openral_safety is a colcon package, not a uv workspace member. Checked as a
# package (-p) from its source root, so imports between its own modules are
# resolved and typed; checking it by path names them packages.openral_safety.*
# and every `from openral_safety.X import Y` falls through to Any.
MYPYPATH=packages/openral_safety uv run mypy --strict -p openral_safety
uv run mypy --strict tools/
