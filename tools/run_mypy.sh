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
uv run mypy --strict tools/
