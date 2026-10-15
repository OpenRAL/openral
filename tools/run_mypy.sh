#!/usr/bin/env bash
# The one `mypy --strict` surface. `just lint`, the pre-commit hook and the
# quality / release-pypi workflows all call this, so the package list cannot
# drift between them. Every git-tracked .py is checked; the guard at the end
# fails when a new file lands outside every lane below.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# uv workspace members (python/*/src/<pkg>), as packages so intra-package
# imports resolve. Derived from the tree so a new member cannot be missed.
# shellcheck disable=SC2046  # word-split the -p list on purpose
uv run mypy --strict $(git ls-files 'python/*/src/*/__init__.py' | awk -F/ 'NF==5' \
  | cut -d/ -f4 | sort -u | sed 's:^:-p :')
# openral_safety is a colcon package, not a uv workspace member. Checked as a
# package (-p) from its source root, so imports between its own modules are
# resolved and typed; checking it by path names them packages.openral_safety.*
# and every `from openral_safety.X import Y` falls through to Any.
MYPYPATH=packages/openral_safety uv run mypy --strict -p openral_safety
uv run mypy --strict tools/

# The other colcon packages, with every package root on MYPYPATH so cross-package
# imports (reasoner_ros -> openral_world_state_ros, launch files ->
# openral_foxglove_bringup) resolve with real types.
pkgs=$(ls -d packages/*/ | sed 's:/$::' | paste -sd:)
# shellcheck disable=SC2046  # word-split the -p list on purpose
MYPYPATH=$pkgs uv run mypy --strict $(git ls-files 'packages/*/*/__init__.py' \
  | grep -v -e '/test/' -e '^packages/openral_safety/' | cut -d/ -f3 | sort -u | sed 's:^:-p :')
# shellcheck disable=SC2046
MYPYPATH=$pkgs uv run mypy --strict $(git ls-files 'packages/*/launch/*.py' 'cpp/*.py')

# Package-local test dirs: one call each, with the test dir as the deepest root,
# so test modules get the bare names pytest's rootdir-prepend gives them
# (`from _launch_test_common import ...`) and never collide across packages.
for d in $(git ls-files 'packages/*/test/*.py' 'python/*/tests/*.py' | cut -d/ -f1-3 | sort -u); do
  MYPYPATH=$pkgs:$d uv run mypy --strict "$d"
done

# Repo tests (`[mypy-tests.*]` relaxes disallow_untyped_defs), rskill helper
# scripts (bare module names, so their own call) and the top-level scripts.
uv run mypy --strict tests/
# shellcheck disable=SC2046
uv run mypy --strict $(git ls-files 'rskills/*.py')
uv run mypy --strict conftest.py docker/inference/smoke_reasoner.py scripts/

# Coverage guard: every tracked .py must fall in one of the lanes above.
pkg_mods=$(git ls-files 'packages/*/*/__init__.py' | grep -v '/test/' | cut -d/ -f1-3 \
  | sort -u | sed 's:$:/:')
missed=$(git ls-files '*.py' | grep -v -E \
  '^(python/[^/]+/(src/[^/]+|tests)/|packages/[^/]+/(launch|test)/|cpp/|tools/|tests/|rskills/|scripts/)|^(conftest\.py|docker/inference/smoke_reasoner\.py)$' \
  | grep -v -F -f <(echo "$pkg_mods") || true)
if [[ -n "$missed" ]]; then
  echo "run_mypy.sh: these tracked .py files are in no mypy lane — add them:" >&2
  echo "$missed" >&2
  exit 1
fi
