#!/usr/bin/env bash
# Apply this directory's upstream patches to a checkout of openarm_description.
#
# Idempotent: an already-applied patch is reported and skipped, so this is safe
# to run from a provisioning script or by hand after every `git pull` upstream.
set -euo pipefail

target="${1:-}"
if [[ -z "$target" || ! -d "$target/.git" ]]; then
    echo "usage: $0 <path to openarm_description git checkout>" >&2
    exit 2
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
status=0

for patch in "$here"/*.patch; do
    name="$(basename "$patch")"
    if git -C "$target" apply --reverse --check "$patch" >/dev/null 2>&1; then
        echo "already applied: $name"
        continue
    fi
    if git -C "$target" apply --check "$patch" >/dev/null 2>&1; then
        git -C "$target" apply "$patch"
        echo "applied:         $name"
    else
        # Upstream moved under the patch. Do not force it — a half-applied
        # xacro would fail at launch, far from here.
        echo "FAILED:          $name (upstream has changed; re-base or drop it)" >&2
        status=1
    fi
done

exit "$status"
