"""Committed vendored URDFs must be machine-portable.

``openral robot vendor-urdf``'s yourdfpy round-trip used to absolutize mesh
``filename`` refs into the vendoring machine's ``robot_descriptions`` cache
(``/home/<user>/.cache/robot_descriptions/…``). Those paths resolve nowhere
else, so on any other host (CI included) collision lowering found no meshes and
silently emitted an empty ACM — the safety kernel would have skipped nothing it
should, but the committed manifests could never be reproduced off the authoring
machine. Vendored URDFs now carry ``rd:<module>:<relpath>`` refs instead; this
guards the whole class: no committed URDF may reference an absolute path.
Real committed files, no mocks (§1.11); pure text, no lowering deps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_URDFS = sorted(Path("robots").glob("*/*.urdf"))


@pytest.mark.parametrize("urdf", _URDFS, ids=lambda p: p.parent.name)
def test_no_absolute_mesh_paths(urdf: Path) -> None:
    """A vendored URDF must not bake in absolute (machine-specific) file paths."""
    offenders = [
        line.strip()
        for line in urdf.read_text(encoding="utf-8").splitlines()
        if 'filename="/' in line
    ]
    assert not offenders, (
        f"{urdf} references absolute paths that only resolve on the vendoring "
        f"machine — re-vendor with `openral robot vendor-urdf` (portable "
        f"rd:<module>:<relpath> refs): {offenders[:3]}"
    )


def test_fleet_is_covered() -> None:
    """Guard against the glob silently matching nothing (e.g. cwd drift)."""
    names = {p.parent.name for p in _URDFS}
    assert {"ur5e", "ur10e", "rizon4"} <= names
