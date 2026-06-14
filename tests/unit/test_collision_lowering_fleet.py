"""Fleet drift guard: every manifest with a collision model matches the tool.

`openral collision lower` owns each robot's `collision_geometry` /
`allowed_collision_pairs`; this asserts the committed manifests have not drifted
from what the tool would regenerate (ADR-0030). Replaces the single hand-pinned
panda test with a parametrized fleet check. Real manifests, no mocks (§1.11);
robots whose URDF/SRDF is unavailable on this host SKIP rather than fake.

Gated on the `[lowering]` group (yourdfpy). On CI runners without it, the whole
module skips — the guard runs on dev hosts and the lowering CI lane.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yourdfpy")

from openral_cli.collision import _lowered_text

_TARGETS = [
    p
    for p in sorted(Path("robots").glob("*/robot.yaml"))
    if "allowed_collision_pairs" in p.read_text(encoding="utf-8")
]


@pytest.mark.parametrize("manifest", _TARGETS, ids=lambda p: p.parent.name)
def test_manifest_matches_lowering_tool(manifest: Path) -> None:
    """A committed collision model must equal what `openral collision lower` emits.

    Robots whose geometry is tool-generated (the ``# GENERATED`` marker) are checked
    in **full** (geometry + ACM + joint FK); robots with hand-tuned geometry
    (panda_mobile) are checked ACM-only so their tuned capsules aren't flagged.
    """
    # Tool-generated GEOMETRY = a `# GENERATED` comment immediately above
    # `collision_geometry:` (panda_mobile's hand geometry has a hand comment there;
    # only its ACM block carries a GENERATED header, so check the geometry block).
    lines = manifest.read_text(encoding="utf-8").splitlines()
    geo_idx = next((i for i, ln in enumerate(lines) if ln.startswith("collision_geometry:")), None)
    tool_generated = (
        geo_idx is not None and geo_idx > 0 and lines[geo_idx - 1].startswith("# GENERATED")
    )
    try:
        current, spliced = _lowered_text(manifest, acm_only=not tool_generated, geometry_only=False)
    except (ValueError, FileNotFoundError) as exc:
        pytest.skip(f"{manifest.parent.name}: {exc}")
    except ImportError as exc:  # MJCF robots (openarm) need mujoco + openral_hal
        pytest.skip(f"{manifest.parent.name}: MJCF lowering deps unavailable ({exc})")
    flag = "" if tool_generated else " --acm-only"
    assert current == spliced, (
        f"{manifest.parent.name} drifted from the lowering tool — "
        f"run `openral collision lower --robot {manifest}{flag} --write`"
    )


def test_panda_mobile_is_covered() -> None:
    """Guard against the parametrization silently dropping panda_mobile."""
    assert any(p.parent.name == "panda_mobile" for p in _TARGETS)
