"""Fleet guard: every in-tree robot with an MJCF lowers to a *live* collision
FK — i.e. its movable joints get real ``dof_index`` columns, never all ``-1``.

This is the fleet-wide regression for the issue-#77 fix. ``openral deploy sim``
and ``deploy run`` prefer the MJCF-lowered collision model
(``openral_safety.mjcf_lowering.lower_collision_params``); before the fix that
lowering keyed ``dof_index`` by *manifest* joint names but read the *MJCF's*
names, so for every robot whose names differ the whole arm FK froze at the rest
pose and the kernel's geometric collision check became a silent no-op.

For each robot that declares ``assets.mjcf`` and collision geometry:

* If the MJCF has lowerable primitive geometry → the movable (hinge/slide)
  joints must map to a contiguous ``0..k`` prefix of the commanded joint vector
  (capped at ``len(joints)``); **not** all ``-1``.
* If the MJCF is mesh-only → the lowering returns the
  ``{self_collision_enabled: False}`` sentinel, and the launch falls back to the
  manifest collision model (whose ``dof_index`` is built from its own joint
  ordering and was never affected by the bug).

Gate (CLAUDE.md §1.11 / §1.12): mujoco + openral_safety + network/cache access to
resolve each robot's MJCF asset. A robot whose asset can't be resolved on this
host SKIPs (never faked).
"""

from __future__ import annotations

import pathlib

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("openral_safety")

import yaml  # noqa: E402
from openral_core import RobotDescription  # noqa: E402
from openral_safety.mjcf_lowering import lower_collision_params  # noqa: E402

_ROBOTS_DIR = pathlib.Path(__file__).resolve().parents[3] / "robots"


def _has_mjcf_and_collision(path: pathlib.Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return "mjcf" in text and "collision_geometry" in text


_MJCF_ROBOTS = [p for p in sorted(_ROBOTS_DIR.glob("*/robot.yaml")) if _has_mjcf_and_collision(p)]


@pytest.mark.parametrize("manifest", _MJCF_ROBOTS, ids=lambda p: p.parent.name)
def test_mjcf_lowering_yields_live_dof_index(manifest: pathlib.Path) -> None:
    """A robot's MJCF lowers to a live FK (real dof_index) or the disabled sentinel."""
    from openral_core.assets import resolve_asset

    robot = RobotDescription.model_validate(yaml.safe_load(manifest.read_text()))
    if not robot.collision_geometry or robot.assets is None or not robot.assets.mjcf:
        pytest.skip(f"{manifest.parent.name}: no MJCF + collision geometry")
    try:
        mjcf_path = resolve_asset(robot.assets.mjcf, "mjcf", manifest_dir=manifest.parent)
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    except Exception as exc:  # asset unavailable on this host — skip, never fake
        pytest.skip(f"{manifest.parent.name}: MJCF asset unavailable ({exc!r})")

    params = lower_collision_params(model, [j.name for j in robot.joints])

    if not params.get("self_collision_enabled"):
        # Mesh-only MJCF → disabled sentinel; the launch uses the manifest model.
        assert params == {"self_collision_enabled": False}
        return

    dof_index = params["collision_dof_index"]
    joint_kind = params["collision_joint_kind"]
    assert isinstance(dof_index, list) and isinstance(joint_kind, list)

    n_cols = len(robot.joints)
    movable = [i for i, k in enumerate(joint_kind) if k != 0]
    mapped = [dof_index[i] for i in movable]

    assert not all(v < 0 for v in mapped), (
        f"{manifest.parent.name}: every movable joint mapped to -1 — the MJCF FK "
        "is frozen at the rest pose (self-collision is a silent no-op)"
    )
    # The movable joints, in order, take the contiguous 0..k prefix of the
    # commanded joint vector; any beyond n_cols (e.g. a second mimic finger) are -1.
    non_negative = [v for v in mapped if v >= 0]
    assert non_negative == list(range(len(non_negative))), (
        f"{manifest.parent.name}: dof_index not a contiguous joint-ordered prefix: {mapped}"
    )
    assert len(non_negative) == min(len(movable), n_cols)
    assert all(v < n_cols for v in non_negative), "dof_index column past the commanded vector"
