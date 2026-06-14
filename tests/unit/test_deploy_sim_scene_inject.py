"""deploy sim forwards the scene YAML to manifest-driven arms in sim mode (ADR-0034).

Uses ``scenes/deploy/libero_pnp.yaml`` — a DeployScene (env-only, no task)
that resolves to ``franka_panda`` via ``SCENES.fixed_robot("libero_spatial")``.
``openral deploy sim --config`` is strict on DeployScene (ADR-0041), so the
fixture must be DeployScene-shaped.

Parents depth: tests/unit/test_*.py → parents[0]=tests/unit, parents[1]=tests,
parents[2]=repo root (matches every other test in this directory, e.g.
test_cli_deploy_sim.py line 38).
"""

from __future__ import annotations

from pathlib import Path

from openral_cli.deploy_sim import resolve_launch_invocation

_REPO = Path(__file__).resolve().parents[2]
_SCENE = _REPO / "scenes/deploy/libero_pnp.yaml"


def test_sim_mode_injects_sim_env_yaml() -> None:
    """Manifest-driven HAL (franka_panda) gets sim_env_yaml in sim mode."""
    assert _SCENE.is_file(), f"missing fixture: {_SCENE}"
    inv = resolve_launch_invocation(
        config=_SCENE,
        robot_override="franka_panda",
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode="sim",
    )
    assert inv.hal_params["sim_env_yaml"] == str(_SCENE.resolve())


def test_real_mode_does_not_inject_scene() -> None:
    """Manifest-driven HAL (franka_panda) does not get sim_env_yaml in real mode."""
    inv = resolve_launch_invocation(
        config=None,
        robot_override="franka_panda",
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_mode="real",
    )
    assert "sim_env_yaml" not in inv.hal_params
