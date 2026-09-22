"""Unit tests for the ``robocasa_navigate.yaml`` VLN deploy example.

``scenes/deploy/robocasa_navigate.yaml`` is documented in
``docs/contributing/toolchain.md``, ``docs/reference/sim-environments.md``,
``docs/tutorials/deploy/deploy-run-and-dashboard.md``, and
``rskills/rskill-internvla_n1-mobile_base-vln-nf4/README.md`` as the VLN
(InternVLA-N1 navigation) deploy example, yet nothing loaded it through the
real CLI resolution path. Exercised via ``resolve_launch_invocation`` +
``--dry-run`` (CLAUDE.md §1.11: no mocks; ``--dry-run`` never shells out to
``ros2 launch``, so no GPU/sidecar is required — see
``test_bh_deploy_sim_dry_run_openarm`` in ``test_cli_deploy_sim.py`` for the
same pattern against a different scene).

Doc-promise gap this file surfaces: the scene is an env-only ``DeployScene``
(no ``task:`` block, no success criterion) — nothing pins InternVLA-N1 as the
dispatched rSkill. The reasoner picks at runtime from the full
capability-matched palette for ``panda_mobile``, which also contains
``rldx1-ft-rc365-nf4`` (a pick-place VLA, not a navigation policy) alongside
the documented InternVLA-N1 and the Nav2 ``ros_action`` wrapper. The docs'
"validated in-tree against panda_mobile in the RoboCasa NavigateKitchen
scene" claim is about capability compatibility, not a guarantee that the
reasoner actually dispatches InternVLA-N1 on this scene.
"""

from __future__ import annotations

from pathlib import Path

from openral_cli.deploy_sim import _capability_matched_manifests, resolve_launch_invocation
from openral_cli.main import app
from openral_core import DeployScene, RobotDescription, RSkillManifest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _REPO_ROOT / "scenes" / "deploy" / "robocasa_navigate.yaml"
_PANDA_MOBILE_YAML = _REPO_ROOT / "robots" / "panda_mobile" / "robot.yaml"
_INTERNVLA_N1_DIR = _REPO_ROOT / "rskills" / "rskill-internvla_n1-mobile_base-vln-nf4"
_INTERNVLA_N1_NAME = "OpenRAL/rskill-internvla_n1-mobile_base-vln-nf4"
_NAV2_NAME = "OpenRAL/rskill-nav2-mobile_base-navigate_to_pose"


def test_scene_fixture_is_env_only_navigate_kitchen() -> None:
    """Real-fixture sanity: NavigateKitchen prebuilt task, no pinned success task."""
    assert _CONFIG.is_file(), f"missing fixture: {_CONFIG}"
    scene = DeployScene.from_yaml(str(_CONFIG))
    assert scene.scene.id == "robocasa/NavigateKitchen"
    assert scene.scene.backend == "mujoco"
    # Env-only: `DeployScene` has no `task`/success-criterion field at all
    # (unlike `SimScene`) — the reasoner picks the active rSkill from the
    # palette at on_configure, per the file's own header comment.
    assert "task" not in type(scene).model_fields


def test_resolve_launch_invocation_facts() -> None:
    """The real resolver, not a hand-built copy, produces the documented facts."""
    invocation = resolve_launch_invocation(
        config=_CONFIG,
        robot_override=None,
        dashboard_port=4318,
        reset_to_pose_service=None,
        hal_param_overrides=None,
    )
    assert invocation.robot_id == "panda_mobile"
    assert invocation.hal.package == "openral_hal_panda_mobile"
    assert invocation.hal.node_name == "openral_hal_panda_mobile"
    # panda_mobile.robot.yaml declares has_lidar: true, so SLAM + Nav2
    # auto-enable and the backend is the lidar leg, not visual SLAM.
    assert invocation.enable_slam is True
    assert invocation.slam_backend == "lidar"
    assert invocation.enable_nav2 is True
    assert invocation.enable_octomap is True
    assert invocation.clock_origin == "simulation"

    joined = " ".join(invocation.argv_template)
    assert "enable_slam:=true" in joined
    assert "slam_backend:=lidar" in joined
    assert "enable_nav2:=true" in joined
    assert "enable_octomap:=true" in joined
    assert "clock_origin:=simulation" in joined


def test_dry_run_cli_reports_nav_stack_enabled() -> None:
    """``openral deploy sim --config … --dry-run`` needs no GPU/sidecar/ROS."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["deploy", "sim", "--config", str(_CONFIG), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "robot=panda_mobile" in flat
    assert "hal_package=openral_hal_panda_mobile" in flat
    assert "slam: enabled" in flat
    assert "nav2: enabled" in flat
    # The dry-run summary names the exact Nav2 rSkill the docs promise.
    assert "OpenRAL/rskill-nav2-mobile_base-navigate_to_pose" in result.output
    assert "octomap: enabled" in flat
    assert "clock_origin:=simulation" in flat


def test_capability_matched_palette_includes_documented_internvla_n1() -> None:
    """The InternVLA-N1 VLN rSkill resolves and is in the panda_mobile palette.

    This is the concrete fact behind the docs' repeated claim that this scene
    is "the" InternVLA-N1 deploy example: the rSkill reference in
    ``docs/reference/sim-environments.md`` and the InternVLA-N1 README must
    actually resolve to the in-tree manifest, and the manifest's
    capabilities (``mobile_base`` + ``body_twist`` + the ``observation.images.head``
    RGB sensor) must actually capability-match ``panda_mobile``.
    """
    assert _PANDA_MOBILE_YAML.is_file(), f"missing fixture: {_PANDA_MOBILE_YAML}"
    description = RobotDescription.from_yaml(str(_PANDA_MOBILE_YAML))
    matched = _capability_matched_manifests(_REPO_ROOT, description)
    matched_names = {m.name for m in matched}

    assert _INTERNVLA_N1_NAME in matched_names
    internvla_n1 = next(m for m in matched if m.name == _INTERNVLA_N1_NAME)
    assert internvla_n1.kind == "vla"
    assert "mobile_base" in internvla_n1.embodiment_tags
    assert any(
        s.vla_feature_key == "observation.images.head" for s in internvla_n1.sensors_required
    )
    # The matched manifest comes straight from this in-tree directory — the
    # exact reference every doc that names this scene points at.
    assert (_INTERNVLA_N1_DIR / "rskill.yaml").is_file()
    assert (
        RSkillManifest.from_yaml(str(_INTERNVLA_N1_DIR / "rskill.yaml")).name == _INTERNVLA_N1_NAME
    )

    # Doc-promise gap: the palette is NOT InternVLA-N1-exclusive. The Nav2
    # wrapper rSkill also matches (as it does for every mobile_base robot)...
    assert _NAV2_NAME in matched_names
    # ...and so does a pick-place VLA fine-tuned for a RoboCasa task
    # (rldx1-ft-rc365-nf4, embodiment tag ``panda_mobile``), since this scene
    # declares no task/success criterion that would exclude it. Nothing in
    # the scene or the resolver pins InternVLA-N1 as THE skill the reasoner
    # dispatches on this scene; that choice is made at runtime.
    assert "OpenRAL/rskill-rldx1_ft-panda_mobile-robocasa-nf4" in matched_names
    assert len(matched) > 1, (
        "fixture drift: expected more than just InternVLA-N1 in the "
        "panda_mobile palette — re-check the doc-promise gap this test guards"
    )
