"""Regression test: ``--no-dashboard`` skips OTLP endpoint forwarding.

Asserts that :func:`sim_e2e.launch.compose_runtime_graph` builds every
spawned node's ``additional_env`` **without** ``OTEL_EXPORTER_OTLP_ENDPOINT``
/ ``OTEL_EXPORTER_OTLP_PROTOCOL`` when ``enable_dashboard=false`` — and
*with* them when ``enable_dashboard=true``.

Why this matters: the OpenTelemetry SDK in
``python/observability/src/openral_observability/_sdk.py:configure_observability``
short-circuits to no-op when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is absent
(``_sdk.py:145``). With it set, every node installs a BatchSpanProcessor /
PeriodicExportingMetricReader / BatchLogRecordProcessor that retries
against the configured endpoint at SIGINT shutdown. Before this guard,
``openral deploy sim --no-dashboard`` always set the endpoint to
``http://127.0.0.1:<dashboard_port>`` even though no dashboard was
running there, so every node blocked for ~30s on connection retries
during teardown. That stalled every headless caller — CI runs, the
``tools/audit_sim_configs.py`` deploy probe, batch scripts — and
manifested as the audit's ``fail-timeout`` status (exit -9, SIGKILL'd
after ``shutdown-grace`` elapsed) on launches that were otherwise
perfectly healthy. The guard at ``sim_e2e.launch.py:433-454`` is what
this test pins.

``OTEL_RESOURCE_ATTRIBUTES`` (ADR-0017 run id / mode / git sha) stays
forwarded under both modes — it is cheap, harmless when no exporter is
wired, and useful if the operator points the parent shell at an
external OTLP collector.

Per CLAUDE.md §1.11: no mocks. Real :class:`openral_core.RobotDescription`
loaded from ``robots/openarm/robot.yaml``; real ``LaunchContext``
exercising the real ``compose_runtime_graph`` opaque function. The
test walks the resulting list of entities and inspects each
``Node``/``LifecycleNode``'s ``additional_env`` directly (private
attribute access matches the pattern in
``test_kernel_params_no_empty_lists.py``).

Run::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    MUJOCO_GL=egl uv run pytest \
        packages/openral_rskill_ros/test/test_no_dashboard_otlp_env.py -v
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

# ── Guards ───────────────────────────────────────────────────────────────────

pytest.importorskip("launch")
pytest.importorskip("launch_ros")
pytest.importorskip("openral_core")
pytest.importorskip("openral_safety")
pytest.importorskip("mujoco")

_ROS2_AVAILABLE = bool(os.environ.get("ROS_DISTRO"))
pytestmark = pytest.mark.skipif(
    not _ROS2_AVAILABLE,
    reason="ROS_DISTRO not set — these tests require a sourced ROS 2 installation.",
)

# parents[3]: test/ → openral_rskill_ros/ → packages/ → <repo-root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCH_FILE = _REPO_ROOT / "packages" / "openral_rskill_ros" / "launch" / "sim_e2e.launch.py"
# openarm is a fixed-base arm with no opt-in extras (no SLAM/Nav2/octomap),
# so the graph it produces is the minimal "kernel + reasoner + prompt_router
# + hal + runtime" set — exactly the surface area we want to audit.
_REPRESENTATIVE_ROBOT = "openarm"


def _import_launch_module() -> Any:
    """Load ``sim_e2e.launch.py`` as a Python module via importlib.

    Duplicated from ``test_kernel_params_no_empty_lists.py`` to keep
    each test file self-contained; consolidation into a shared helper
    can wait until a third test file needs the same scaffolding.
    """
    spec = importlib.util.spec_from_file_location("sim_e2e_launch", _LAUNCH_FILE)
    assert spec is not None and spec.loader is not None, f"failed to spec {_LAUNCH_FILE}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_launch_context(*, enable_dashboard: bool) -> Any:
    """Return a :class:`launch.LaunchContext` pre-populated with the defaults.

    Mirrors the ``DeclareLaunchArgument(default_value=…)`` block at the
    bottom of ``sim_e2e.launch.py`` and pins ``enable_dashboard`` to the
    value the test cares about. Only ``robot_yaml`` / ``hal_*`` lack
    defaults (CLI-required) so we set them by hand to a representative HAL.
    """
    from launch import LaunchContext

    ctx = LaunchContext()
    cfg = ctx.launch_configurations
    cfg["robot_yaml"] = str(_REPO_ROOT / "robots" / _REPRESENTATIVE_ROBOT / "robot.yaml")
    cfg["hal_package"] = "openral_hal_openarm"
    cfg["hal_executable"] = "lifecycle_node.py"
    cfg["hal_node_name"] = "openral_hal_test"
    cfg["hal_params_file"] = "/tmp/openral-test-hal-params.yaml"
    cfg["reset_to_pose_service"] = ""
    cfg["dashboard_port"] = "4318"
    cfg["reasoner_provider"] = "ollama"
    cfg["reasoner_model"] = "gemma4:31b-cloud"
    cfg["spatial_memory_path"] = ""
    cfg["spatial_memory_ingest"] = "false"
    cfg["hal_mode"] = "sim"
    cfg["enable_slam"] = "false"
    cfg["enable_nav2"] = "false"
    cfg["enable_octomap"] = "false"
    cfg["octomap_cloud_topic"] = "/openral/cameras/front_depth/points"
    cfg["enable_object_detector"] = "false"
    cfg["object_detector_onnx"] = str(_REPO_ROOT / "rskills" / "rtdetr-coco-r18" / "model.onnx")
    cfg["enable_dashboard"] = "true" if enable_dashboard else "false"
    return ctx


def _collect_additional_envs(*, enable_dashboard: bool) -> list[tuple[str, dict[str, str]]]:
    """Return ``[(node_package, additional_env), ...]`` for every spawned node.

    Walks the entities ``compose_runtime_graph`` returns, picks the
    ``Node`` / ``LifecycleNode`` instances, and resolves each one's
    ``additional_env`` (a list of ``(key_substitutions, value_substitutions)``
    tuples) via :func:`launch.utilities.perform_substitutions` against
    the launch context — the same call ``ros2 launch`` makes before
    handing env to subprocess.Popen.

    Non-Node entities (ExecuteProcess for the dashboard,
    RegisterEventHandler for lifecycle wiring) are skipped — they
    aren't subject to the OTLP forwarding contract.
    """
    from launch.utilities import perform_substitutions
    from launch_ros.actions import LifecycleNode, Node

    module = _import_launch_module()
    ctx = _make_launch_context(enable_dashboard=enable_dashboard)
    entities = module.compose_runtime_graph(ctx)

    envs: list[tuple[str, dict[str, str]]] = []
    for entity in entities:
        if not isinstance(entity, Node | LifecycleNode):
            continue
        # ``_Node__package`` is the literal package string the launch
        # file passed to the constructor; safer than ``node_name``
        # which raises before the action executes.
        pkg = getattr(entity, "_Node__package", None)
        # ``additional_env`` is a public property exposing the raw
        # substitution pairs; resolve them with the launch context to
        # get a concrete ``{str: str}`` dict.
        raw_env = entity.additional_env
        if pkg is None or raw_env is None:
            continue
        resolved: dict[str, str] = {}
        for key_subs, val_subs in raw_env:
            resolved[perform_substitutions(ctx, key_subs)] = perform_substitutions(ctx, val_subs)
        envs.append((pkg, resolved))
    # Sanity floor: the openarm graph spawns at minimum
    # safety_kernel + reasoner + prompt_router + hal + runtime → 5 nodes.
    assert len(envs) >= 5, (
        f"expected ≥5 nodes in the openarm graph, got {len(envs)}: {[pkg for pkg, _ in envs]}"
    )
    return envs


def test_no_dashboard_omits_otlp_endpoint_from_every_node() -> None:
    """``--no-dashboard`` → every node's additional_env lacks the OTLP endpoint.

    The exact failure this pins: setting ``OTEL_EXPORTER_OTLP_ENDPOINT``
    on every node when nothing is listening on ``dashboard_port`` causes
    BatchSpanProcessor / PeriodicExportingMetricReader to retry against
    the dead port at SIGINT, blocking shutdown for ~30s and stalling
    every headless caller.
    """
    envs = _collect_additional_envs(enable_dashboard=False)
    offenders = {
        pkg: env
        for pkg, env in envs
        if "OTEL_EXPORTER_OTLP_ENDPOINT" in env or "OTEL_EXPORTER_OTLP_PROTOCOL" in env
    }
    assert not offenders, (
        "--no-dashboard nodes must not forward OTEL_EXPORTER_OTLP_ENDPOINT / "
        "OTEL_EXPORTER_OTLP_PROTOCOL (the SDK would install exporters that "
        "block on shutdown retries). Offenders:\n"
        + "\n".join(
            f"  {pkg}: "
            + ", ".join(f"{k}={v!r}" for k, v in env.items() if k.startswith("OTEL_EXPORTER_OTLP_"))
            for pkg, env in offenders.items()
        )
    )


def test_no_dashboard_keeps_otel_resource_attributes() -> None:
    """``--no-dashboard`` → ``OTEL_RESOURCE_ATTRIBUTES`` still forwarded.

    Resource attributes (ADR-0017 run id / mode / git sha) are cheap,
    harmless when no exporter is wired, and useful if the operator
    points the parent shell at an external OTLP collector. Skipping the
    endpoint is the bug fix; skipping the resource attrs would be
    over-zealous.
    """
    envs = _collect_additional_envs(enable_dashboard=False)
    missing = [pkg for pkg, env in envs if "OTEL_RESOURCE_ATTRIBUTES" not in env]
    assert not missing, (
        "OTEL_RESOURCE_ATTRIBUTES must reach every node even under --no-dashboard. "
        f"Missing on: {missing}"
    )


def test_dashboard_enabled_forwards_otlp_endpoint_to_every_node() -> None:
    """Default (``--dashboard``) → every node points OTLP at the dashboard port.

    The symmetric guarantee: when the dashboard IS spawned, every node's
    additional_env carries ``OTEL_EXPORTER_OTLP_ENDPOINT`` =
    ``http://127.0.0.1:<dashboard_port>`` and the HTTP/protobuf protocol
    selector. Pins the contract so a future refactor that accidentally
    silences observability under the default mode is caught.
    """
    envs = _collect_additional_envs(enable_dashboard=True)
    for pkg, env in envs:
        assert env.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "http://127.0.0.1:4318", (
            f"{pkg} must forward OTLP endpoint to the dashboard; got "
            f"{env.get('OTEL_EXPORTER_OTLP_ENDPOINT')!r}"
        )
        assert env.get("OTEL_EXPORTER_OTLP_PROTOCOL") == "http/protobuf", (
            f"{pkg} must forward OTLP protocol selector; got "
            f"{env.get('OTEL_EXPORTER_OTLP_PROTOCOL')!r}"
        )
        assert "OTEL_RESOURCE_ATTRIBUTES" in env, (
            f"{pkg} must forward OTEL_RESOURCE_ATTRIBUTES even under --dashboard"
        )
