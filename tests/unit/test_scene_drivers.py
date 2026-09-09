# SPDX-License-Identifier: Apache-2.0
"""A deploy scene brings up the vendor drivers its sensor bindings read from.

A `SensorDeployBinding` with a `ros2_*` backend *subscribes* to a topic — it
names the topic but not who publishes it. For the OpenArm cell that publisher is
`zed_wrapper`, and until `DeployScene.drivers` existed an operator had to start
it by hand in a second terminal. Forgetting is not an error: subscribing to an
unpublished topic is perfectly legal, so the camera panel is simply empty and
the octomap/voxel chain downstream of it produces nothing, with every node
reporting healthy.

The driver's own config travels with the scene rather than living in a home
directory, so a committed scene works on more than one machine.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCENE = _ROOT / "scenes" / "deploy" / "openarm_restock_shelf.yaml"
_LAUNCH = _ROOT / "packages" / "openral_rskill_ros" / "launch" / "deploy_e2e.launch.py"


def _scene() -> dict[str, object]:
    doc = yaml.safe_load(_SCENE.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def test_the_scene_declares_the_driver_that_publishes_its_ros2_binding() -> None:
    """Every `ros2_*` sensor binding needs a declared publisher.

    This is the pairing that was missing: the `context` camera reads
    `/zed/zed_node/...` and nothing in the graph produced it.
    """
    doc = _scene()
    drivers = doc.get("drivers") or []
    sensors = doc.get("sensors") or []
    assert isinstance(drivers, list) and isinstance(sensors, list)

    ros2_bound = [
        s
        for s in sensors
        if isinstance(s, dict)
        and str((s.get("deploy_binding") or {}).get("backend", "")).startswith("ros2_")
    ]
    assert ros2_bound, "fixture moved: no ros2_* sensor binding left in this scene"
    assert drivers, (
        f"{[s['name'] for s in ros2_bound]} subscribe to topics this scene declares no "
        "publisher for — deploy run would come up with those cameras silently empty"
    )
    assert any(d.get("package") == "zed_wrapper" for d in drivers)


def test_the_driver_config_travels_with_the_scene() -> None:
    """A committed scene must not carry someone's home directory."""
    doc = _scene()
    for driver in doc.get("drivers") or []:
        for key, value in (driver.get("args") or {}).items():
            if not key.endswith("_path"):
                continue
            assert not str(value).startswith("/"), (
                f"{key}={value} is absolute; a committed scene must reference driver "
                "config relative to itself so it works on more than one host"
            )
            assert (_SCENE.parent / str(value)).is_file(), (
                f"{key}={value} does not resolve against {_SCENE.parent}"
            )


def test_the_zed_override_keeps_positional_tracking_off() -> None:
    """With it on, `zed_camera_link` gets a second TF parent.

    The manifest already parents that frame to `openarm_base`; the wrapper's
    visual odometry would parent it again. A frame with two parents is not a
    tree — lookups go order-dependent and octomap_server drops clouds silently,
    which is the failure this override exists to prevent.
    """
    doc = _scene()
    override = next(
        (d["args"]["ros_params_override_path"] for d in doc["drivers"] if "args" in d),  # type: ignore[index]  # reason: shape asserted above
        None,
    )
    assert override is not None, "the ZED driver must pin its parameter override"
    cfg = yaml.safe_load((_SCENE.parent / str(override)).read_text(encoding="utf-8"))
    params = cfg["/**"]["ros__parameters"]
    assert params["pos_tracking"]["pos_tracking_enabled"] is False


def test_drivers_are_included_on_the_real_path_only() -> None:
    """Sim renders cameras; it never needs a vendor driver."""
    text = _LAUNCH.read_text(encoding="utf-8")
    assert "_build_driver_includes(scene_drivers, deploy_config)" in text
    real_branch = text.split('if hal_mode == "real":', 1)[1].split("\n    urdf_asset", 1)[0]
    assert "_build_driver_includes" in real_branch, (
        "driver includes must sit inside the hal_mode=='real' branch"
    )


def test_relative_path_args_are_resolved_against_the_scene() -> None:
    """The launch must do the resolving; `ros2 launch` gets an absolute path."""
    text = _LAUNCH.read_text(encoding="utf-8")
    assert 'key.endswith("_path")' in text
    assert "pathlib.Path(deploy_config).parent" in text


def test_a_scene_without_drivers_is_valid() -> None:
    """Most workcells open their cameras directly and need no driver."""
    core = pytest.importorskip("openral_core")
    assert core.DeployScene.model_fields["drivers"].default_factory() == []  # type: ignore[misc]  # reason: pydantic FieldInfo


def test_an_unresolvable_driver_package_says_what_to_do() -> None:
    """ "Package not found" alone does not point at the overlay you forgot.

    A vendor driver is normally built into its *own* colcon workspace —
    `zed_wrapper` lives in a `zed_ws`, not in the OpenRAL overlay — so the real
    cause is almost always an unsourced workspace. This bit the first real run
    of the feature.
    """
    text = _LAUNCH.read_text(encoding="utf-8")
    assert "PackageNotFoundError" in text, "the package must be resolved eagerly, not lazily"
    assert "install/setup.bash" in text, "the error must name the remedy"
    assert "ROSConfigError" in text, "a bad scene is a config error (CLAUDE.md §5)"
