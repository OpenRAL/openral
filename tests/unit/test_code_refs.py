"""Config-named code (``hal.sim`` / ``hal.real`` / ``composition.composer``) is openral_* only.

A manifest or scene names a ``"module:attribute"`` that OpenRAL imports and calls with
config-supplied kwargs; ``os:system`` there was arbitrary code execution. Each test
loads a real shipped fixture, swaps in a hostile reference and expects a refusal at
schema load AND at the import seam (``build_hal`` / the HAL node's composer call).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from openral_core import (
    DeployScene,
    RobotDescription,
    RosIntegration,
    RSkillManifest,
    SceneComposition,
)
from openral_core.code_refs import (
    ALLOW_EXTERNAL_PLUGINS_ENV,
    check_code_ref,
    check_ros_interface_package,
    import_code_ref,
)
from openral_core.exceptions import ROSConfigError
from openral_rskill.ros_action_rskill import _import_action_or_service
from pydantic import ValidationError

REPO = Path(__file__).resolve().parents[2]
_UR5E = REPO / "robots" / "ur5e" / "robot.yaml"
_OPENARM_SCENE = REPO / "scenes" / "deploy" / "openarm_tabletop.yaml"

_HOSTILE = ["os:system", "subprocess:Popen", "builtins:exec", "pkg.mod:compose"]


@pytest.mark.parametrize("ref", _HOSTILE)
def test_a_reference_outside_openral_is_refused(ref: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOW_EXTERNAL_PLUGINS_ENV, raising=False)
    with pytest.raises(ROSConfigError, match=r"hal\.real .*outside OpenRAL"):
        check_code_ref(ref, setting="hal.real")
    with pytest.raises(ROSConfigError, match="outside OpenRAL"):
        import_code_ref(ref, setting="hal.real")


@pytest.mark.parametrize("ref", ["openral_hal", "openral_hal:", "openral_hal:x;y", "a b:c"])
def test_a_malformed_reference_is_refused(ref: str) -> None:
    with pytest.raises(ROSConfigError, match="malformed"):
        check_code_ref(ref, setting="composition.composer")


def test_openral_and_pro_packages_are_admitted() -> None:
    assert check_code_ref("openral_hal.ur_real:UR5eRealHAL", setting="hal.real")
    # OpenRAL Pro's packages share the prefix; no Pro install is needed to check it.
    assert check_code_ref("openral_pro_trt.hal:X", setting="hal.real")
    # A look-alike top-level module is not an OpenRAL package.
    with pytest.raises(ROSConfigError):
        check_code_ref("openralx.hal:X", setting="hal.real")


def test_the_opt_in_admits_an_external_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOW_EXTERNAL_PLUGINS_ENV, "1")
    assert import_code_ref("json:dumps", setting="hal.sim") is __import__("json").dumps


def _robot_with(hal: dict[str, object]) -> dict[str, object]:
    raw: dict[str, object] = yaml.safe_load(_UR5E.read_text())
    raw["hal"] = {**raw["hal"], **hal}  # type: ignore[dict-item]  # reason: YAML mapping
    return raw


@pytest.mark.parametrize("field", ["sim", "real"])
def test_a_robot_manifest_naming_foreign_code_does_not_load(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ALLOW_EXTERNAL_PLUGINS_ENV, raising=False)
    with pytest.raises(ValidationError, match=rf"hal\.{field} = 'os:system'"):
        RobotDescription.model_validate(_robot_with({field: "os:system"}))


def test_build_hal_rechecks_a_reference_that_skipped_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openral_hal import build_hal

    monkeypatch.delenv(ALLOW_EXTERNAL_PLUGINS_ENV, raising=False)
    description = RobotDescription.from_yaml(str(_UR5E))
    # model_construct-style mutation bypasses the validator; the import seam must not.
    object.__setattr__(description.hal, "real", "os:system")
    with pytest.raises(ROSConfigError, match=r"hal\.real = 'os:system'"):
        build_hal(description, mode="real")


def test_a_scene_composer_naming_foreign_code_does_not_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ALLOW_EXTERNAL_PLUGINS_ENV, raising=False)
    raw = yaml.safe_load(_OPENARM_SCENE.read_text())
    assert raw["composition"]["composer"].startswith("openral_sim.")
    DeployScene.model_validate(raw)  # the shipped scene loads
    raw["composition"]["composer"] = "os:system"
    with pytest.raises(ValidationError, match=r"composition\.composer = 'os:system'"):
        DeployScene.model_validate(raw)
    # The HAL node's wire format (scene_composition_json) is validated the same way.
    with pytest.raises(ValidationError, match="outside OpenRAL"):
        SceneComposition.model_validate_json('{"composer": "os:system", "params": {}}')


# ROSActionRskill imports ``<ros_integration.package>.action``, which runs the package's
# ``__init__`` first: the manifest field is a code reference too.
_MOVEIT_JOINTS = REPO / "rskills" / "rskill-moveit-joints" / "rskill.yaml"


@pytest.mark.parametrize("package", ["os", "antigravity", "openral_core.code_refs", "Moveit_msgs"])
def test_a_ros_integration_package_that_is_not_idl_does_not_load(
    package: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ALLOW_EXTERNAL_PLUGINS_ENV, raising=False)
    raw = yaml.safe_load(_MOVEIT_JOINTS.read_text())
    assert RSkillManifest.model_validate(raw).ros_integration is not None
    raw["ros_integration"]["package"] = package
    with pytest.raises(ValidationError, match=r"ros_integration\.package .*not a ROS interface"):
        RSkillManifest.model_validate(raw)


def test_the_import_seam_rechecks_a_package_that_skipped_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ALLOW_EXTERNAL_PLUGINS_ENV, raising=False)
    raw = yaml.safe_load(_MOVEIT_JOINTS.read_text())["ros_integration"]
    integration = RosIntegration.model_construct(**{**raw, "package": "antigravity"})
    with pytest.raises(ROSConfigError, match="not a ROS interface"):
        _import_action_or_service(integration)


def test_ros_idl_packages_and_the_opt_in_are_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    for package in ("moveit_msgs", "nav2_msgs", "std_srvs", "example_interfaces"):
        assert check_ros_interface_package(package, setting="p") == package
    monkeypatch.setenv(ALLOW_EXTERNAL_PLUGINS_ENV, "1")
    assert check_ros_interface_package("vendor_actions", setting="p") == "vendor_actions"


def test_a_trailing_newline_does_not_pass_the_reference_check() -> None:
    with pytest.raises(ROSConfigError, match="malformed"):
        check_code_ref("openral_hal.ur_real:UR5eRealHAL\n", setting="hal.real")


def test_a_foreign_callable_re_exported_by_an_openral_module_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # backend_registry does ``from importlib import import_module``: an openral_* path to
    # stdlib code that a composer's config kwargs could drive (``name=<any module>``).
    ref = "openral_rskill.backend_registry:import_module"
    monkeypatch.delenv(ALLOW_EXTERNAL_PLUGINS_ENV, raising=False)
    assert check_code_ref(ref, setting="composition.composer") == ref
    with pytest.raises(ROSConfigError, match="defined in 'importlib'"):
        import_code_ref(ref, setting="composition.composer")
    monkeypatch.setenv(ALLOW_EXTERNAL_PLUGINS_ENV, "1")
    assert callable(import_code_ref(ref, setting="composition.composer"))


def test_a_refused_code_ref_is_a_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ValueError subclass, so `except (OSError, ValueError)` skip-loops skip the file."""
    monkeypatch.delenv(ALLOW_EXTERNAL_PLUGINS_ENV, raising=False)
    raw = yaml.safe_load(_MOVEIT_JOINTS.read_text())
    raw["ros_integration"]["package"] = "turtlesim"
    with pytest.raises(ValueError, match="not a ROS interface"):
        RSkillManifest.model_validate(raw)
